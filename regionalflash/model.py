"""RegionalFlash JiT-3D DiT model (2 timestamps: 1 conditional context + 1 noisy target).

Adapted from FlashEdges `JiT3D_Modern`:
  - Modern DiT trunk: RMSNorm, SwiGLU, QK-Norm, 3D Axial RoPE, SDPA FlashAttention
  - Low-rank AdaLN-Zero conditioning (`gate_init=0.0`, `cond_additive=False` for from-scratch)
  - Prefix causal attention (`prefix_attn=True`): frame 0 (known context) attends only to itself,
    frame 1 (noisy target) attends to both frame 0 and frame 1 -> zero target leakage into context
  - Split dual decoder heads:
      1. `head_grid`: dense atmospheric column + satellite + radar (69 ch)
      2. `head_metar`: dedicated conv-refined sparse airport head (6 ch: tmpc, dwpc, mslp, u10, v10, p01m)
         so sparse station supervision never introduces blocky 8x8 artifacts into dense fields.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config

SAT_WAVELENGTHS_NM = [
    640.0, 863.0, 1610.0,
    3830.0, 6200.0, 7330.0, 8590.0, 9620.0, 11200.0, 12400.0, 13300.0,
]


# ==============================================================================
# 1. Modern DiT Building Blocks (RMSNorm, SwiGLU, Low-Rank AdaLN-Zero, 3D RoPE)
# ==============================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int | None = None):
        super().__init__()
        out_features = out_features or in_features
        self.w1 = nn.Linear(in_features, hidden_features, bias=False)
        self.w2 = nn.Linear(in_features, hidden_features, bias=False)
        self.w3 = nn.Linear(hidden_features, out_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class AdaLNModulation(nn.Module):
    """Low-rank AdaLN-Zero modulation head."""

    def __init__(self, dim: int, n_params: int, rank: int = 128, gate_init: float = 0.0, gate_idx: tuple[int, ...] = ()):
        super().__init__()
        self.dim = dim
        self.n_params = int(n_params)
        self.gate_init = float(gate_init)
        self.gate_idx = tuple(int(i) for i in gate_idx)
        self.silu = nn.SiLU()
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, self.n_params * dim)

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        return self.up(self.silu(self.down(c))).view(-1, self.n_params, self.dim)


def _precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    return torch.polar(torch.ones_like(freqs), freqs)


def _apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, 1, *freqs_cis.shape)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class RoPE3D(nn.Module):
    def __init__(self, head_dim: int, max_t: int, max_h: int, max_w: int, base: float = 10000.0):
        super().__init__()
        chunk = head_dim // 3
        self.d_t = (chunk // 2) * 2
        self.d_h = (chunk // 2) * 2
        self.d_w = head_dim - (self.d_t + self.d_h)
        assert self.d_w % 2 == 0
        self.register_buffer("freqs_t", _precompute_freqs_cis(self.d_t, max_t, base), persistent=False)
        self.register_buffer("freqs_h", _precompute_freqs_cis(self.d_h, max_h, base), persistent=False)
        self.register_buffer("freqs_w", _precompute_freqs_cis(self.d_w, max_w, base), persistent=False)

    def forward(self, xq: torch.Tensor, xk: torch.Tensor, T: int, H: int, W: int) -> tuple[torch.Tensor, torch.Tensor]:
        q_t, q_h, q_w = torch.split(xq, [self.d_t, self.d_h, self.d_w], dim=-1)
        k_t, k_h, k_w = torch.split(xk, [self.d_t, self.d_h, self.d_w], dim=-1)
        f_t = self.freqs_t[:T].view(T, 1, 1, -1).expand(T, H, W, -1).flatten(0, 2)
        f_h = self.freqs_h[:H].view(1, H, 1, -1).expand(T, H, W, -1).flatten(0, 2)
        f_w = self.freqs_w[:W].view(1, 1, W, -1).expand(T, H, W, -1).flatten(0, 2)
        q_t, k_t = _apply_rotary_emb(q_t, k_t, f_t)
        q_h, k_h = _apply_rotary_emb(q_h, k_h, f_h)
        q_w, k_w = _apply_rotary_emb(q_w, k_w, f_w)
        return torch.cat([q_t, q_h, q_w], dim=-1), torch.cat([k_t, k_h, k_w], dim=-1)


# ==============================================================================
# 2. Transformer Block with Prefix Attention & Hyperspherical Context KV Noise
# ==============================================================================

class JiTAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        n_ctx_tokens: int,
        seq_len: int,
        prefix_attn: bool = False,
        kv_ctx_noise: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.n_ctx_tokens = n_ctx_tokens
        self.prefix_attn = prefix_attn
        self.kv_ctx_noise = float(kv_ctx_noise)

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

        if prefix_attn and 0 < n_ctx_tokens < seq_len:
            qi = torch.arange(seq_len).view(seq_len, 1)
            ki = torch.arange(seq_len).view(1, seq_len)
            allow = (ki < n_ctx_tokens) | (qi >= n_ctx_tokens)
            mask = torch.where(allow, 0.0, torch.finfo(torch.float32).min)
        else:
            mask = None
        self.register_buffer("_prefix_mask", mask, persistent=False)

    def _noise_ctx_kv(self, t: torch.Tensor, scale: torch.Tensor | None) -> torch.Tensor:
        if scale is None or not self.training or self.n_ctx_tokens <= 0:
            return t
        nc = self.n_ctx_tokens
        ctx = t[..., :nc, :]
        r = ctx.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        sb = scale.clamp(min=0.0).view(-1, 1, 1, 1)
        y = ctx + (sb / math.sqrt(self.head_dim)) * r * torch.randn_like(ctx)
        y = y * (r / y.norm(dim=-1, keepdim=True).clamp(min=1e-6))
        return torch.cat([y, t[..., nc:, :]], dim=2)

    def forward(
        self,
        x: torch.Tensor,
        rope: RoPE3D,
        T: int,
        H: int,
        W: int,
        ctx_noise_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = self.q_norm(qkv[0]), self.k_norm(qkv[1]), qkv[2]
        k = self._noise_ctx_kv(k, ctx_noise_scale)
        v = self._noise_ctx_kv(v, ctx_noise_scale)
        q, k = rope(q, k, T, H, W)

        if self._prefix_mask is not None and self._prefix_mask.shape[0] == N:
            x = F.scaled_dot_product_attention(q, k, v, attn_mask=self._prefix_mask.to(q.dtype))
        else:
            x = F.scaled_dot_product_attention(q, k, v)
        return self.proj(x.transpose(1, 2).reshape(B, N, C))


class JiTBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        n_ctx_tokens: int,
        seq_len: int,
        mlp_ratio: float = 2.6,
        prefix_attn: bool = False,
        kv_ctx_noise: float = 0.0,
        adaln_rank: int = 128,
        adaln_gate_init: float = 0.0,
    ):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = JiTAttention(
            dim=dim,
            num_heads=num_heads,
            n_ctx_tokens=n_ctx_tokens,
            seq_len=seq_len,
            prefix_attn=prefix_attn,
            kv_ctx_noise=kv_ctx_noise,
        )
        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim, int(dim * mlp_ratio), dim)
        self.adaln = AdaLNModulation(dim, n_params=6, rank=adaln_rank, gate_init=adaln_gate_init, gate_idx=(2, 5))

    def forward(
        self,
        x: torch.Tensor,
        rope: RoPE3D,
        T: int,
        H: int,
        W: int,
        c: torch.Tensor,
        ctx_noise_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.adaln(c).unsqueeze(1).unbind(dim=2)
        h = self.norm1(x) * (1.0 + scale_a) + shift_a
        x = x + gate_a * self.attn(h, rope, T, H, W, ctx_noise_scale)
        h = self.norm2(x) * (1.0 + scale_m) + shift_m
        x = x + gate_m * self.mlp(h)
        return x


# ==============================================================================
# 3. Physics-Structured Atmospheric Patch Embedding & Dual Output Heads
#    (Inspired by Pangu-Weather 3D vertical convs, Aurora/Stormer variable+level
#     embeddings, and NVIDIA STRATA structured vertical decoding)
# ==============================================================================

def _sinusoidal_1d_emb(values: list[float], dim: int) -> torch.Tensor:
    """Deterministic sinusoidal embedding over log-scaled physical coordinates (pressure / wavelength)."""
    v = torch.tensor(values, dtype=torch.float32)
    v = torch.log(v.clamp(min=1e-3))
    v = (v - v.mean()) / v.std().clamp(min=1e-6)  # standardize to ~[-2, 2]
    half = dim // 2
    freqs = torch.exp(-math.log(100.0) * torch.arange(half, dtype=torch.float32) / max(half - 1, 1))
    args = v[:, None] * freqs[None, :] * math.pi
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb  # (N, dim)


class AtmosphericPatchEmbed(nn.Module):
    """Physics-aware multi-modal patch tokenizer for `(B, 84, 2, H, W)` inputs.

    Exploits shared physical structure across channels without expanding DiT token count:
      1. **3D Upper-Air (`50` ch = 5 vars x 10 pressure levels):**
         Reshaped to `(B*T, 5, L=10, H, W)` and tokenized by a shared 3D vertical-spatial
         `Conv3d(5 -> d_lvl, kernel=(3, ph, pw), stride=(1, ph, pw))` so adjacent pressure levels
         share filters and directly compute vertical shear & lapse rates, modulated by a
         sinusoidal log-pressure embedding `E_level(log(p_hPa))` and aggregated across `L=10`
         into the column token via learned vertical attention pooling.
      2. **Factorized Variable-Family Embeddings (Surface CERRA + METAR):**
         Physical variables share semantic family embeddings (`TEMP`, `HUMID`, `U_WIND`,
         `V_WIND`, `PRESS`, `CLOUD`, `PRECIP`) + modality embeddings (`CERRA_SFC` vs `METAR`).
      3. **Spectral SatWeave (`11` bands + mask):**
         Each band is tokenized with a shared spatial filter bank modulated by its sinusoidal
         log-wavelength embedding `E_lambda(log(nm))` so neighboring IR/WV bands share structure.
      4. **OPERA Radar (`2` ch) & Static Terrain (`6` ch):**
         Dedicated spatial patch embeddings fused with all branches via RMSNorm + linear projection.
    """

    # Shared physical variable families
    VAR_FAMILIES = ("temp", "humid", "u_wind", "v_wind", "press", "cloud", "precip")

    def __init__(
        self,
        patch_size: tuple[int, int, int],
        embed_dim: int,
        pressure_levels: list[int] | None = None,
        sat_wavelengths_nm: list[float] | None = None,
    ):
        super().__init__()
        self.patch_size = patch_size
        _, ph, pw = patch_size
        self.ph, self.pw = ph, pw
        self.levels = pressure_levels or list(config.PRESSURE_LEVELS_HPA)
        self.n_levels = len(self.levels)  # 10
        self.wavelengths = sat_wavelengths_nm or list(SAT_WAVELENGTHS_NM)
        self.n_sat = len(self.wavelengths)  # 11

        d_air = embed_dim // 2
        d_sub = embed_dim // 4

        # --- 1. 3D Upper-Air: (5 vars, L=10, H, W) -> vertical-spatial Conv3d + log-pressure emb ---
        self.air_conv3d = nn.Conv3d(5, d_air, kernel_size=(3, ph, pw), stride=(1, ph, pw), padding=(1, 0, 0))
        self.register_buffer("level_emb", _sinusoidal_1d_emb([float(p) for p in self.levels], d_air), persistent=False)
        self.level_proj = nn.Linear(d_air, d_air)
        # Learned vertical attention pooling over L=10 pressure levels -> 1 rich column vector
        self.air_attn_score = nn.Linear(d_air, 1)
        self.air_out = nn.Linear(d_air, d_air)

        # --- 2. Shared Physical Variable-Family Embeddings (Surface CERRA + METAR) ---
        self.var_family_emb = nn.Embedding(len(self.VAR_FAMILIES), d_sub)
        # Surface CERRA (7 ch): [t2m, r2, u10, v10, msl_anom, sp_anom, tcc]
        self.register_buffer("sfc_var_ids", torch.tensor([0, 1, 2, 3, 4, 4, 5], dtype=torch.long), persistent=False)
        # METAR (6 ch): [tmpc, dwpc, mslp_anom, u10, v10, p01m]
        self.register_buffer("metar_var_ids", torch.tensor([0, 1, 4, 2, 3, 6], dtype=torch.long), persistent=False)

        # Per-channel shared spatial patch projector (1 ch -> d_sub) modulated by variable-family emb
        self.sfc_patch = nn.Conv2d(1, d_sub, kernel_size=(ph, pw), stride=(ph, pw))
        self.sfc_mod_emb = nn.Parameter(torch.zeros(1, 1, d_sub))
        self.metar_patch = nn.Conv2d(1, d_sub, kernel_size=(ph, pw), stride=(ph, pw))
        self.metar_mask_patch = nn.Conv2d(1, d_sub, kernel_size=(ph, pw), stride=(ph, pw))
        self.metar_mod_emb = nn.Parameter(torch.zeros(1, 1, d_sub))

        # --- 3. Spectral SatWeave (11 bands + mask) with sinusoidal log-wavelength embedding ---
        self.sat_patch = nn.Conv2d(1, d_sub, kernel_size=(ph, pw), stride=(ph, pw))
        self.sat_mask_patch = nn.Conv2d(1, d_sub, kernel_size=(ph, pw), stride=(ph, pw))
        self.register_buffer("wave_emb", _sinusoidal_1d_emb(self.wavelengths, d_sub), persistent=False)
        self.wave_proj = nn.Linear(d_sub, d_sub)

        # --- 4. Radar (1 dbzh + 1 mask) & Static (6 ch) ---
        self.radar_patch = nn.Conv2d(2, d_sub // 2, kernel_size=(ph, pw), stride=(ph, pw))
        self.static_patch = nn.Conv2d(config.N_STATIC_CHANNELS, d_sub // 2, kernel_size=(ph, pw), stride=(ph, pw))

        # --- 5. Fusion into DiT `embed_dim` ---
        fused_dim = d_air + d_sub + d_sub + d_sub + (d_sub // 2) + (d_sub // 2)
        self.fuse_norm = RMSNorm(fused_dim)
        self.fuse_proj = nn.Linear(fused_dim, embed_dim)

    def _embed_grouped_2d(
        self,
        x_ch: torch.Tensor,
        conv: nn.Conv2d,
        ch_emb: torch.Tensor,
        mod_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply shared spatial conv to each channel independently, add physical emb, mean-pool."""
        BT, C, H, W = x_ch.shape
        tok = conv(x_ch.reshape(BT * C, 1, H, W))           # (BT*C, D, Hp, Wp)
        D, Hp, Wp = tok.shape[1], tok.shape[2], tok.shape[3]
        tok = tok.reshape(BT, C, D, Hp * Wp).permute(0, 3, 1, 2)  # (BT, N, C, D)
        tok = tok + ch_emb.view(1, 1, C, D)
        if mod_bias is not None:
            tok = tok + mod_bias.view(1, 1, 1, D)
        return tok.mean(dim=2)  # (BT, N, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 84, T=2, H, W) -> tokens: (B, T * Hp * Wp, embed_dim)."""
        B, _, T, H, W = x.shape
        BT = B * T
        xf = x.permute(0, 2, 1, 3, 4).reshape(BT, -1, H, W)

        # Slice structured channel layout:
        # [0:50] CERRA upper-air (5 vars x 10 levels), [50:57] CERRA surface (7),
        # [57:68] SatWeave (11), [68:69] Radar dbzh (1), [69:75] METAR (6),
        # [75:76] sat_mask, [76:77] radar_mask, [77:78] metar_mask, [78:84] static (6)
        air = xf[:, :50].reshape(BT, 5, self.n_levels, H, W)
        sfc = xf[:, 50:57]
        sat = xf[:, 57:68]
        rad_dbz = xf[:, 68:69]
        metar = xf[:, 69:75]
        sat_m = xf[:, 75:76]
        rad_m = xf[:, 76:77]
        met_m = xf[:, 77:78]
        static = xf[:, 78:84]

        # 1. 3D Upper-Air vertical-spatial tokens + log-pressure embedding + vertical attention pool
        air_feat = self.air_conv3d(air)  # (BT, d_air, L, Hp, Wp)
        d_air, L, Hp, Wp = air_feat.shape[1], air_feat.shape[2], air_feat.shape[3], air_feat.shape[4]
        N = Hp * Wp
        air_tok = air_feat.reshape(BT, d_air, L, N).permute(0, 3, 2, 1)  # (BT, N, L, d_air)
        lvl_e = self.level_proj(self.level_emb.to(air_tok.dtype))        # (L, d_air)
        air_tok = air_tok + lvl_e.view(1, 1, L, d_air)
        attn_w = torch.softmax(self.air_attn_score(air_tok), dim=2)      # (BT, N, L, 1)
        tok_air = self.air_out((air_tok * attn_w).sum(dim=2))            # (BT, N, d_air)

        # 2. CERRA surface & METAR sharing physical variable-family embeddings
        sfc_emb = self.var_family_emb(self.sfc_var_ids)
        tok_sfc = self._embed_grouped_2d(sfc, self.sfc_patch, sfc_emb, self.sfc_mod_emb)

        met_emb = self.var_family_emb(self.metar_var_ids)
        tok_met = self._embed_grouped_2d(metar, self.metar_patch, met_emb, self.metar_mod_emb)
        tok_met = tok_met + self.metar_mask_patch(met_m).flatten(2).transpose(1, 2)

        # 3. SatWeave with sinusoidal log-wavelength embeddings
        wave_e = self.wave_proj(self.wave_emb.to(xf.dtype))
        tok_sat = self._embed_grouped_2d(sat, self.sat_patch, wave_e)
        tok_sat = tok_sat + self.sat_mask_patch(sat_m).flatten(2).transpose(1, 2)

        # 4. Radar & Static
        tok_rad = self.radar_patch(torch.cat([rad_dbz, rad_m], dim=1)).flatten(2).transpose(1, 2)
        tok_stat = self.static_patch(static).flatten(2).transpose(1, 2)

        # 5. Fuse all structured modality tokens
        fused = torch.cat([tok_air, tok_sfc, tok_sat, tok_met, tok_rad, tok_stat], dim=-1)
        tokens = self.fuse_proj(self.fuse_norm(fused))  # (BT, N, embed_dim)
        return tokens.reshape(B, T * N, -1)


class FinalLayer(nn.Module):
    def __init__(self, patch_size: tuple[int, int, int], out_channels: int, embed_dim: int):
        super().__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.patch_dim = out_channels * patch_size[0] * patch_size[1] * patch_size[2]
        self.linear = nn.Linear(embed_dim, self.patch_dim)

    def forward(self, x: torch.Tensor, T: int, H: int, W: int) -> torch.Tensor:
        pt, ph, pw = self.patch_size
        Tp, Hp, Wp = T // pt, H // ph, W // pw
        x = self.linear(x).view(x.shape[0], Tp, Hp, Wp, self.out_channels, pt, ph, pw)
        x = x.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
        return x.view(x.shape[0], self.out_channels, T, H, W)


class StructuredGridDecoder(nn.Module):
    """Physics-structured decoder for the 69 dense gridded channels (`50` upper-air + `19` 2D).

    Decodes the 10 pressure levels with a shared per-level head modulated by sinusoidal
    log-pressure embeddings + a 1D vertical conv (`k=3`) across pressure levels to enforce
    hydrostatic & vertical continuity, while decoding surface/sat/radar with a dedicated head.
    """

    def __init__(
        self,
        patch_size: tuple[int, int, int],
        embed_dim: int,
        n_levels: int = 10,
        n_air_vars: int = 5,
        n_2d_channels: int = 19,  # 7 CERRA sfc + 11 SatWeave + 1 Radar dbzh
        pressure_levels: list[int] | None = None,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.n_levels = n_levels
        self.n_air_vars = n_air_vars
        self.n_2d_channels = n_2d_channels
        _, ph, pw = patch_size
        self.ph, self.pw = ph, pw

        levels = pressure_levels or list(config.PRESSURE_LEVELS_HPA)
        d_lvl = embed_dim // 2

        self.air_in = nn.Linear(embed_dim, d_lvl)
        self.register_buffer("level_emb", _sinusoidal_1d_emb([float(p) for p in levels], d_lvl), persistent=False)
        self.level_proj = nn.Linear(d_lvl, d_lvl)
        self.vert_conv = nn.Conv1d(d_lvl, d_lvl, kernel_size=3, padding=1)
        self.air_out = nn.Linear(d_lvl, n_air_vars * ph * pw)

        self.head_2d = FinalLayer(patch_size, n_2d_channels, embed_dim)

    def forward(self, target_tokens: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """target_tokens: (B, Hp*Wp, D) -> (B, 69, H, W)."""
        B, N, _ = target_tokens.shape
        Hp, Wp = H // self.ph, W // self.pw

        # 1. Structured 3D upper-air decode across L=10 pressure levels
        h = self.air_in(target_tokens)  # (B, N, d_lvl)
        lvl_e = self.level_proj(self.level_emb.to(h.dtype))  # (L, d_lvl)
        h_lvl = h.unsqueeze(2) + lvl_e.view(1, 1, self.n_levels, -1)  # (B, N, L, d_lvl)
        # Vertical 1D conv across neighboring pressure levels (1000 <-> 925 <-> ... <-> 100 hPa)
        h_v = h_lvl.reshape(B * N, self.n_levels, -1).transpose(1, 2)  # (B*N, d_lvl, L)
        h_lvl = h_lvl + F.silu(self.vert_conv(h_v)).transpose(1, 2).reshape(B, N, self.n_levels, -1)

        air_patches = self.air_out(h_lvl)  # (B, N, L, 5 * ph * pw)
        air_patches = air_patches.view(B, Hp, Wp, self.n_levels, self.n_air_vars, self.ph, self.pw)
        # Reorder to canonical channel order: [var0_lvl0..var0_lvl9, var1_lvl0..var4_lvl9]
        air_grid = air_patches.permute(0, 4, 3, 1, 5, 2, 6).contiguous().view(B, self.n_air_vars * self.n_levels, H, W)

        # 2. 2D surface + sat + radar decode
        grid_2d = self.head_2d(target_tokens, T=1, H=H, W=W).squeeze(2)  # (B, 19, H, W)
        return torch.cat([air_grid, grid_2d], dim=1)  # (B, 69, H, W)


class MetarRefinementHead(nn.Module):
    """Dedicated 2D conv refinement head for sparse METAR airport fields on the target frame.

    Unpatchifies trunk tokens for the target frame (t=1), fuses with context METAR + static
    terrain features, and applies 2 residual 3x3 conv layers at full 5.5 km resolution so
    sparse airport observations propagate smoothly across terrain without creating 8x8 blocks.
    """

    def __init__(
        self,
        patch_size: tuple[int, int, int],
        embed_dim: int,
        metar_channels: int,
        static_channels: int,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.unpatch = FinalLayer(patch_size, hidden_dim, embed_dim)
        in_ch = hidden_dim + metar_channels * 2 + 1 + static_channels  # trunk + ctx_metar + noisy_metar + ctx_mask + static
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, metar_channels, kernel_size=1),
        )

    def forward(
        self,
        target_tokens: torch.Tensor,
        ctx_metar: torch.Tensor,
        noisy_metar: torch.Tensor,
        ctx_metar_mask: torch.Tensor,
        static: torch.Tensor,
    ) -> torch.Tensor:
        H, W = static.shape[-2], static.shape[-1]
        feat = self.unpatch(target_tokens, T=1, H=H, W=W).squeeze(2)  # (B, hidden, H, W)
        x = torch.cat([feat, ctx_metar, noisy_metar, ctx_metar_mask, static], dim=1)
        return self.conv(x)


# ==============================================================================
# 4. Full RegionalFlash JiT-3D Model (T = 2: 1 Conditional + 1 Noisy Target)
# ==============================================================================

class RegionalFlashJiT3D(nn.Module):
    """2-frame JiT-3D DiT for RegionalFlash (CERRA + SatWeave + OPERA Radar + METAR).

    Input tensor shape: `(B, C_in, 2, H, W)` where:
      - `t = 0`: clean known state `[CERRA(t), Sat(t+2h), Radar(t+2h), METAR(t+2h), masks, static]`
      - `t = 1`: noisy target state `[noisy_CERRA(t+3h), noisy_Sat(t+5h), noisy_Radar(t+5h),
                                      noisy_METAR(t+5h), known_masks, static]`

    Outputs (for target frame `t = 1` only, x-prediction):
      - `pred_grid`:  `(B, C_grid_out, H, W)`  -> `[CERRA(57), Sat(11), Radar_dbzh(1)]` = 69 ch
      - `pred_metar`: `(B, C_metar_out, H, W)` -> `[tmpc, dwpc, mslp_anom, u10, v10, p01m]` = 6 ch
    """

    def __init__(
        self,
        c_grid_out: int = 69,      # 57 CERRA + 11 SatWeave + 1 OPERA dbzh
        c_metar_out: int = 6,      # tmpc, dwpc, mslp_anom_hpa, u10, v10, p01m
        c_cond_extra: int = 9,     # 1 sat_mask + 1 radar_mask + 1 metar_mask + 6 static
        c_static: int = 6,
        img_size: tuple[int, int] = (128, 128),
        patch_size: tuple[int, int, int] = (1, 8, 8),
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        global_cond_dim: int = 15,  # [time(4), lat/lon(2), solar_ctx+tgt+noon(8), t_flow(1)]
        time_emb_dim: int = 64,
        adaln_rank: int = 128,
        prefix_attn: bool = False,
        kv_ctx_noise: float = 0.0,
        metar_head_dim: int = 64,
        isolate_metar_grad: bool = True,
    ):
        super().__init__()
        self.c_grid_out = c_grid_out
        self.c_metar_out = c_metar_out
        self.c_in = c_grid_out + c_metar_out + c_cond_extra
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.time_emb_dim = time_emb_dim
        self.kv_ctx_noise = float(kv_ctx_noise)
        self.isolate_metar_grad = bool(isolate_metar_grad)

        H, W = img_size
        self.grid_t = 2
        self.grid_h = H // patch_size[1]
        self.grid_w = W // patch_size[2]
        self.tokens_per_frame = self.grid_h * self.grid_w
        seq_len = self.grid_t * self.tokens_per_frame

        self.patch_embed = AtmosphericPatchEmbed(patch_size, embed_dim)
        self.rope = RoPE3D(embed_dim // num_heads, max_t=4, max_h=self.grid_h * 2, max_w=self.grid_w * 2)

        # Global conditioning: [scalars (global_cond_dim - 1), sin/cos_time(time_emb_dim)]
        self.context_mlp = nn.Sequential(
            nn.Linear((global_cond_dim - 1) + time_emb_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        self.blocks = nn.ModuleList([
            JiTBlock(
                dim=embed_dim,
                num_heads=num_heads,
                n_ctx_tokens=self.tokens_per_frame,
                seq_len=seq_len,
                mlp_ratio=2.6,
                prefix_attn=prefix_attn,
                kv_ctx_noise=kv_ctx_noise,
                adaln_rank=adaln_rank,
                adaln_gate_init=0.0,  # canonical AdaLN-Zero for training from scratch
            )
            for _ in range(depth)
        ])

        self.norm_final = RMSNorm(embed_dim)
        self.adaLN_final = AdaLNModulation(embed_dim, n_params=2, rank=adaln_rank, gate_init=0.0)

        self.head_grid = StructuredGridDecoder(
            patch_size=patch_size,
            embed_dim=embed_dim,
            n_levels=len(config.PRESSURE_LEVELS_HPA),
            n_air_vars=len(config.CERRA_PL_VARS),
            n_2d_channels=c_grid_out - len(config.PRESSURE_LEVELS_HPA) * len(config.CERRA_PL_VARS),
        )
        self.head_metar = MetarRefinementHead(
            patch_size=patch_size,
            embed_dim=embed_dim,
            metar_channels=c_metar_out,
            static_channels=c_static,
            hidden_dim=metar_head_dim,
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv3d, nn.Conv2d, nn.Conv1d)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight, std=0.02)
        # Zero-init all AdaLN-Zero modulation up-projections
        for blk in self.blocks:
            nn.init.zeros_(blk.adaln.up.weight)
            nn.init.zeros_(blk.adaln.up.bias)
        nn.init.zeros_(self.adaLN_final.up.weight)
        nn.init.zeros_(self.adaLN_final.up.bias)
        # Zero-init final output projections so residual prediction starts near identity
        nn.init.zeros_(self.head_grid.air_out.weight)
        nn.init.zeros_(self.head_grid.air_out.bias)
        nn.init.zeros_(self.head_grid.head_2d.linear.weight)
        nn.init.zeros_(self.head_grid.head_2d.linear.bias)

    def _sinusoidal_time_emb(self, t: torch.Tensor) -> torch.Tensor:
        half = self.time_emb_dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / max(half - 1, 1))
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0) * 1000.0
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    def forward(
        self,
        x_in: torch.Tensor,
        global_cond: torch.Tensor,
        ctx_metar: torch.Tensor,
        noisy_metar: torch.Tensor,
        ctx_metar_mask: torch.Tensor,
        static: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x_in:           (B, C_in, 2, H, W)
            global_cond:    (B, 7) with diffusion flow time `t_flow` in the LAST column
            ctx_metar:      (B, 6, H, W) clean normalized context METAR
            noisy_metar:    (B, 6, H, W) noisy target METAR at flow time t
            ctx_metar_mask: (B, 1, H, W) binary valid station mask in context
            static:         (B, 6, H, W) normalized static terrain + coords
        Returns:
            pred_grid:  (B, 69, H, W)
            pred_metar: (B, 6, H, W)
        """
        B, _, T, H, W = x_in.shape
        assert T == 2, f"Expected T=2 (1 context + 1 target), got T={T}"

        t_flow = global_cond[:, -1]
        t_emb = self._sinusoidal_time_emb(t_flow).to(global_cond.dtype)
        c = self.context_mlp(torch.cat([global_cond[:, :-1], t_emb], dim=-1))

        tokens = self.patch_embed(x_in)  # (B, 2 * Hp * Wp, D)
        Hp, Wp = H // self.patch_size[1], W // self.patch_size[2]

        ctx_noise_scale = None
        if self.training and self.kv_ctx_noise > 0.0:
            ctx_noise_scale = torch.rand(B, device=x_in.device, dtype=tokens.dtype) * self.kv_ctx_noise

        for blk in self.blocks:
            tokens = blk(tokens, self.rope, T, Hp, Wp, c, ctx_noise_scale)

        shift_f, scale_f = self.adaLN_final(c).unsqueeze(1).unbind(dim=2)
        tokens = self.norm_final(tokens) * (1.0 + scale_f) + shift_f

        # Decode ONLY target frame tokens (frame 1: second half of sequence)
        n_tpf = Hp * Wp
        target_tokens = tokens[:, n_tpf:, :]

        pred_grid = self.head_grid(target_tokens, H=H, W=W)  # (B, 69, H, W)
        metar_tokens = target_tokens.detach() if self.isolate_metar_grad else target_tokens
        pred_metar = self.head_metar(metar_tokens, ctx_metar, noisy_metar, ctx_metar_mask, static)  # (B, 6, H, W)
        return pred_grid, pred_metar
