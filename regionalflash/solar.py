"""Vectorized suncalc solar position features (matching suncalc-py / suncalc.js)."""
from __future__ import annotations

import numpy as np

_RAD = np.pi / 180.0
_DAY_S = 86400.0
_J1970 = 2440588.0
_J2000 = 2451545.0
_OBLIQ = _RAD * 23.4397


def sun_position(
    unix_s: np.ndarray | float,
    lon_deg: np.ndarray | float,
    lat_deg: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return `(azimuth_rad, altitude_rad)` matching `suncalc.get_position`."""
    d = (np.asarray(unix_s, dtype=np.float64) / _DAY_S) - 0.5 + _J1970 - _J2000
    lw = _RAD * (-np.asarray(lon_deg, dtype=np.float64))
    phi = _RAD * np.asarray(lat_deg, dtype=np.float64)

    m = _RAD * (357.5291 + 0.98560028 * d)
    c = _RAD * (1.9148 * np.sin(m) + 0.02 * np.sin(2.0 * m) + 0.0003 * np.sin(3.0 * m))
    l = m + c + _RAD * 102.9372 + np.pi

    dec = np.arcsin(np.sin(_OBLIQ) * np.sin(l))
    ra = np.arctan2(np.sin(l) * np.cos(_OBLIQ), np.cos(l))
    h = _RAD * (280.16 + 360.9856235 * d) - lw - ra

    az = np.arctan2(np.sin(h), np.cos(h) * np.sin(phi) - np.tan(dec) * np.cos(phi))
    alt = np.arcsin(np.sin(phi) * np.sin(dec) + np.cos(phi) * np.cos(dec) * np.cos(h))
    return az, alt


def solar_step_features(
    ctx_unix_s: np.ndarray | float,
    tgt_unix_s: np.ndarray | float,
    noon_unix_s: np.ndarray | float,
    lon_deg: np.ndarray | float,
    lat_deg: np.ndarray | float,
) -> np.ndarray:
    """Return 8D solar geometry features for AdaLN conditioning over a 3h forecast step:
    `[sin(az_ctx), cos(az_ctx), sin(alt_ctx), sin(noon_alt),
      sin(az_tgt), cos(az_tgt), sin(alt_tgt), sin(alt_tgt) - sin(alt_ctx)]`.
    """
    az_c, alt_c = sun_position(ctx_unix_s, lon_deg, lat_deg)
    az_t, alt_t = sun_position(tgt_unix_s, lon_deg, lat_deg)
    _, alt_noon = sun_position(noon_unix_s, lon_deg, lat_deg)

    sin_alt_c = np.sin(alt_c)
    sin_alt_t = np.sin(alt_t)
    feats = np.stack(
        [
            np.sin(az_c),
            np.cos(az_c),
            sin_alt_c,
            np.sin(alt_noon),
            np.sin(az_t),
            np.cos(az_t),
            sin_alt_t,
            sin_alt_t - sin_alt_c,
        ],
        axis=-1,
    )
    return feats.astype(np.float32)
