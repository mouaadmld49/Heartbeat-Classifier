"""Augmentation à la volée (entraînement uniquement, jamais validation/test)."""
import math

import torch


def augment_batch(x, y=None, only_classes=None, max_shift=10, scale_range=(0.9, 1.1),
                  noise_std=0.05, wander_amp=0.2, fs=360):
    """x : tenseur (B, C, L). Retourne un tenseur de même forme.

    - décalage temporel entier dans [-max_shift, max_shift] (réplication des bords)
    - échelle d'amplitude aléatoire par battement
    - bruit gaussien (écart-type noise_std, en unités du signal normalisé)
    - dérive sinusoïdale lente (0,1 à 0,5 Hz, amplitude <= wander_amp)
    Si only_classes est donné (avec y), seuls ces battements sont modifiés.
    Les valeurs par défaut sont PROVISOIRES : à régler sur la validation (Phase 8).
    """
    B, C, L = x.shape
    dev = x.device
    out = x
    if max_shift > 0:
        s = torch.randint(-max_shift, max_shift + 1, (B,), device=dev)
        idx = (torch.arange(L, device=dev)[None, :] + s[:, None]).clamp(0, L - 1)
        out = out.gather(2, idx[:, None, :].expand(B, C, L))
    lo, hi = scale_range
    out = out * (lo + (hi - lo) * torch.rand(B, 1, 1, device=dev))
    if noise_std > 0:
        out = out + noise_std * torch.randn_like(out)
    if wander_amp > 0:
        t = torch.arange(L, device=dev)[None, None, :] / fs
        freq = 0.1 + 0.4 * torch.rand(B, 1, 1, device=dev)
        phase = 2 * math.pi * torch.rand(B, 1, 1, device=dev)
        amp = wander_amp * torch.rand(B, 1, 1, device=dev)
        out = out + amp * torch.sin(2 * math.pi * freq * t + phase)
    if only_classes is not None:
        if y is None:
            raise ValueError("y est requis avec only_classes")
        mask = torch.isin(y, torch.as_tensor(list(only_classes), device=y.device))
        out = torch.where(mask[:, None, None], out, x)
    return out