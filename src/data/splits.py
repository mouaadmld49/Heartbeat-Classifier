"""Split par record (patient) : test = DS2, validation choisie dans DS1."""
import numpy as np


def choose_val_records(counts, ds1, classes, n_val=5, frac=0.2,
                       n_trials=5000, seed=42):
    """Choisit n_val records de DS1 pour la validation (reproductible).

    counts : DataFrame (record_id x classes) des effectifs.
    Contraintes : chaque classe présente dans la validation ; on minimise
    l'écart entre la part de chaque classe en validation et `frac`.
    Seules les étiquettes de DS1 sont utilisées (jamais DS2).
    """
    rng = np.random.default_rng(seed)
    ds1 = list(ds1)
    totals = counts.loc[ds1, classes].sum()
    best, best_score = None, np.inf
    for _ in range(n_trials):
        pick = [str(r) for r in rng.choice(ds1, size=n_val, replace=False)]
        c = counts.loc[pick, classes].sum()
        if (c == 0).any():
            continue
        score = float(np.abs(c / totals - frac).sum())
        if score < best_score:
            best, best_score = pick, score
    if best is None:
        raise RuntimeError("Aucune validation valide trouvée : augmentez n_trials.")
    return sorted(best)


def assign_splits(record_ids, ds1, ds2, val_records):
    """Retourne un tableau 'train' / 'val' / 'test' selon le record."""
    rec = np.asarray(record_ids).astype(str)
    out = np.full(rec.shape, "train", dtype=object)
    out[np.isin(rec, list(val_records))] = "val"
    out[np.isin(rec, list(ds2))] = "test"
    unknown = ~np.isin(rec, list(ds1) + list(ds2))
    if unknown.any():
        raise ValueError(f"Records hors DS1/DS2 : {set(rec[unknown])}")
    return out.astype(str)