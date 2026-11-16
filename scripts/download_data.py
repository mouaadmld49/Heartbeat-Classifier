"""Télécharge la base MIT-BIH dans data/raw/mitdb."""
from pathlib import Path

import wfdb

DEST = Path(__file__).resolve().parents[1] / "data" / "raw" / "mitdb"
DEST.mkdir(parents=True, exist_ok=True)

wfdb.dl_database("mitdb", dl_dir=str(DEST))

n = {ext: len(list(DEST.glob(f"*.{ext}"))) for ext in ("dat", "hea", "atr")}
print("Fichiers téléchargés :", n)  # attendu : 48 de chaque