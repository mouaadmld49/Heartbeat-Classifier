"""Vérifie que l'environnement est prêt pour le projet Heartbeat Classifier."""
import importlib
import sys

# Bibliothèques indispensables et leur nom d'import
PACKAGES = ["numpy", "pandas", "scipy", "matplotlib", "seaborn",
            "sklearn", "wfdb", "torch", "mlflow"]


def check_imports() -> bool:
    ok = True
    print(f"Python : {sys.version.split()[0]}")
    for name in PACKAGES:
        try:
            mod = importlib.import_module(name)
            print(f"  [OK]  {name:<12} {getattr(mod, '__version__', '?')}")
        except ImportError:
            print(f"  [KO]  {name:<12} non installé")
            ok = False
    return ok


def check_torch() -> None:
    import torch
    print(f"PyTorch : CUDA disponible = {torch.cuda.is_available()}")


def check_physionet() -> bool:
    """Télécharge 10 s du record 100 depuis PhysioNet (test réseau + WFDB)."""
    import wfdb
    try:
        rec = wfdb.rdrecord("100", pn_dir="mitdb", sampto=3600)
        print(f"Record 100 : fs = {rec.fs} Hz, "
              f"canaux = {rec.sig_name}, forme = {rec.p_signal.shape}")
        return True
    except Exception as e:  # réseau, proxy, pare-feu...
        print(f"[KO] Accès PhysioNet impossible : {e}")
        return False


if __name__ == "__main__":
    ok = check_imports()
    if ok:
        check_torch()
        ok = check_physionet()
    sys.exit(0 if ok else 1)