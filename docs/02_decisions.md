# Décisions méthodologiques (Phases 2-3)

- Records stimulés (102, 104, 107, 217) exclus : ils contiennent 8 028 des 8 043 battements Q.
- Classe Q retirée : 15 battements restants (8 DS1, 7 DS2), inapprenable et non évaluable.
  Problème à 4 classes : N, S, V, F.
- Dérivation : MLII, cherchée par son nom (le canal 0 n'est pas toujours MLII).
- Filtre : Butterworth passe-bande 0,5-40 Hz, ordre 4, sosfiltfilt (phase nulle). Notch 60 Hz optionnel.
- Normalisation : robuste (médiane/IQR) par record, sans statistiques inter-records.
- Fenêtre : 90 échantillons avant, 110 après le pic annoté (L = 200, fs = 360 Hz).
- Positions des pics : annotations pour l'entraînement ; détecteur en production.
- Split : test = DS2 (de Chazal et al., 2004) ; validation = 5 records de DS1 choisis par tirage
  reproductible (seed 42) sur les seules étiquettes de DS1.
- Limite connue : les records 201 (DS1) et 202 (DS2) viennent du même patient.
  Protocole conservé pour la comparabilité ; analyse de sensibilité sans 202 prévue en Phase 9.
- Changement de distribution : S = 1,85 % dans DS1 contre 3,70 % dans DS2.
- F est essentiellement un seul patient par split : 208 = 372/385 F (train), 213 = 362/388 F (test).
  Le recall de F est donc quasi un recall par patient. Il est toujours rapporté avec sa répartition par record.
- Sélection d'hyperparamètres : StratifiedGroupKFold par record sur DS1, critère Macro-F1 sur N, S, V (F exclu du critère).
- Évaluation finale : DS1 -> DS2 (principale), puis DS2 -> DS1 (sensibilité : F testé sur 208 puis 213).
- Détecteur R : rel_thr = 0,15, floor_ratio = 0,03, choisis sur DS1 (F1 max avec pire record >= 95 %).
- SMOTE non utilisé : F vient d'un seul patient, l'interpolation créerait des quasi-copies de ce patient.