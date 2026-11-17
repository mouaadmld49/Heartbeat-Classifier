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