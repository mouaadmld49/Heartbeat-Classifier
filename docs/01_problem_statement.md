# Heartbeat Classifier : énoncé du problème

## Avertissement
Prototype de recherche. Ce n'est pas un dispositif médical et il ne fournit pas de diagnostic.

## Problème
Classer chaque battement d'un ECG (fenêtre centrée sur le pic R) en classe AAMI : N, S, V, F, (Q).
Tâche : classification supervisée multiclasse, évaluée sur des patients jamais vus.

## Données
MIT-BIH Arrhythmia Database (PhysioNet) : 48 enregistrements, 360 Hz.

## Objectifs
O1 mapping AAMI documenté ; O2 dataset sans fuite ; O3 baselines ML ;
O4 CNN 1D évalué en inter-patients ; O5 comparaison des méthodes de déséquilibre ;
O6 application complète (API, frontend, base, Docker, MLflow).

## Hypothèses
H1 CNN 1D > baselines (Macro-F1). H2 pondération : recall de S et F en hausse.
H3 split aléatoire > split inter-patients (fuite). H4 le contexte RR aide pour S.

## Métriques
Recall par classe, Macro-F1, matrice de confusion (jamais l'accuracy seule).

## Limites
Petite base mono-centre, 2 dérivations, pas de diagnostic de rythme, aucune validation clinique.