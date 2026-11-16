"""Constantes du projet : records MIT-BIH, mapping AAMI, splits DS1/DS2."""

FS = 360  # fréquence d'échantillonnage attendue (Hz)

# Les 48 enregistrements de la base MIT-BIH
_100 = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 111, 112, 113, 114,
        115, 116, 117, 118, 119, 121, 122, 123, 124]
_200 = [200, 201, 202, 203, 205, 207, 208, 209, 210, 212, 213, 214, 215, 217,
        219, 220, 221, 222, 223, 228, 230, 231, 232, 233, 234]
ALL_RECORDS = [str(r) for r in _100 + _200]

# Records avec rythme stimulé (pacemaker) : exclus par convention AAMI
PACED_RECORDS = ["102", "104", "107", "217"]

# Split inter-patients de Chazal et al. (2004)
DS1 = ["101", "106", "108", "109", "112", "114", "115", "116", "118", "119",
       "122", "124", "201", "203", "205", "207", "208", "209", "215", "220",
       "223", "230"]
DS2 = ["100", "103", "105", "111", "113", "117", "121", "123", "200", "202",
       "210", "212", "213", "214", "219", "221", "222", "228", "231", "232",
       "233", "234"]

# Garde-fous : toute erreur de saisie fait échouer l'import immédiatement
assert len(ALL_RECORDS) == 48
assert not (set(DS1) & set(DS2)), "DS1 et DS2 ne doivent pas se chevaucher"
assert set(DS1) | set(DS2) | set(PACED_RECORDS) == set(ALL_RECORDS)

# Mapping explicite symbole d'annotation -> classe AAMI
AAMI_MAP = {
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
    "A": "S", "a": "S", "J": "S", "S": "S",
    "V": "V", "E": "V",
    "F": "F",
    "/": "Q", "f": "Q", "Q": "Q",
}
CLASS_ORDER = ["N", "S", "V", "F", "Q"]

# Descriptions pour les tableaux (symboles de battements, puis non-battements)
SYMBOL_DESCRIPTIONS = {
    "N": "Normal", "L": "Bloc de branche gauche", "R": "Bloc de branche droit",
    "e": "Echappement auriculaire", "j": "Echappement nodal",
    "A": "Extrasystole auriculaire", "a": "Extrasystole auriculaire aberrante",
    "J": "Extrasystole nodale", "S": "Extrasystole supraventriculaire",
    "V": "Extrasystole ventriculaire (PVC)", "E": "Echappement ventriculaire",
    "F": "Fusion ventriculaire + normal", "/": "Battement stimule",
    "f": "Fusion stimule + normal", "Q": "Non classable",
    "+": "Changement de rythme", "~": "Changement de qualite du signal",
    "|": "Artefact isole", "x": "Onde P non conduite", "!": "Flutter ventriculaire",
    "[": "Debut flutter/fibrillation", "]": "Fin flutter/fibrillation",
    '"': "Commentaire",
}