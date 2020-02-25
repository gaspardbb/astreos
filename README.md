# Outline 

## État

* Le DataFrame général est bien organisé. Dans `load_utils.py`.
* La création de features à la main est dans `features.py`, et est standardisée (cf. `feature_engineering.ipynb`). 
But: rendre l'aggrégation de tous les features et cie plus simple.

## À faire

### Features

Ajouter plus de features. Pour l'instant, seulement 3: moyenne de T, CLCT, windspeed. 

Idées:
* Variation de variables (est-ce qu'il a fait meilleur que les jours précédents?)
* changement de *direction* du vent

### Modèles

#### Pipeline 
Créer un fichier qui s'inscrive bien dans la pipeline, ie. qui prenne en entrée un DF avec des colonnes:
* WF: 1 ... 6
* var: mean_T, mean_U, ...

#### Architecture
Plusieurs stratégies à essayer:
* 1 modèle pour 1 WF centré
* + 1 modèle pour superposition de tous les WF centr***ées***. 