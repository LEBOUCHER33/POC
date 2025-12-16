# POC


## Objectif

Identifier une méthode récente de NLP ou de Computer Vision pour réaliser une preuve de concept de la méthodologie.

## Workflow 

1- Identifier une méthodologie et réaliser une analyse bibliographique sur le sujet
2- Identifier un jeu de données pour la tester
3- Entrainer ou loader un modèle pré-entrainé sur le jeu de données
4- Interpréter et comparer les résultats obtenus

## Data et objectif

Nous reprendrons le jeu de données des articles d'une marketplace afin de tester la classification automatique des données textuelles ou visuelles.

Description du jeu de données :
Chaque article référencé est associé à une description textuelle et une image.
Il y a 7 catégories différentes pour classer les articles sans biais de représentativité.

## Méthode

Nous testerons le dernier grand modèle LLM de traitement de texte développé par Méta, LLAMA2.

Raisons argumentant ce choix dans le cadre d'une preuve de concept de la méthode :

- LLM Open Source
- performances du modèle comparativement à d'autres
- disponibilité du modèle pré-entrainé avec 4 poids différents
- très bon compromis qualité / simplicité
- facile à intégrer avec Hugging Face
- pas besoin de fine-tuning pour démontrer de la valeur
