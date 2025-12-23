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

Nous testerons le dernier grand modèle LLM de traitement de texte développé par Mistral AI, Devstral‑2‑2512 [août 2025].

Devstral est un modèle de langage spécialisé pour les agents logiciels et la génération de code, mais sa nature de LLM instructif lui permet également d’être utilisé pour des tâches de NLP classiques, comme la classification de textes, le résumé ou l’extraction d’information.

Sa capacité à suivre des instructions et à traiter de larges contextes textuels le rend applicable à des tâches de NLP classiques. Dans ce travail, nous l’avons utilisé pour la classification multi-classe de textes, en exploitant le prompt engineering, en utilisant un prompt clair et structuré.