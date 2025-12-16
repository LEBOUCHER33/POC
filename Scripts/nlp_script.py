"""
Script d'implémentation de techniques de traitement du langage naturel (NLP) sur un dataset textuel.

modèle utilisé : Llama-2 

"""



# Importation des bibliothèques nécessaires
import pandas as pd
import torch


# Chargement du dataset nettoyé
data = pd.read_csv('./Data/flipkart_cleaned.csv')