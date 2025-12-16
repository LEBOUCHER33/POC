"""
Script d'implémentation de techniques de traitement du langage naturel (NLP) sur un dataset textuel.

modèle utilisé : Llama-2 

Worflow :
1- Nettoyage et préparation des données textuelles
2- Loading du modèle pré-entraîné et du tokenizer
3- définir le pipeline Hugging Face
4- définir le prompt de classification
5- exécution de l'inférence sur les données textuelles avec le modèle LLaMA 2 prentraîné

"""



# Importation des bibliothèques nécessaires
import pandas as pd
import torch
import re
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer


# Chargement du dataset epuré
data = pd.read_csv('./Data/flipkart_cleaned.csv')

print(f"données textuelles descriptives du premier article :\n {data['description'].iloc[0]}",
      f"type de données : {type(data['description'].iloc[0])}")


# /////////////////////////////////////////////////////////
# nettoyage et préparation des données textuelles
# /////////////////////////////////////////////////////////



def clean_text(text):
    # Convertir en minuscules
    text = text.lower()
    # Supprimer les balises HTML
    text = re.sub(r'<.*?>', '', text)
    # Supprimer la ponctuation et les caractères spéciaux
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Supprimer les espaces supplémentaires
    text = re.sub(r'\s+', ' ', text).strip()
    return text

data['description'] = data['description'].apply(clean_text)

print(f"données textuelles descriptives du premier article après nettoyage :\n {data['description'].iloc[0]}")








