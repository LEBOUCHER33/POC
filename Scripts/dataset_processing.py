"""

Script de processing des données textuelles pour la préparation du dataset.


"""


# Importation des bibliothèques nécessaires
import os
import pandas as pd
import numpy as np
import re

# loading des data

data = pd.read_csv('./Data/flipkart_com-ecommerce_sample_1050.csv')

data['product_category'] = data['product_category_tree'].str.split('>>').str[0].str.strip() 
data['product_category']=data['product_category'].str.replace(r'[\[\]"]', '', regex=True)
print(data['product_category'].value_counts())


# sélection des colonnes pertinentes
df = data[["uniq_id", "product_name","description", "product_category"]]


df = pd.DataFrame(df)
print("Aperçu du dataset nettoyé :")
print(df.head())
print(f"categorie du premier article : {df.iloc[0][-1]}\n",
      f"description du premier article : {df['description'].iloc[0]}")  

# Sauvegarde du dataset nettoyé
df.to_csv('./Data/flipkart_cleaned.csv', index=False)