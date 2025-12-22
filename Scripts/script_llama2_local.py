import pandas as pd
import torch
import re
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer
from sklearn.metrics import classification_report, accuracy_score
import requests
import os


# ---- Chargement du dataset epuré ----
data = pd.read_csv('../Data/flipkart_cleaned.csv')

# ---- Nettoyage des données textuelles descriptives ----
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


data.to_csv('../Data/flipkart_prepared.csv', index=False)

categories = data['product_category'].unique()


# ---- chargement du modèle pré-entrainé et du tokenizer ----

from transformers import AutoTokenizer, AutoModelForCausalLM


model_id = "meta-llama/Llama-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    use_auth_token=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
    use_auth_token=True
)


# GPU obligatoire