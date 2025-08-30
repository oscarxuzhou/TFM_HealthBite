import os, json, uuid
import io
from datetime import datetime
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from ultralytics import YOLO
import unicodedata
import re
import requests
import ast
from sklearn.preprocessing import MultiLabelBinarizer
from scipy import sparse
from typing import List, Tuple, Optional
import streamlit as st
from app_key_functions import *


# st.set_page_config(page_title="HealthBite", page_icon="🥗", layout="centered")
# st.image('app images\HealthBite_image.png', use_column_width=True)
# st.title("🥗 HealthBite")
# st.caption("Un recomendador inteligente de recetas que también intenta ayudarte a sentir " \
# "mejor a traves de una nutrición adecuada")

# # ---------------- UI Streamlit ----------------
# # En esta sección se recopila el código relacionado al UI de la app en sí
# st.subheader("1) Sube una imagen de la nevera y describe cómo te sientes últimamente!")
# user_text = st.text_area("Cómo te sientes últimamente?", placeholder="Por ejemplo: Últimamente me siento cansado constantemente")
# photo = st.file_uploader("Sube la foto de tu nevera", placeholder="Detectaremos automáticamente los ingredientes que tienes disponibles!", type=["jpg","jpeg","png","webp"])

# st.divider()

# if st.button("Recommend", type="primary", use_container_width=True):
#     # Load recipes
#     if up:
#         try:
#             df = pd.read_csv(io.BytesIO(up.read()))
#         except Exception as e:
#             st.error(f"Could not read CSV: {e}")
#             st.stop()
#     else:
#         df = demo_recipes()


# HealthBite_app.py
# HealthBite_app.py
import io
from pathlib import Path

import pandas as pd
import streamlit as st

from app_key_functions import (
    NLP_YOLO_predictor_function,
    compute_recipe_scores,
    rank_recipes,
    run_rag_over_top10_human_output,
    safe_to_list,
)

st.image(r'app_images\HealthBite_banner_image.png')
st.markdown("<h1 style='text-align: center;'>HealthBite</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: left;'>Un recomendador inteligente de recetas que detecta ingredientes en tu nevera y también intenta ayudarte a sentir mejor a traves de una nutrición adecuada</h1>", unsafe_allow_html=True)
st.divider()

# Sección para describir el estado físico/anímico
st.subheader("1) Describe cómo te sientes últimamente")
user_text = st.text_area(
    "¿Cómo te has sentido últimamente?",
    placeholder="Por ejemplo: Llevo días que me duele la cabeza constantemente",
)
st.divider()

# Sección para la foto de nevera
st.subheader("2) Sube una foto de tu nevera")
photo = st.file_uploader("Foto de tu nevera", type=["jpg", "jpeg", "png", "webp"])
st.divider()

# Botón
go = st.button("🥘Recomiéndame recetas", type="primary", use_container_width=True)
if go:
    with st.spinner("Procesando…"):

        # Guardando la foto cuando el usuario sube foto
        image_path = None
        if photo is not None:
            tmp_dir = Path(".tmp_inputs")
            tmp_dir.mkdir(parents=True, exist_ok=True)
            image_path = str(tmp_dir / f"fridge_{photo.name}")
            with open(image_path, "wb") as f:
                f.write(photo.read())

        # Se carga el dataset de recetas traducidas
        df_recipes = pd.read_csv(r'..\dataset\Recipes dataset\recipes_dataset_translated_ingredients.csv')

        # Paso 1+2: Predictor de síntoma y detector de ingredientes disponibles en la nevera
        pred_df = NLP_YOLO_predictor_function(user_text, image_path=image_path)
        pred_df = pred_df[['run_id', 'timestamp', 'predicted_symptom', 'fridge_ingredients_available', 'nutritional deficiency', 'ingredients supplying deficiency']]
        pred_df = pred_df.rename(columns= {'timestamp': 'marca de tiempo', 'predicted_symptom': 'síntoma predecido', 
                                           'fridge_ingredients_available': 'ingredientes en la nevera', 
                                           'nutritional deficiency': 'posibles deficiencias nutricionales', 
                                           'ingredients supplying deficiency': 'ingredientes nutricionales útiles'})

        st.divider()
        
        st.subheader("Resultado de las predicciones")
        st.markdown("<p style='text-align: left;'>En base a la foto que has subido y tu descripción anímico/físico, detecto:</h1>", unsafe_allow_html=True)
        st.dataframe(pred_df, use_container_width=True)
        st.markdown("<p style='text-align: left;'>Dado estos resultados, te recomendaré recetas que encajen con tus ingredientes disponibles y que puedan ayudar con posibles deficiencias nutricionales</h1>", unsafe_allow_html=True)

        row = pred_df.iloc[0]  
        fridge_list = row['ingredientes en la nevera']
        deficiency_list = row['ingredientes nutricionales útiles']

        # Paso 3: Puntuación y ranking de recetas
        scored = compute_recipe_scores(
            df_recipes,
            fridge_list=fridge_list,
            deficiency_ingredients_list=deficiency_list
        )

        scored = scored.merge(pred_df, how='cross')
        top10 = rank_recipes(scored, top_n=10)

        # st.subheader("Top 10 recetas candidatas")
        # st.dataframe(
        #     top10[["title", "jaccard_score", "DRC_coverage", "missing_penalization", "effort", "final_score"]],
        #     use_container_width=True, height=350
        # )

        # Paso 4: RAG LLM como juez
        final_text = run_rag_over_top10_human_output(
            top10_df=top10,
            user_text=user_text or "",
            model_name="llama3.1:8b",
            top_n_final=3
        )

        st.subheader("Recetas recomendadas")
        st.markdown(final_text)

        st.success("Recetas listas!")
