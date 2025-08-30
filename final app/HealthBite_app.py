import io
import re
import json
from typing import List, Tuple, Optional
import streamlit as st
import pandas as pd

st.set_page_config(page_title="HealthBite", page_icon="🥗", layout="centered")
st.image('app images\HealthBite_image.png', use_column_width=True)
st.title("🥗 HealthBite")
st.caption("Un recomendador inteligente de recetas que también intenta ayudarte a sentir " \
"mejor a traves de una nutrición adecuada")

# ---------------- UI Streamlit ----------------
# En esta sección se recopila el código relacionado al UI de la app en sí
st.subheader("1) Sube una imagen de la nevera y describe cómo te sientes últimamente!")
user_text = st.text_area("Cómo te sientes últimamente?", placeholder="Por ejemplo: Últimamente me siento cansado constantemente")
photo = st.file_uploader("Sube la foto de tu nevera", placeholder="Detectaremos automáticamente los ingredientes que tienes disponibles!", type=["jpg","jpeg","png","webp"])

st.divider()

if st.button("Recommend", type="primary", use_container_width=True):
    # Load recipes
    if up:
        try:
            df = pd.read_csv(io.BytesIO(up.read()))
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            st.stop()
    else:
        df = demo_recipes()



