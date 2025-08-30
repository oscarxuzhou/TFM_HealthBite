import os, json, uuid
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

# ------------------ Sección de funciones normalizadoras de texto -----------------
def _strip_accents(s: str) -> str:
    """
    Función normalizadora que elimina acentos y carácteres no estándares
    Ejemplos:
    "Niño"    → "Nino"
    "canción"   → "cancion

    Parámetros
    ----------
    s : str
        Texto de entrada

    Devuelve
    -------
    str
        Cadena sin acentos/diacríticos y normalizada (NFKC).
    """
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFKC", s)

def _norm(s: str) -> str:
    """
    Normaliza texto para comparaciones quitando espacios y pasando a minúsculas.
    En esta función se reutiliza la función strip_accents definido previamente

    Pasos:
      - strip(): recorta espacios iniciales/finales.
      - lower(): minúsculas.
      - _strip_accents(): función anterior definida para quitar acentos y carácteres no estándares
      - re.sub(r"\\s+", " ", ...): reemplaza cualquier secuencia de espacios en blanco
        (incluye tabs/saltos de línea) por un único espacio.

    Ejemplos
    --------
    _norm("  Niño   Feliz\\n")
    'nino feliz'
    _norm("  CANCIÓN  ")
    'cancion'

    Parámetros
    ----------
    s : str
        Texto de entrada.

    Devuelve
    -------
    str
        Texto normalizado para comparaciones futuras.
    """
    s = _strip_accents((s or "").strip().lower())
    s = re.sub(r"\s+", " ", s)
    return s

# ------------------ NLP Wrapper -----------------
class SymptomClassifier:
    """
    Una clase llamada "SymptomClassifier" que contiene el modelo de clasificación de texto NLP transformers.
    El modelo predice un síntoma a partir de la descripción del usuario 

    SymptomClassifier:
      - Carga `AutoTokenizer` y `AutoModelForSequenceClassification` desde `model_dir`.
      - Selecciona automáticamente el dispositivo, utiliza GPU si está disponible (para que sea más rápido)
      - Tiene un mapeo para traducir ids ↔ etiquetas (es decir, la etiqueta está en número pero hay un mapeo que lo traduce a algo "legible")

    Parámetros
    ----------
    model_dir : str
        Ruta a la carpeta del modelo entrenado
    labels_path : str
        Ruta a `symptom2id.json`, el documento de mapeo
    device : str o None, opcional
        Dispositivo sobre el que ejecutar el modelo (p. ej., "cuda" / "cpu").
        Si es None, se detecta automáticamente.
    max_length : int, opcional
        Longitud máxima de secuencia para el tokenizador (truncación si se excede).

    Ejemplo
    -------
    El modelo devuelve el síntoma predecido y la confianza de predicción.
    >>> clf = SymptomClassifier("../mi_modelo")
    >>> label, conf = clf.predict_one("Me siento muy cansado últimamente")
    ('fatiga', 0.842)
    """
    
    def __init__(self, model_dir, labels_path, device=None, max_length=256):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.max_length = max_length

        if os.path.exists(labels_path):
            with open(labels_path, "r", encoding="utf-8") as f:
                mapping = json.load(f)
            if all(not str(k).isdigit() for k in mapping.keys()):   
                self.label2id = {str(k): int(v) for k, v in mapping.items()}
                self.id2label = {v: k for k, v in self.label2id.items()}
            else:                                                    
                self.id2label = {int(k): str(v) for k, v in mapping.items()}
                self.label2id = {v: k for k, v in self.id2label.items()}
        else:
            id2label = getattr(self.model.config, "id2label", None)
            if id2label:
                self.id2label = {int(k): v for k, v in id2label.items()}
            else:
                self.id2label = {i: f"LABEL_{i}" for i in range(self.model.config.num_labels)}

# --- Parte de predicción -----
    @torch.inference_mode()
    def predict_one(self, text: str):
        enc = self.tokenizer([text], padding=True, truncation=True,
                             max_length=self.max_length, return_tensors="pt").to(self.device)
        self.model.eval()
        out = self.model(**enc)
        probs = torch.softmax(out.logits, dim=1).detach().cpu().numpy()[0]
        pred_id = int(np.argmax(probs))
        label = self.id2label.get(pred_id, f"LABEL_{pred_id}")
        conf = float(probs[pred_id])

# ---- instantiate models (adjust paths) ----
model_dir = r"..\models\NLP\NLP best model" 
labels_path = r"..\models\NLP\NLP best model\symptom2id.json" 

sym_clf = SymptomClassifier(model_dir, labels_path=labels_path)



# ------------------ Modelo de YOLO -----------------
yolo_weights = r"..\models\YOLO\runs\YOLO best model\train 3 - 100 epochs with synthetic\weights\best.pt"
yolo = YOLO(yolo_weights)

def detect_ingredients_list(image_path: str, conf_threshold: float = 0.5):
    """
    Detecta ingredientes en una imagen usando un modelo YOLO y 
    devuelve una lista de nombres de ingredientes.

    Flujo:
      1) Ejecuta el detector: `yolo(image_path)[0]` -> coge la primera imagen detectada
      2) Recorre las cajas detectadas (`res.boxes`), leyendo:
         - `cls_id`: id de clase predicha.
         - `conf`: confianza de la predicción.
      3) Filtra por `conf_threshold` (omite detecciones de baja confianza).
      4) Convierte `cls_id` a nombre humano con `res.names[cls_id]`.
      5) Normaliza el texto con `_norm` (minúsculas, sin acentos, espacios colapsados).
      6) Elimina duplicados y ordena alfabéticamente antes de devolver.

    Parámetros
    ----------
    image_path : str
        Ruta a la imagen de entrada (la foto de la nevera)
    conf_threshold : float, opcional (por defecto=0.5)
        Umbral mínimo de confianza para aceptar una detección.
        Valores más altos (p. ej., 0.25–0.5) reducen falsos positivos, si el objetivo es detectar lo máximo posible, se puede usar un umbral más bajo.

    Devuelve
    -------
    list[str]
        Lista de nombres de ingredientes

    Ejemplo
    -------
    >>> detect_ingredients_list("fridge.jpg", conf_threshold=0.3)
    ['huevo', 'leche', 'tomate']
    """
    res = yolo(image_path)[0]
    names = res.names  
    detected = []
    for i in range(len(res.boxes)):
        cls_id = int(res.boxes.cls[i].item())
        conf = float(res.boxes.conf[i].item())
        if conf < conf_threshold:
            continue
        raw = names.get(cls_id, str(cls_id))
        normed = _norm(raw)  
        detected.append(normed)
    return sorted(set(detected))


# ------------------ Combinación YOLO + NLP -----------------

def NLP_YOLO_predictor_function(text: str, image_path: str) -> pd.DataFrame:
    """
    Runs NLP + YOLO and returns a ONE-ROW DataFrame with list in 'ingredients_list'.
    """
    run_id = str(uuid.uuid4())[:8]
    ts = datetime.now().isoformat(timespec="seconds")

    # Predictions
    symptom_label, conf = sym_clf.predict_one(text)
    ingredients = detect_ingredients_list(image_path, conf_threshold=0.05)

    # build one-row DF (wrap dict in a list!)
    df_predicted = pd.DataFrame([{
        "run_id": run_id,
        "timestamp": ts,
        "input_text": text,
        "image_path_used": image_path,
        "predicted_symptom": symptom_label,  # human-readable label
        "confidence": conf,
        "fridge_ingredients_available": ingredients      # stays as Python list in a single cell
    }])

    deficiency_df = pd.read_csv(r'..\dataset\Nutritional deficiency dataset\deficiencies_dataset.csv')
    deficiency_df = deficiency_df[['sintoma', 'deficiencia de nutrientes', 'disponible en ingredientes']].drop_duplicates().reset_index(drop=True)
    deficiency_df.rename(columns={'sintoma':'predicted_symptom', 'deficiencia de nutrientes': 'nutritional deficiency', 'disponible en ingredientes' : 'ingredients supplying deficiency'}, inplace=True)
    df_predicted = pd.merge(df_predicted, deficiency_df, on='predicted_symptom', how='left')  
    df_predicted['ingredients supplying deficiency'] = (df_predicted['ingredients supplying deficiency'].str.split(';').apply(lambda lst: [x.strip() for x in lst if x.strip()])  # strip spaces, drop empties
)  
    return df_predicted

# ------------------ Sistema y algoritmo de puntuación -----------------

def _parse_terms(x):
    """
    Normaliza y convierte distintos posibles formatos de términos de ingredientes a una lista de strings
    en minúsculas y sin espacios extra.

    Inputs aceptados:
      - list: se normaliza cada elemento con str().strip().lower().
      - str que representa una lista: intenta parsear con ast.literal_eval (p. ej. "['tomate','leche']").
      - str con términos separados por comas: "tomate, leche, huevo".


    Ejemplos
    --------
    >>> _parse_terms(["Tomate", "  Leche  ", ""])
    ['tomate', 'leche']
    >>> _parse_terms("['Tomate','LECHE']")
    ['tomate', 'leche']
    >>> _parse_terms("tomate, leche ,  huevo")
    ['tomate', 'leche', 'huevo']

    Devuelve
    -------
    list[str]
        Lista normalizada de términos.
    """

    if isinstance(x, list):
        return [str(t).strip().lower() for t in x if str(t).strip()]

    if isinstance(x, str):
        try:
            v = ast.literal_eval(x)
            if isinstance(v, list):
                return [str(t).strip().lower() for t in v if str(t).strip()]
        except Exception:
            pass
        return [t.strip().lower() for t in x.split(",") if t.strip()]
    return []

def compute_recipe_scores(recipes_df, fridge_list, deficiency_ingredients_list=None):
    """
    Esta función calcula las siguientes métricas:
    1) Jaccard score
    2) Penalización por faltantes (missing_penalization)
    3) Cobertura de deficiencias (DRC_coverage)
    4) Esfuerzo (effort)


    Parámetros
    --------
    recipes_df : pd.DataFrame que contendrá los ingredientes de la receta y también las instrucciones
    fridge_list : list[str]
        Ingredientes detectados en la nevera
    deficiency_ingredients_list : list[str]
        Ingredientes que ayudan a cubrir deficiencias

    Comentarios adicionales
    -----------------------
    - Convierte 'NER_terms_es' (la columna del df que contiene los ingredientes necesarios de la receta) a listas limpias con `_parse_terms`.
    - Usa MultiLabelBinarizer para crear una **matriz binaria dispersa** X (recetas × vocabulario),
      lo que permite calcular intersecciones y tamaños de conjuntos sin bucles Python (es más eficiente y más rápido que un bucle Python en 2 millones de filas).
    - El tamaño de la receta |R| se obtiene con `np.diff(X.indptr)` (traversal O(1) por fila en CSR).
    - La normalización de esfuerzo usa min–max global

    Devuelve
    -------
    pd.DataFrame
        Copia de `recipes_df` con columnas añadidas:
        ['recipe_ingredients','jaccard_score','missing_penalization',
        'DRC_coverage','Effort']
    """

     
    df = recipes_df.copy()

    # 1) Normalizamos los inputs
    fridge = {str(i).strip().lower() for i in (fridge_list or []) if str(i).strip()}
    deficiencies = {str(i).strip().lower() for i in (deficiency_ingredients_list or []) if str(i).strip()}

    terms = df["NER_terms_es"].map(_parse_terms)

    # 2) Matriz binaria recetas×vocabulario
    mlb = MultiLabelBinarizer(sparse_output=True)  
    X = mlb.fit_transform(terms)                   
    vocab = np.array(mlb.classes_)                 
    fridge_mask = np.isin(vocab, list(fridge))
    def_mask = np.isin(vocab, list(deficiencies)) if deficiencies else np.zeros_like(vocab, dtype=bool)


    # ------------ Cálculo de métricas ----------------------
    # 1. Cálculo de jaccard score
    # inter = |R ∩ F|  → suma por fila de las columnas del vocab que están en la nevera
    inter = (X[:, fridge_mask].sum(axis=1).A1) if fridge_mask.any() else np.zeros(X.shape[0], dtype=int)

    # recipe_size = |R| -> nº de no-nulos por fila en CSR (usando indptr)
    recipe_size = np.diff(X.indptr)

    # union = |R ∪ F| = |R| + |F| - |R ∩ F|
    union = recipe_size + fridge_mask.sum() - inter
    jaccard = np.divide(inter, np.maximum(union, 1), dtype=float)

    # Cálculo de missing penalization
    num_needed = recipe_size - inter
    missing_penalization = np.divide(num_needed, np.maximum(recipe_size, 1), dtype=float)

    # Cálculo de deficiency coverage
    den = int(def_mask.sum())
    drc_abs = X[:, def_mask].sum(axis=1).A1 if den > 0 else np.zeros(X.shape[0], dtype=int)
    drc_coverage = np.divide(drc_abs, den, out=np.zeros_like(drc_abs, dtype=float), where=den > 0)
    
    # Cálculo de effort -> número effort = normalización min–max del nº de tokens en 'directions' (más largo → mayor esfuerzo)
    def _as_text(x):
        return " ".join(x) if isinstance(x, list) else str(x or "")
    tokens = df["directions"].map(_as_text).str.findall(r"\w+").str.len()
    t_min, t_max = int(tokens.min() or 0), int(tokens.max() or 0)
    effort = (tokens - t_min) / max(1, t_max - t_min)

    # Construimos el resultado df final
    df["recipe_ingredients"] = terms  
    df["inter"] = inter
    df["union"] = union
    df["jaccard_score"] = jaccard
    df["missing_penalization"] = missing_penalization
    df["DRC_abs"] = drc_abs
    df["DRC_coverage"] = drc_coverage
    df["tokens"] = tokens
    df["effort"] = effort

    return df


def rank_recipes(df,
                 w_jac=0.5, w_drc=0.25, w_mp=0.15, w_effort=0.10,
                 top_n=10):
    """
    Calcula un puntaje final por receta y devuelve los top n(parámetro) con mejor puntuación de forma descendiente.
    La puntuación máxima es de 1.

    Parámetros
    ----------
    df : pd.DataFrame
    w_jac, w_drc, w_mp, w_effort: los pesos para calcular la puntuación
    top_n: la cantidad de recetas a devolver

    Devuelve
    -------
    pd.DataFrame
        Copia de `df` ordenada por `final_score` (desc).
    """
    df = df.copy()
    df["final_score"] = (
        w_jac   * df["jaccard_score"] +
        w_drc   * df["DRC_coverage"] -
        w_mp    * df["missing_penalization"] -
        w_effort* df["effort"]
    )
    return df.sort_values("final_score", ascending=False).head(top_n)

# ------------------ LLM RAG -----------------

def safe_to_list(value):

    """
    Convierte entradas heterogéneas en una lista de strings consistente.

    Parámetros (acepta):
    -------------------
      - list o tuple: devuelve list(...) tal cual.
      - str que "parece" lista/tupla (p. ej., "['a','b']" o "('a','b')"):
          en estos casos se intenta parsear con ast.literal_eval de forma segura.
      - str simple con separadores por coma: "a, b, c" -> ["a", "b", "c"]
      - etc

    Ejemplos
    --------
    >>> safe_to_list(["Tomate", "  Leche  "])
    ['Tomate', 'Leche']
    >>> safe_to_list("['Tomate','Leche']")
    ['Tomate', 'Leche']
    """

    if isinstance(value, list):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if (stripped.startswith("[") and stripped.endswith("]")) or (stripped.startswith("(") and stripped.endswith(")")):
            try:
                parsed = ast.literal_eval(stripped)
                if isinstance(parsed, (list, tuple)):
                    return list(parsed)
            except Exception:
                pass
        return [item.strip() for item in value.split(",") if item.strip()]
    return []

def format_ingredients(value):
    """
    Normaliza y formatea ingredientes a una cadena legible

    - Internamente usa `safe_to_list` para obtener la lista homogénea.
    - Pensado para imprimir en prompts al LLM u outputs de usuario.

    Ejemplos
    --------
    >>> format_ingredients("['tomate','leche']")
    'tomate, leche'
    >>> format_ingredients(["tomate", "leche", "huevo"])
    'tomate, leche, huevo'
    """

    ingredients_list = safe_to_list(value) if value is not None else []
    return ", ".join(map(str, ingredients_list))

def format_directions(value, max_chars=900):
    """
    Unifica el formato de instrucciones de las receta en un único string legible
    y se pone un máximo de caracteres para controlar el tamaño en prompts/LLMs.

    Parámetros (acepta):
    -------------------
      - list de pasos -> se unen con espacios.
      - str que "parece" lista (p. ej. "['Paso 1','Paso 2']") -> se evalúa con
        `ast.literal_eval` y luego se unen los elementos si realmente es una lista.
      - str normal -> se devuelve tal cual (recortado si excede max_chars).

    Ejemplos
    --------
    >>> format_directions(["Corta el tomate.", "Añade la sal."])
    'Corta el tomate. Añade la sal.'
    >>> format_directions("['Corta','Mezcla','Sirve']")
    'Corta Mezcla Sirve'
    >>> format_directions("Texto largo...", max_chars=5)
    'Texto'
    """
    
    if isinstance(value, list):
        text = " ".join(map(str, value))
    elif isinstance(value, str):
        possible_list = None
        stripped = value.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                possible_list = ast.literal_eval(stripped)
            except Exception:
                possible_list = None
        text = " ".join(possible_list) if isinstance(possible_list, list) else value
    else:
        text = ""
    return text[:max_chars]

def make_context_from_top10(top10_df):

    """
    Construye un contexto para un LLM a partir de las 10 mejores recetas devueltas por el sistema de puntuación.
    Para cada fila/receta genera un bloque con:
      - Título
      - Ingredientes formateados (con `format_ingredients`)
      - Instrucciones normalizadas y recortadas (con `format_directions`)
      - Señales numéricas (métricas) redondeadas a 3 decimales:
        jaccard, DRC_coverage, missing_penalization, effort

    El resultado une todos los bloques listos para pasarlo al prompt del LLM.

    Parámetros
    ----------
    top10_df : pd.DataFrame
        DataFrame ordenado por ranking previamente (aquí se está asumiendo que es top 10).

    Devuelve
    -------
    str
        Un único string con 10 bloques (o tantos como filas tenga el DataFrame), separados
        por líneas "---", adecuado como contexto de entrada para el LLM.
    """
    blocks = []
    for _, row in top10_df.iterrows():
        title_text = str(row.get("title", "")).strip()
        ingredients_text = format_ingredients(row.get("ingredients"))
        directions_text = format_directions(row.get("directions"), max_chars=900)

        jaccard = float(row.get("jaccard_score", 0.0))
        drc_cov = float(row.get("DRC_coverage", 0.0))
        missing_pen = float(row.get("missing_penalization", 0.0))
        effort_norm = float(row.get("Effort", 0.0))

        block = (
            f"Title: {title_text}\n"
            f"Recipe ingredients: {ingredients_text}\n"
            f"Directions: {directions_text}\n"
            f"[signals] jaccard={jaccard:.3f} drc_coverage={drc_cov:.3f} "
            f"missing_penalization={missing_pen:.3f} effort={effort_norm:.3f}"
        )
        blocks.append(block)

    return "\n\n---\n\n".join(blocks)


def call_ollama_chat(
        model_name, 
        system_prompt_text, 
        user_prompt_text, 
        temperature=0.2, 
        num_ctx=8192):
    
    """
    Envía una conversación de 2 turnos (system + user) al servidor local de Ollama (el LLM usado en cuestión)
    y devuelve el contenido del mensaje de salida del modelo.

    El flujo
    --------
    - Construye el payload para POST /api/chat en http://localhost:11434.
    - Incluye 2 mensajes como parte de prompt engineering (los mensajes se definen posteriormente):
        1) system: se fija el comportamiento global (p. ej., “usa solo el contexto”, “responde en JSON”).
        2) user  : contiene un prompt simulando la llamade de un usuario pidiendo el contenido deseado de cierta forma.
        En este caso, el user prompt se usa más para formatear la salida ya que en la aplicación real, el usuario no manda ningún prompt.
        No interactúa directamente con el LLM si no que el LLM actúa como "juez final"
    - Ajusta opciones de generación: `temperature` (determina la aleatoriedad) y `num_ctx` (longitud ventana de contexto).

    Parámetros
    ----------
    model_name : str
        Nombre del modelo en Ollama
    system_prompt_text : str
        Mensaje de sistema (instrucciones globales).
    user_prompt_text : str
        Mensaje de usuario (tarea concreta).
    temperature : float, opcional
        Aleatoriedad de la generación (0 = más determinista).
    num_ctx : int, opcional
        Tamaño máximo de contexto en tokens

    Devuelve
    --------
    str
        Contenido textual de la respuesta del modelo
    """

    url = "http://localhost:11434/api/chat"
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt_text},
            {"role": "user",   "content": user_prompt_text}
        ],
        "options": {"temperature": temperature, "num_ctx": num_ctx},
        "stream": False
    }
    response = requests.post(url, json=payload, timeout=120)
    response.raise_for_status()
    return response.json()["message"]["content"]

def run_rag_over_top10_human_output(
    top10_df: pd.DataFrame,
    user_text,
    model_name = "llama3.1:8b",
    top_n_final = 3
):
    
    """
    Esta función actua como paso final para establecer el LLM RAG y une todas las funciones determinadas previamente.
    El objetivo final es devolver 3 recetas de las top 10 devueltas por el sistema de puntuación.

    Flujo
    --------
    1) Lee metadatos claves del dataframe de top n recetas(síntoma detectado, posibles
       deficiencias, ingredientes disponibles, etc.).
    2) Construye el contexto con las top 10 recetas candidatas (título, ingredientes e
       instrucciones) mediante `make_context_from_top10`.
    3) Se pasa al LLM:
       - un **system prompt** que fija reglas/criterios de evaluación,
       - un **user prompt** que especifica formato de salida y recordatorios (no inventar).
    4) Llama a `call_ollama_chat` (Ollama local) y devuelve `str` con la respuesta generada.

    Parámetros
    ----------
    top10_df : pd.DataFrame
        DataFrame con las top 10 recetas candidatas 
    user_text : 
        Texto original del usuario (descripción anímica/física) que se quiere reflejar en la salida.
    model_name : 
        Nombre del modelo en Ollama (por defecto se usa "llama3.1:8b").
    top_n_final : int
        Número de recetas finales que el LLM debe elegir (por defecto 3).

    Devuelve
    -------
    str
        Texto optimizado para un usuario final con las recetas elegidas y explicaciones, tal como lo devuelva el LLM.
    """

    first_row = top10_df.iloc[0]
    detected_symptom    = str(first_row.get("predicted_symptom", "")).strip()
    deficiency_list     = safe_to_list(first_row.get("deficiencia de nutrientes", []))
    deficiency_covered_by_ingredients  = safe_to_list(first_row.get("disponible en ingredientes", []))
    available_from_fridge  = safe_to_list(first_row.get("fridge_ingredients_available", []))
    available_ingredients = sorted({ing.lower() for ing in available_from_fridge})
    recipe_instructions = first_row.get('directions')
    recipe_title = first_row.get('title')
    recipe_ingredients = first_row.get('recipe_ingredients')

    context_text = make_context_from_top10(top10_df)

    system_prompt_text = (
        f"""
        Eres un asistente culinario y nutricional que se encargará de escoger las {top_n_final} mejores recetas para un usuario en base a los inputs del usuario y el contexto que te voy a dar.

        El input del usuario son los siguientes:
        Entrada del usuario: {user_text}
        Síntoma detectado: {detected_symptom}
        Ingredientes disponibles: {available_ingredients}

        El Contexto con las 10 recetas candidatas ({context_text}).
        Tarea: elige las {top_n_final} MEJORES recetas usando SOLO el contexto.
        Criterios:
        - Usa EXCLUSIVAMENTE la información del contexto proporcionado, no inventes nada
        - Tienes que verificar si la receta de verdad es para una comida. Es decir, descarta recetas de cremas, mascarillas y otras cosas que usan ingredientes de comida pero realmente no son para la nutrición
        - Tienes que verificar que la receta es realista para hacer, es decir que el usuario tiene bastantes ingredientes en la nevera de los requeridos por la receta
        - Tienes que verificar si la receta de verdad cubriría con la posible deficiencia nutricional asociada al síntoma detectado"""
    )

    
    user_prompt_text = (
    f"""
    Acuérdate que no puedes usar ninguna información que no existe o inventártelo.
    El texto que me tienes devolver tiene que seguir el siguiente formato:
        - Primero, me tienes que hablar directamente, no uses tercera persona
        - Segundo, dime que síntoma me detectas en base a mi texto 
          y que deficiencias nutriocionales puede que padezca en base a {deficiency_list}. (Asegúrate de mencionar que siempre es mejor visitar el médico si es grave) 
        - Tercero, dime los ingredientes detectados en mi nevera ({available_ingredients}) 
        - En base a esta información, recomiendame recetas:

        Para cada receta elegida, tienes que explicarme:
        - El nombre de la receta {recipe_title}
        - Los ingredientes necesarios de la receta {recipe_ingredients} (traducidos al español)
        - explica en 1–2 frases por qué ayuda al síntoma (cita qué alimentos o ingredientes cubren las deficiencias)
        - sugiere algunas sustituciones lógicas o sugiere qué ingredientes el usuario podría saltarse de la receta. Solo si el usuario no tiene el ingrediente de la nevera.
          No hace falta mencionar sustituciones de ingredientes comunes que todo el mundo tendría como agua, sal, aceite, etc.
        - Describir las instrucciones y los pasos de la receta (información disponible en {recipe_instructions})""")

    return call_ollama_chat(
        model_name=model_name,
        system_prompt_text=system_prompt_text,
        user_prompt_text=user_prompt_text,
        temperature=0.2  # baja temperatura = más consistente
        
    )