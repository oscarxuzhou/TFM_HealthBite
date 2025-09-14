# Health Bite
Este repositorio contiene el código completo del proyecto fin de máster que se centra en el desarrollo de HealthBite, una aplicación inteligente que combina visión artificial, procesamiento de lenguaje natural y modelos de lenguaje de gran escala (LLMs) para ofrecer recomendaciones de recetas personalizadas. La propuesta introduce tres factores innovadores principalmente: (1) la capacidad de detectar automáticamente los alimentos disponibles en la nevera del usuario, (2) la opción de inferir posibles síntomas o deficiencias nutricionales a partir de una breve descripción de estado anímico o físico y (3) una comunicación transparente e interactiva con el usuario utilizando la tecnología de LLMs que permite al usuario comprender la razón detrás de cada recomendación.

La aplicación sigue el siguiente flujo:
![Flujo de la aplicación](/final%20app/app%20flowchart.jpg)

El flujo de uso de la aplicación es sencillo para el usuario: 
- El usuario describe su estado anímico o físico
- El usuario sube una fotografía de su nevera
- El usuario aprieta el botón de "Recomiéndame recetas" 
Tras estas acciones, el usuario puede obtener recetas personalizadas en base a su contexto.
---

## Guía de Instalación y Uso

1. **Descargar los archivos necesarios desde la sección** [Releases](https://github.com/oscarxuzhou/TFM_HealthBite/releases):  
   - `recipes_dataset_translated.parquet`  
   - `NLP.model.zip`
Estos archivos contienen el modelo NLP y el dataset de recetas que debido a su tamaño, no se podían subir directamente al repositorio como archivos normales

2. **Clona este repositorio**:
   ```bash
   git clone https://github.com/oscarxuzhou/TFM_HealthBite.git
   cd TFM_HealthBite
   
3. **Crea el entorno Conda a partir de `environment.yml`:**
   ```bash
   conda env create -f environment.yml

4. **Activa el entorno**:
   ```bash
   conda activate healthbite

5. **Ajusta la ruta de los archivos**:
   Se necesita ajustar las rutas del modelo NLP y dataset de recetas en `final app/app_key_functions.py` a la nueva ubicación del archivo después de descargarlos en el paso 1

6. **Ejecuta la aplicación de streamlit en Anaconda Prompt**:
   ```bash
   streamlit run "final app/HealthBite_app.py"
