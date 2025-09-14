# Health Bite
Este repositorio contiene el código completo del proyecto fin de máster que se centra en el desarrollo de HealthBite, una aplicación inteligente que combina visión artificial, procesamiento de lenguaje natural y modelos de lenguaje de gran escala (LLMs) para ofrecer recomendaciones de recetas personalizadas. La propuesta introduce tres factores innovadores principalmente: (1) la capacidad de detectar automáticamente los alimentos disponibles en la nevera del usuario, (2) la opción de inferir posibles síntomas o deficiencias nutricionales a partir de una breve descripción de estado anímico o físico y (3) una comunicación transparente e interactiva con el usuario utilizando la tecnología de LLMs que permite al usuario comprender la razón detrás de cada recomendación.

La aplicación sigue el siguiente flujo:
![Flujo de la aplicación](/final%20app/app%20flowchart.jpg)

El flujo de uso de la aplicación es sencillo para el usuario: basta con describir su estado anímico o físico y subir una fotografía de su nevera. Con esta información, intervienen de forma integrada los distintos modelos. En primer lugar, un modelo de visión artificial (YOLO), entrenado con más de 17 000 imágenes correspondientes a 30 clases únicas de alimentos, identifica los ingredientes disponibles. Paralelamente, un modelo deprocesamiento de lenguaje natural (NLP), entrenado con descripciones en español de estados anímicos y físicos, infiere y predice posibles síntomas junto con las deficiencias nutricionales asociadas. Los resultados de ambos modelos se combinan en un algoritmo de puntuación que evalúa un conjunto de más de dos millonesde recetas, priorizando aquellas que mejor se ajustan al contexto específico del usuario. Finalmente, una capa adicional basada en un LLM con RAG revisa las recetas mejor puntuadas y actúa como agente de razonamiento: no solo selecciona las opciones definitivas que se recomendarán, sino que también proporciona explicaciones claras y justificadas sobre por qué fueron elegidas y cómo responden a las necesidades particulares del individuo. Durante el desarrollo, se comprobó que los modelos construidos alcanzan un rendimiento sólido y generan predicciones fiables, validando así la viabilidad del sistema propuesto.

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
   
3.**Crea el entorno Conda a partir de `environment.yml`**:
   ```bash
   conda env create -f environment.yml


