# Life Expectancy Prediction (Equipo 4)

Este es un proyecto End-to-End de Machine Learning enfocado en predecir la Expectativa de Vida basándose en datos del "Life Expectancy Dataset" proveniente de la OMS.

## Estructura del Repositorio

- `data/`: Contiene el archivo dataset `Life Expectancy Data.csv`.
- `src/`: Scripts del pipeline de datos:
  - `preprocessing.py`: Carga el dataset, limpia nombres de columnas, elimina nulos y trata outliers.
  - `features.py`: Aplica transformaciones logarítmicas a campos asimétricos, variables derivadas (índices), y OneHotEncode/StandardScaler.
  - `train.py`: Separa los datos, entrena el modelo de Regresión Multilineal, lo evalúa y serializa.
  - `predict.py`: Lógica principal para obtener la predicción cargando el artefacto `.pkl` guardado.
- `model/`: Contiene el modelo y el preprocesador empaquetados en `model.pkl`.
- `app/`: Contiene `app.py`, una API en Flask para realizar inferencias.
- `requirements.txt`: Dependencias necesarias (Pandas, Numpy, Scikit-learn, Flask, etc.)
- `.ebextensions/`, `buildspec.yml` y `application.py`: Configuraciones de CI/CD para AWS Elastic Beanstalk y CodeBuild/CodePipeline.

## Instrucciones de Instalación (Local)

1. **Crear e inicializar un entorno virtual**
   ```bash
   python -m venv venv
   # En Windows:
   venv\Scripts\activate
   # En Mac/Linux:
   source venv/bin/activate
   ```

2. **Instalar dependencias**
   ```bash
   pip install -r requirements.txt
   ```

3. **Pipelines de Entrenamiento**
   ```bash
   # Preprocesado, feature engineering, y entrenamiento del modelo (Exporta model/model.pkl)
   python src/train.py
   
   # Probar la inferencia desde consola
   python src/predict.py
   ```

4. **Inicializar Servidor FLASK de la API**
   ```bash
   # Ejecutar
   python app/app.py
   ```
   La API levantará en `http://localhost:5000/`. La ruta es `POST /predict`.

   **Ejemplo de JSON de Entrada:**
   ```json
   {
       "Country": "Afghanistan",
       "Year": 2015,
       "Status": "Developing",
       "Adult Mortality": 263.0,
       "infant deaths": 62,
       "Alcohol": 0.01,
       "percentage expenditure": 71.27,
       "Hepatitis B": 65.0,
       "Measles": 1154,
       "BMI": 19.1,
       "under-five deaths": 83,
       "Polio": 6.0,
       "Total expenditure": 8.16,
       "Diphtheria": 65.0,
       "HIV/AIDS": 0.1,
       "GDP": 584.25,
       "Population": 33736494.0,
       "thinness 1-19 years": 17.2,
       "thinness 5-9 years": 17.3,
       "Income composition of resources": 0.479,
       "Schooling": 10.1
   }
   ```
