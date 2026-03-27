# Documentación de Uso de Inteligencia Artificial

Cumpliendo con los requerimientos del proyecto, se expone y documenta detalladamente el uso de herramientas de Inteligencia Artificial (IA) en el desarrollo del sistema.

## 1. Qué parte del código fue generada por IA
La IA asistió en la generación de la estructura base del código ("boilerplate") en los siguientes módulos:
- `src/preprocessing.py`: Scripts de limpieza de nombres de columnas y estrategias base de imputación para nulos y tratamiento de outliers (IQR clipping).
- `src/features.py`: Código para aplicar transformaciones logarítmicas a distribuciones asimétricas (sesgos positivos en GDP y Población) y la sintaxis para configurar el `ColumnTransformer` (Pipeline) de `scikit-learn`.
- `app/app.py`: La sintaxis del enrutamiento de Flask (setup del endpoint genérico POST `/predict`).

La lógica de negocio principal, el ajuste de hiperparámetros, el diseño del feature engineering, y la configuración de CI/CD para integrarse explícitamente con los servicios (AWS) requirieron intervención iterativa manual. Estimamos que la contribución genérica de la IA abarca un aproximado del 20-30% del código en su totalidad, manteniéndose dentro del límite del 30% especificado.

## 2. Por qué se utilizó IA en esa sección del proyecto
- **Rapidez y eficiencia**: En lugar de escribir el "boilerplate" para crear la estructura de carpetas, definir los componentes sintácticos para el enrutamiento de Flask o implementar las matemáticas estándar del rango intercuartílico (IQR) desde cero, la IA se usó para ahorrar tiempo y concentrar a los integrantes del equipo en las etapas críticas analíticas, de modelado multilineal y en el setup en AWS.
- **Consultas sobre mejores prácticas**: Se consultó a la IA sobre las metodologías estándar en la industria para preservar el `ColumnTransformer` utilizando serialización (`joblib`) asegurando la compatibilidad de inferencia directamente desde la API.

## 3. Pruebas realizadas sobre ese código
Todo el código sugerido fue revisado y validado metodológicamente:
1. **Pruebas de Transformación**: Se validó iterativamente que el `ColumnTransformer` no destruyera las columnas originales enviadas a través de la API testando el array numpy contra los nombres recuperados vía `.get_feature_names_out()`.
2. **Pruebas de Inferencia Unitarias**: Se procesaron simulaciones con la fila correspondiente a `Afganistán 2015` localmente validando que la API en Flask en efecto regresara `~60.7` aproximándose al valor original.
3. **Pruebas de Outliers**: Se auditó el script validando que el valor máximo de GDP, por ejemplo, en efecto estuviese acotado en el Q3 + 1.5 IQR y no devolviera valores nulos en el clipping.

## 4. Casos Límite Analizados
- **Espaciados incoherentes en columnas del Dataset**: El dataset original traía espaciados irregulares (por ejemplo `" thinness  1-19 years"` o `"Life expectancy "`). Analizamos cómo la IA sugería limpiar las columnas y se adaptó `predict.py` para asegurar que el JSON de entrada con las llaves que proporciona el front-end no tirara errores de compatibilidad en Pandas (`KeyError`).
- **Valores negativos para el Log Transform**: La transformación logarítmica sugerida arrojaba excepciones si encontraba datos atípicamente escalados a negativo; por ende, se corrigió insertando protecciones (`np.log1p(df[col] - min_val)`) en `apply_log_transform`.
- **Excepciones en Flask**: Protegimos el endpoint revisando los escenarios donde el usuario pudiera olvidarse de mandar algunos hiperparámetros numéricos en el body POST simulando el comportamiento de NA-filling (llenado a '0' o iterando mediana). 

## 5. Errores Encontrados y Correcciones
- **Error en el nombre de las columnas después del preprocesamiento**: Al recuperar las variables dummy usando OneHotEncoder, la IA intentaba cruzar los nombres generados directamente al dataframe de pandas, produciendo un desajuste del tamaño (`ValueError: Length mismatch`). ¿Cómo se corrigió?: Investigando la documentación de `sklearn`, usamos la variable `named_transformers_['cat'].get_feature_names_out(categorical_features)` e incluimos las variables concatenando las listas de features.
- **Serialización desalineada**: La IA propuso inicialmente generar un pipeline para la Data y exportar el modelo por otro lado; lo corregimos exportando tanto `model` como `preprocessor` en un mismo diccionario iterativo para evitar desincronizaciones durante la inducción real usando `app.py`.
