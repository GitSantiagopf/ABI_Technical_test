# AB InBev Technical Test – Data Scientist Project

La implementación se realizó de forma modular y orientada a objetos, incluye una documentación detallada de cada paso. Todos los informes, gráficos y directrices se almacenan en la carpeta `docs`.

## Visión General

Este proyecto está diseñado para resolver problemas de credit scoring y segmentación de clientes.
- **Credit Scoring:** Se desarrolló un pipeline que abarca el análisis exploratorio, preprocesamiento, feature engineering (incluyendo transformaciones logarítmicas), entrenamiento de modelos (utilizando GridSearchCV) y calibración de probabilidades. Además, se evaluaron distintos thresholds para ajustar la sensibilidad del modelo.
- **Segmentación de Clientes:** Se aplicaron algoritmos no supervisados (KMeans y DBSCAN) sobre las variables originales, escaladas mediante RobustScaler, para identificar grupos de clientes. Se generaron visualizaciones (boxplots, PCA, curvas de métricas) que permiten interpretar y extraer insights accionables para el negocio.

## Estructura del Proyecto

La estructura del repositorio es la siguiente:

```
AB-InBev-Technical-Test/
├── docs/
│   ├── guidelines.pdf         # Directrices y requisitos de la prueba técnica
│   ├── informe.pdf    # Informe final (en PDF)
│   └── report.pdf                    # Informe final en inglés (en PDF)
├── data_pipeline/
│   ├── data_loader.py         # Clase para cargar datos desde CSV
│   ├── data_cleaner.py        # Clase para limpiar e imputar datos
│   ├── data_explorer.py       # Clase para realizar análisis exploratorio y visualizar datos
│   ├── data_transformer.py    # Clase para escalar y transformar datos
│   └── feature_engineer.py    # Clase para crear y transformar variables
├── modeling/
│   ├── model_trainer.py       # Clase para entrenar modelos usando GridSearchCV
│   ├── model_evaluator.py     # Clase para evaluar modelos y generar plots de métricas
│   ├── model_calibrator.py    # Clase para calibrar modelos (sigmoide o isotónica)
│   ├── predictor.py           # Clase para generar predicciones y exportarlas a CSV
│   └── threshold_analyzer.py  # Clase para analizar la variación de métricas con diferentes thresholds y plotear matrices de confusión
├── main.py                    # Script principal que orquesta el flujo completo
├── requirements.txt           # Lista de dependencias necesarias
└── README.md                  #
```

## Instalación y Configuración

1. **Clonar el repositorio:**

   ```bash
   git clone https://github.com/GitSantiagopf/ABI_Technical_test.git
   cd AB-InBev-Technical-Test
   ```

2. **Crear y activar un entorno virtual (opcional pero recomendado):**

   ```bash
   python -m venv venv
   # En Windows:
   venv\Scripts\activate
   # En Unix o MacOS:
   source venv/bin/activate
   ```

3. **Instalar las dependencias:**

   ```bash
   pip install -r requirements.txt
   ```

## Ejecución

Para ejecutar el proyecto, simplemente ejecutar el script principal `main.py`:

```bash
python main.py
```

Durante la ejecución se generarán distintos plots (gráficos de distribución, heatmaps, matrices de confusión, curvas ROC, curvas de calibración, etc.) y se guardarán automáticamente en la carpeta `plots`. Además, se generarán archivos CSV con los resultados de la segmentación y las predicciones en el caso del modelo de Credit Scoring.

## Documentación

- **Directrices de la Prueba Técnica:** Se encuentran en el archivo `docs/guidelines.pdf`.
- **Informe Ejecutivo:** Se ha elaborado un informe ejecutivo que resume el proceso y los resultados obtenidos tanto para el Credit Scoring como para la Segmentación de Clientes. Este informe se puede encontrar en `docs/informe.pdf`.

---
