# Predicción de la demanda eléctrica en España mediante técnicas de Machine Learning

Trabajo de Fin de Máster — Máster en Big Data, Data Science e Inteligencia Artificial, Universidad Complutense de Madrid (curso 2025/2026).

**Autora**: Almina Ovaeva

## Descripción

Modelo de predicción de la demanda eléctrica horaria peninsular en España (2019-2025), a partir de datos públicos de Red Eléctrica de España (REE) y de la Agencia Estatal de Meteorología (AEMET). Se comparan tres modelos predictivos de naturaleza distinta —SARIMAX, XGBoost y LightGBM—, complementados con un análisis de interpretabilidad (SHAP) y una aplicación interactiva de productivización.

## Contenido del repositorio

- `tfm.ipynb` — notebook principal: extracción de datos, feature engineering, análisis exploratorio, modelización y evaluación.
- `app_tfm.py` — aplicación interactiva (Streamlit) de predicción, simulación de escenarios e interpretabilidad.
- `requirements.txt` — dependencias necesarias.

La memoria completa y los anexos técnicos se entregan como documento aparte junto a este repositorio.

## Cómo ejecutarlo

### 1. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 2. Configurar la API key de AEMET

Este proyecto necesita una clave gratuita de la API OpenData de AEMET. Crea un archivo `.env` en la raíz del proyecto (no incluido en este repositorio por seguridad) con el siguiente contenido:

```
AEMET_API_KEY=tu_clave_aqui
```

La API de REE (REData) no requiere autenticación.

### 3. Ejecutar el notebook

```bash
jupyter notebook tfm.ipynb
```

Ejecuta las celdas en orden. La extracción completa del histórico de REE y AEMET puede tardar entre 20 y 40 minutos por los límites de peticiones de ambas APIs.

### 4. Ejecutar la aplicación interactiva

Tras ejecutar el notebook completo (que genera los modelos entrenados y los datos que usa la app):

```bash
streamlit run app_tfm.py
```

## Resultados principales

| Modelo | MAE (MW) | RMSE (MW) | MAPE |
|---|---|---|---|
| SARIMAX (rolling) | 839,3 | 1.318,8 | **3,16%** |
| LightGBM | 596,9 | 1.059,6 | 3,23% |
| XGBoost | 601,9 | 1.083,7 | 3,28% |
| Naive (persistencia 24h) | 1.819,9 | 2.817,7 | 7,62% |

## Fuentes de datos

- [REData API (REE)](https://www.ree.es/es/apidatos)
- [AEMET OpenData](https://opendata.aemet.es)
