# -*- coding: utf-8 -*-
"""
TFM - App de demostración (ampliada): predicción de demanda eléctrica en España

Ejecutar con: streamlit run app_tfm.py

pip install streamlit plotly shap --break-system-packages
"""

import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import shap
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Demanda eléctrica España - TFM",
    page_icon="⚡",
    layout="wide"
)

# =============================================================
# CARGA DE DATOS Y MODELOS
# =============================================================
@st.cache_resource
def cargar_modelos():
    modelo_xgb = joblib.load("modelo_xgb.pkl")
    modelo_lgbm = joblib.load("modelo_lgbm.pkl")
    with open("features_modelo.json") as f:
        features = json.load(f)
    explainer = shap.TreeExplainer(modelo_xgb)
    return modelo_xgb, modelo_lgbm, features, explainer


@st.cache_data
def cargar_datos():
    pred_horarias = pd.read_csv("predicciones_horarias.csv")
    pred_horarias["datetime"] = pd.to_datetime(pred_horarias["datetime"], utc=True).dt.tz_convert("Europe/Madrid")

    pred_diarias = pd.read_csv("predicciones_diarias.csv")
    pred_diarias["fecha"] = pd.to_datetime(pred_diarias["fecha"], utc=True).dt.tz_convert("Europe/Madrid")

    comparativa = pd.read_csv("comparativa_modelos.csv")
    importancias = pd.read_csv("importancia_variables.csv")
    return pred_horarias, pred_diarias, comparativa, importancias


@st.cache_data
def cargar_temperatura_mapa():
    temp_ciudades = pd.read_csv("temperatura_por_ciudad.csv", parse_dates=["fecha"])
    coordenadas = pd.read_csv("coordenadas_ciudades.csv")
    return temp_ciudades, coordenadas


modelo_xgb, modelo_lgbm, FEATURES, explainer = cargar_modelos()
pred_horarias, pred_diarias, comparativa, importancias = cargar_datos()

# Residuos precalculados (para bandas de incertidumbre y análisis de errores)
pred_horarias["error_xgb"] = pred_horarias["demanda_mw"] - pred_horarias["pred_xgboost"]
pred_horarias["error_lgbm"] = pred_horarias["demanda_mw"] - pred_horarias["pred_lightgbm"]
pred_horarias["error_abs_pct_xgb"] = (pred_horarias["error_xgb"].abs() / pred_horarias["demanda_mw"]) * 100

# Bandas de incertidumbre empíricas: cuantiles 5%/95% del error, por hora del día
# (la varianza del error no es igual a las 4am que a las 9am, así que se calcula por hora)
bandas_por_hora = (
    pred_horarias.groupby("hour")["error_xgb"]
    .quantile([0.05, 0.95])
    .unstack()
    .rename(columns={0.05: "q05", 0.95: "q95"})
)

# =============================================================
# CABECERA
# =============================================================
st.title("⚡ Predicción de demanda eléctrica en España")
st.caption("TFM · Datos de Red Eléctrica de España (REE) y AEMET · 2019-2025")

tabs = st.tabs([
    "🔮 Predicción interactiva",
    "🌡️ Simulador de escenarios",
    "🧩 SHAP por día",
    "📉 Errores por segmento",
    "📊 Comparativa de modelos",
    "🧠 Importancia de variables",
    "🗺️ Mapa de temperaturas",
])
(tab_prediccion, tab_simulador, tab_shap_dia, tab_errores,
 tab_comparativa, tab_importancia, tab_mapa) = tabs

# =============================================================
# PESTAÑA 1: PREDICCIÓN INTERACTIVA (ampliada)
# =============================================================
with tab_prediccion:
    st.subheader("Elige una fecha del periodo de test")

    fechas_disponibles = sorted(pred_horarias["datetime"].dt.date.unique())
    fecha_sel = st.selectbox("Fecha", fechas_disponibles, key="fecha_pred")
    modelos_a_mostrar = st.multiselect(
        "Modelos a mostrar", ["XGBoost", "LightGBM"], default=["XGBoost", "LightGBM"]
    )
    mostrar_banda = st.checkbox("Mostrar banda de incertidumbre (XGBoost, percentil 5-95%)", value=True)

    dia_data = pred_horarias[pred_horarias["datetime"].dt.date == fecha_sel].sort_values("datetime").copy()

    if not dia_data.empty:
        dia_data = dia_data.merge(bandas_por_hora, on="hour", how="left")

        fig = go.Figure()

        if mostrar_banda:
            fig.add_trace(go.Scatter(
                x=pd.concat([dia_data["datetime"], dia_data["datetime"][::-1]]),
                y=pd.concat([dia_data["pred_xgboost"] + dia_data["q95"], (dia_data["pred_xgboost"] + dia_data["q05"])[::-1]]),
                fill="toself", fillcolor="rgba(214,39,40,0.12)",
                line=dict(color="rgba(255,255,255,0)"),
                name="Banda XGBoost (5-95%)", showlegend=True, hoverinfo="skip"
            ))

        fig.add_trace(go.Scatter(
            x=dia_data["datetime"], y=dia_data["demanda_mw"],
            name="Real", mode="lines+markers", line=dict(color="#1f77b4", width=3)
        ))
        if "XGBoost" in modelos_a_mostrar:
            fig.add_trace(go.Scatter(
                x=dia_data["datetime"], y=dia_data["pred_xgboost"],
                name="XGBoost", mode="lines+markers", line=dict(color="#d62728", width=2, dash="dash")
            ))
        if "LightGBM" in modelos_a_mostrar:
            fig.add_trace(go.Scatter(
                x=dia_data["datetime"], y=dia_data["pred_lightgbm"],
                name="LightGBM", mode="lines+markers", line=dict(color="#2ca02c", width=2, dash="dot")
            ))

        fig.update_layout(
            title=f"Demanda real vs. predicha — {fecha_sel}",
            xaxis_title="Hora", yaxis_title="Demanda (MW)",
            hovermode="x unified", height=450
        )
        st.plotly_chart(fig, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Tipo de día", dia_data["tipo_dia"].iloc[0].capitalize())
        c2.metric("MAE XGBoost", f"{dia_data['error_xgb'].abs().mean():,.0f} MW")
        c3.metric("MAPE XGBoost", f"{dia_data['error_abs_pct_xgb'].mean():.2f}%")

        st.download_button(
            "⬇️ Descargar predicciones de este día (CSV)",
            data=dia_data.to_csv(index=False).encode("utf-8"),
            file_name=f"predicciones_{fecha_sel}.csv",
            mime="text/csv"
        )
    else:
        st.warning("No hay datos de test para esa fecha.")

# =============================================================
# PESTAÑA 2: SIMULADOR DE ESCENARIOS
# =============================================================
with tab_simulador:
    st.subheader("¿Y si la temperatura fuera distinta?")
    st.write(
        "Elige un día real y ajusta la temperatura para ver cómo cambiaría "
        "la predicción del modelo — útil para explorar el efecto de una ola "
        "de calor o una ola de frío sobre la demanda estimada."
    )

    fecha_sim = st.selectbox("Fecha base", fechas_disponibles, key="fecha_sim")
    delta_temp = st.slider("Variación de temperatura (°C)", -10.0, 10.0, 0.0, 0.5)

    dia_sim = pred_horarias[pred_horarias["datetime"].dt.date == fecha_sim].sort_values("datetime").copy()

    if not dia_sim.empty:
        X_original = dia_sim[FEATURES].copy()
        X_modificado = X_original.copy()
        X_modificado["temperatura_c"] = X_modificado["temperatura_c"] + delta_temp

        pred_original = modelo_xgb.predict(X_original)
        pred_modificada = modelo_xgb.predict(X_modificado)

        fig_sim = go.Figure()
        fig_sim.add_trace(go.Scatter(
            x=dia_sim["datetime"], y=dia_sim["demanda_mw"],
            name="Real", line=dict(color="#1f77b4", width=3)
        ))
        fig_sim.add_trace(go.Scatter(
            x=dia_sim["datetime"], y=pred_original,
            name="Predicción original", line=dict(color="#7f7f7f", width=2, dash="dash")
        ))
        fig_sim.add_trace(go.Scatter(
            x=dia_sim["datetime"], y=pred_modificada,
            name=f"Predicción con {delta_temp:+.1f}°C", line=dict(color="#d62728", width=3)
        ))
        fig_sim.update_layout(
            title=f"Simulación — {fecha_sim} ({delta_temp:+.1f}°C)",
            xaxis_title="Hora", yaxis_title="Demanda (MW)",
            hovermode="x unified", height=450
        )
        st.plotly_chart(fig_sim, use_container_width=True)

        impacto_medio = (pred_modificada - pred_original).mean()
        st.metric(
            "Impacto medio en la demanda estimada",
            f"{impacto_medio:+,.0f} MW",
            help="Diferencia media entre la predicción original y la simulada con el ajuste de temperatura."
        )
        st.caption(
            "Nota: el resto de variables (lags, calendario) se mantienen fijas al valor real "
            "del día elegido — esto simula únicamente el efecto de la temperatura, no un "
            "escenario completo con cascada de efectos sobre días posteriores."
        )
    else:
        st.warning("No hay datos de test para esa fecha.")

# =============================================================
# PESTAÑA 3: SHAP POR DÍA INDIVIDUAL
# =============================================================
with tab_shap_dia:
    st.subheader("¿Por qué el modelo predijo esto para un día concreto?")

    fecha_shap = st.selectbox("Fecha", fechas_disponibles, key="fecha_shap")
    hora_shap = st.slider("Hora del día", 0, 23, 12, key="hora_shap")

    fila = pred_horarias[
        (pred_horarias["datetime"].dt.date == fecha_shap) & (pred_horarias["hour"] == hora_shap)
    ]

    if not fila.empty:
        X_fila = fila[FEATURES]
        shap_values_fila = explainer(X_fila)

        col1, col2 = st.columns([2, 1])
        with col1:
            fig_waterfall, ax = plt.subplots(figsize=(9, 6))
            shap.plots.waterfall(shap_values_fila[0], show=False, max_display=10)
            st.pyplot(fig_waterfall)
            plt.close(fig_waterfall)
        with col2:
            st.metric("Demanda real", f"{fila['demanda_mw'].values[0]:,.0f} MW")
            st.metric("Predicción XGBoost", f"{fila['pred_xgboost'].values[0]:,.0f} MW")
            st.metric("Tipo de día", fila["tipo_dia"].values[0].capitalize())

        st.caption(
            "El gráfico muestra cómo cada variable empuja la predicción hacia arriba (rojo) "
            "o hacia abajo (azul) respecto al valor base del modelo, para esta hora concreta."
        )
    else:
        st.warning("No hay datos para esa combinación de fecha y hora.")

# =============================================================
# PESTAÑA 4: ANÁLISIS DE ERRORES POR SEGMENTO
# =============================================================
with tab_errores:
    st.subheader("¿Dónde falla más el modelo?")

    col1, col2 = st.columns(2)

    with col1:
        error_por_tipo_dia = (
            pred_horarias.groupby("tipo_dia")["error_abs_pct_xgb"].mean().reset_index()
        )
        fig_tipo = px.bar(
            error_por_tipo_dia, x="tipo_dia", y="error_abs_pct_xgb",
            text_auto=".2f", title="MAPE medio por tipo de día (XGBoost)",
            color="tipo_dia"
        )
        fig_tipo.update_layout(showlegend=False, height=380, yaxis_title="MAPE (%)")
        st.plotly_chart(fig_tipo, use_container_width=True)

    with col2:
        error_por_hora = (
            pred_horarias.groupby("hour")["error_abs_pct_xgb"].mean().reset_index()
        )
        fig_hora = px.line(
            error_por_hora, x="hour", y="error_abs_pct_xgb", markers=True,
            title="MAPE medio por hora del día (XGBoost)"
        )
        fig_hora.update_layout(height=380, xaxis_title="Hora", yaxis_title="MAPE (%)")
        st.plotly_chart(fig_hora, use_container_width=True)

    st.subheader("Peores días del periodo de test")
    peores_dias = (
        pred_horarias.groupby(pred_horarias["datetime"].dt.date)["error_abs_pct_xgb"]
        .mean().sort_values(ascending=False).head(10).reset_index()
    )
    peores_dias.columns = ["fecha", "MAPE medio (%)"]
    st.dataframe(peores_dias.style.format({"MAPE medio (%)": "{:.2f}%"}), use_container_width=True)
    st.caption("Consulta estos días en la pestaña de Predicción interactiva o SHAP por día para entender por qué fallan más.")

# =============================================================
# PESTAÑA 5: COMPARATIVA DE MODELOS
# =============================================================
with tab_comparativa:
    st.subheader("Comparativa de métricas de error por modelo")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.dataframe(
            comparativa.style.format({"MAE": "{:,.1f}", "RMSE": "{:,.1f}", "MAPE": "{:.2f}%"}),
            use_container_width=True
        )
    with col2:
        fig_mape = px.bar(
            comparativa.sort_values("MAPE"), x="modelo", y="MAPE",
            text_auto=".2f", title="MAPE por modelo (%)", color="modelo"
        )
        fig_mape.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig_mape, use_container_width=True)

    st.subheader("Predicción vs. real — periodo completo de test (diario)")
    fig_diario = go.Figure()
    fig_diario.add_trace(go.Scatter(
        x=pred_diarias["fecha"], y=pred_diarias["demanda_real"],
        name="Real", line=dict(color="black", width=2)
    ))
    fig_diario.add_trace(go.Scatter(
        x=pred_diarias["fecha"], y=pred_diarias["pred_sarimax"],
        name="SARIMAX (rolling)", line=dict(color="#ff7f0e", width=2, dash="dash")
    ))
    fig_diario.update_layout(xaxis_title="Fecha", yaxis_title="Demanda media diaria (MW)", height=400)
    st.plotly_chart(fig_diario, use_container_width=True)

# =============================================================
# PESTAÑA 6: IMPORTANCIA DE VARIABLES
# =============================================================
with tab_importancia:
    st.subheader("¿Qué variables pesan más en la predicción? (XGBoost, global)")
    fig_imp = px.bar(
        importancias.sort_values("importancia"),
        x="importancia", y="variable", orientation="h",
        title="Importancia de variables"
    )
    fig_imp.update_layout(height=500)
    st.plotly_chart(fig_imp, use_container_width=True)
    st.caption(
        "Esta es la importancia global (todo el conjunto de test). Para ver la explicación "
        "de una predicción concreta, usa la pestaña 'SHAP por día'."
    )

# =============================================================
# PESTAÑA 7: MAPA DE TEMPERATURAS
# =============================================================
with tab_mapa:
    st.subheader("Temperatura por ciudad — periodo de test")

    temp_ciudades, coordenadas = cargar_temperatura_mapa()
    fechas_mapa = sorted(temp_ciudades["fecha"].dt.date.unique())
    fecha_mapa_sel = st.select_slider("Fecha", options=fechas_mapa, value=fechas_mapa[-1])

    dia_mapa = temp_ciudades[temp_ciudades["fecha"].dt.date == fecha_mapa_sel]
    dia_mapa = dia_mapa.merge(coordenadas, on="ciudad", how="left")

    if not dia_mapa.empty:
        fig_mapa = px.scatter_mapbox(
            dia_mapa, lat="lat", lon="lon",
            size="poblacion", color="temp_media_c",
            hover_name="ciudad", hover_data={"temp_media_c": ":.1f", "lat": False, "lon": False},
            color_continuous_scale="RdYlBu_r",
            size_max=30, zoom=4.2,
            center={"lat": 40.0, "lon": -3.7},
            mapbox_style="carto-positron",
            title=f"Temperatura media — {fecha_mapa_sel}"
        )
        fig_mapa.update_layout(height=550, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig_mapa, use_container_width=True)
        st.caption(
            "El tamaño del punto representa el peso poblacional de la ciudad en la "
            "temperatura nacional ponderada usada por el modelo; el color, la temperatura del día."
        )
    else:
        st.warning("No hay datos de temperatura para esa fecha.")

st.divider()
st.caption("TFM — Predicción de demanda eléctrica en España · Datos: REData API (REE) y AEMET OpenData")