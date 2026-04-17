import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import google.generativeai as genai

# Configuración de la página
st.set_page_config(page_title="Proyecto Final Estadística", layout="wide")

# Estado de la sesión para mantener los datos al navegar
if 'df' not in st.session_state:
    st.session_state.df = None
if 'resumen_stats' not in st.session_state:
    st.session_state.resumen_stats = ""

st.title("📊 App Estadística: Visualización e IA")
st.markdown("---")

# Menú lateral
with st.sidebar:
    st.header("Secciones")
    modo = st.radio("Selecciona:", ["Carga de Datos", "Visualización", "Prueba de Hipótesis", "Consultar IA"])

# --- MODULO 1: CARGA ---
if modo == "Carga de Datos":
    st.header("📂 Carga de Datos")
    opcion = st.radio("Origen:", ["Generar Datos Sintéticos (Normal)", "Subir archivo CSV"])
    
    if opcion == "Generar Datos Sintéticos (Normal)":
        n = st.number_input("Tamaño de muestra (n)", min_value=30, value=100)
        media = st.number_input("Media deseada", value=50.0)
        desv = st.number_input("Desviación estándar", value=10.0)
        if st.button("Generar Datos"):
            datos = np.random.normal(media, desv, n)
            st.session_state.df = pd.DataFrame(datos, columns=["Variable_Analizada"])
            st.success("¡Datos generados correctamente!")
    
    else:
        archivo = st.file_uploader("Sube tu archivo CSV", type=['csv'])
        if archivo is not None:
            st.session_state.df = pd.read_csv(archivo)
            st.success("Archivo cargado.")

    if st.session_state.df is not None:
        st.write("Vista previa de los datos:", st.session_state.df.head())

# --- MODULO 2: VISUALIZACIÓN ---
elif modo == "Visualización":
    st.header("📈 Visualización de Distribuciones")
    if st.session_state.df is not None:
        col = st.selectbox("Selecciona columna:", st.session_state.df.columns)
        
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Histograma y KDE")
            fig1, ax1 = plt.subplots()
            sns.histplot(st.session_state.df[col], kde=True, ax=ax1, color="skyblue")
            st.pyplot(fig1)
        with c2:
            st.subheader("Boxplot")
            fig2, ax2 = plt.subplots()
            sns.boxplot(y=st.session_state.df[col], ax=ax2, color="lightgreen")
            st.pyplot(fig2)
    else:
        st.error("⚠️ Carga datos primero.")

# --- MODULO 3: PRUEBA DE HIPÓTESIS ---
elif modo == "Prueba de Hipótesis":
    st.header("🧪 Prueba Z de una Muestra")
    if st.session_state.df is not None:
        col = st.selectbox("Variable para la prueba:", st.session_state.df.columns)
        mu0 = st.number_input("Media Hipotética (H0):", value=50.0)
        alpha = st.selectbox("Nivel de significancia (α):", [0.01, 0.05, 0.10], index=1)
        
        # Cálculos
        datos = st.session_state.df[col]
        x_bar = datos.mean()
        s = datos.std()
        n = len(datos)
        z_stat = (x_bar - mu0) / (s / np.sqrt(n))
        p_val = stats.norm.sf(abs(z_stat)) * 2 # Prueba bilateral

        # Mostrar métricas
        m1, m2, m3 = st.columns(3)
        m1.metric("Media Muestral", f"{x_bar:.2f}")
        m2.metric("Estadístico Z", f"{z_stat:.4f}")
        m3.metric("P-Value", f"{p_val:.4f}")

        if p_val < alpha:
            st.error(f"Se RECHAZA la hipótesis nula (H0). El p-value ({p_val:.4f}) es menor que alfa.")
        else:
            st.success(f"NO se rechaza H0. El p-value ({p_val:.4f}) es mayor que alfa.")
        
        # Guardar para la IA
        st.session_state.resumen_stats = f"Media={x_bar:.2f}, n={n}, Z={z_stat:.4f}, p={p_val:.4f}, alfa={alpha}"
    else:
        st.error("⚠️ Carga datos primero.")

# --- MODULO 4: IA ---
elif modo == "Consultar IA":
    st.header("🤖 Análisis con Gemini IA")
    api_key = st.text_input("Pega tu Gemini API Key:", type="password")
    
    if st.button("Pedir análisis a la IA"):
        if api_key and st.session_state.resumen_stats:
            try:
                # 1. Configuramos la API (Agregamos la versión de transporte)
                genai.configure(api_key=api_key)
                
                # 2. Usamos el nombre del modelo más estable para la versión gratuita
                model = genai.GenerativeModel('gemini-1.5-flash')
                
                prompt = f"Analiza estos resultados estadísticos: {st.session_state.resumen_stats}. Explica en términos sencillos si la decisión de rechazar o no la hipótesis fue correcta y qué significa."
                
                with st.spinner("La IA está pensando..."):
                    # 3. Intentamos generar el contenido
                    response = model.generate_content(prompt)
                    st.markdown("### Conclusión de la IA:")
                    st.write(response.text)
            except Exception as e:
                st.error(f"Hubo un detalle: {e}")