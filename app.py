import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import google.generativeai as genai

# --- CONFIGURACIÓN VISUAL ---
plt.style.use('ggplot') 
st.set_page_config(page_title="Proyecto Estadística U3", layout="wide")

if 'df' not in st.session_state:
    st.session_state.df = None
if 'api_key' not in st.session_state:
    st.session_state.api_key = "AQ.Ab8RN6KA-pXty5iJYUvwbLgZiCxtnOJf_W2AwBc5UdRL2nywCg"

st.title("📊 Análisis Estadístico Avanzado")

with st.sidebar:
    st.header("⚙️ Navegación")
    modo = st.radio("Módulos:", ["1. Dataset", "2. Análisis de Distribución", "3. Inferencia Estadística", "4. Consultoría IA"])

# --- 1. DATASET ---
if modo == "1. Dataset":
    st.header("📂 Gestión de Datos")
    col1, col2 = st.columns([1, 2])
    with col1:
        n = st.number_input("Tamaño de muestra (n):", value=100, min_value=1)
        if st.button("✨ Generar Muestra Aleatoria"):
            # Fijamos el nombre de la columna para evitar el KeyError
            data = np.random.normal(50, 15, int(n))
            st.session_state.df = pd.DataFrame(data, columns=["Datos_Muestrales"])
            st.success("Muestra generada.")
    with col2:
        if st.session_state.df is not None:
            st.subheader("Vista Previa")
            st.dataframe(st.session_state.df, height=300, use_container_width=True)

# --- 2. ANÁLISIS DE DISTRIBUCIÓN ---
elif modo == "2. Análisis de Distribución":
    st.header("📈 Caracterización Estructural")
    if st.session_state.df is not None:
        col = "Datos_Muestrales"
        c1, c2 = st.columns(2)
        
        with c1:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.histplot(st.session_state.df[col], kde=True, color="#4e79a7", alpha=0.7, ax=ax)
            ax.set_title("Distribución de Frecuencias", fontweight='bold')
            st.pyplot(fig)
            
        with c2:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.boxplot(x=st.session_state.df[col], color="#76b7b2", notch=True, ax=ax)
            ax.set_title("Detección de Outliers", fontweight='bold')
            st.pyplot(fig)

        st.markdown("---")
        st.subheader("📝 Evaluación Técnica de Supuestos")
        ca, cb = st.columns(2)
        with ca:
            st.markdown("#### **I. Simetría y Normalidad**")
            norm = st.radio("Conclusión:", ["Pendiente", "Normalidad Detectada", "Asimetría Presente"], label_visibility="collapsed")
            if norm == "Normalidad Detectada":
                st.info("**Justificación:** La morfología mesocúrtica y simetría bilateral validan el uso de modelos paramétricos.")
            elif norm == "Asimetría Presente":
                st.warning("**Justificación:** El coeficiente de skewness indica una desviación significativa de la campana de Gauss.")

        with cb:
            st.markdown("#### **II. Análisis de Varianza**")
            out = st.radio("Observación:", ["Pendiente", "Varianza Estable", "Presencia de Atípicos"], label_visibility="collapsed")
            if out == "Varianza Estable":
                st.success("**Justificación:** Homocedasticidad confirmada. No se detectan valores fuera de los límites 1.5*IQR.")
            elif out == "Presencia de Atípicos":
                st.error("**Justificación:** Outliers detectados. Se recomienda precaución ya que pueden inflar el error estándar.")
    else:
        st.error("⚠️ Genera datos primero.")

# --- 3. INFERENCIA ESTADÍSTICA ---
elif modo == "3. Inferencia Estadística":
    st.header("🧪 Prueba de Hipótesis Z")
    if st.session_state.df is not None:
        col1, col2, col3 = st.columns(3)
        mu0 = col1.number_input("Media H0:", value=50.0)
        sigma = col2.number_input("Desviación σ:", value=15.0)
        alpha = col3.slider("Alfa (α):", 0.01, 0.10, 0.05)

        datos = st.session_state.df["Datos_Muestrales"]
        x_bar, n = datos.mean(), len(datos)
        z_stat = (x_bar - mu0) / (sigma / np.sqrt(n))
        p_val = stats.norm.sf(abs(z_stat)) * 2

        m1, m2, m3 = st.columns(3)
        m1.metric("Media (x̄)", f"{x_bar:.2f}")
        m2.metric("Estadístico Z", f"{z_stat:.4f}")
        m3.metric("P-Valor", f"{p_val:.4f}")

        # Gráfica Premium
        x = np.linspace(-4, 4, 1000)
        y = stats.norm.pdf(x, 0, 1)
        z_crit = stats.norm.ppf(1 - alpha/2)
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(x, y, color='#2c3e50', lw=2)
        ax.fill_between(x, y, where=(x >= z_crit) | (x <= -z_crit), color='#e15759', alpha=0.6, label='Zona de Rechazo')
        ax.fill_between(x, y, where=(x < z_crit) & (x > -z_crit), color='#59a14f', alpha=0.2, label='Zona de Aceptación')
        ax.axvline(z_stat, color='#4e79a7', ls='--', lw=4, label=f'Z-Calc: {z_stat:.2f}')
        ax.legend(); st.pyplot(fig)

        if p_val < alpha: st.error("⚠️ DECISIÓN: RECHAZAR H0")
        else: st.success("✅ DECISIÓN: NO RECHAZAR H0")
        
        st.session_state.datos_ia = f"Media={x_bar:.2f}, Z={z_stat:.2f}, P={p_val:.4f}, Alfa={alpha}"
    else:
        st.error("⚠️ Sin datos.")

# --- 4. CONSULTORÍA IA (FIX FINAL 403) ---
elif modo == "4. Consultoría IA":
    st.header("🤖 Consultoría con Gemini")
    if 'datos_ia' in st.session_state:
        if st.button("🚀 Generar Informe"):
            try:
                # Nuevo método de configuración para evitar el 403
                genai.configure(api_key=st.session_state.api_key)
                # Forzamos la versión flash que es la más estable para API Keys gratuitas
                model = genai.GenerativeModel('gemini-1.5-flash')
                
                # Prompt estructurado
                prompt = f"Analiza estos resultados estadísticos: {st.session_state.datos_ia}. Dame una conclusión técnica."
                
                with st.spinner("Procesando..."):
                    response = model.generate_content(prompt)
                    st.info(response.text)
            except Exception as e:
                st.error(f"Error 403 / Permisos: Google requiere validación manual. Intenta crear una nueva clave en Google AI Studio si persiste.")
    else:
        st.warning("⚠️ Completa el Paso 3 primero.")