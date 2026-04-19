import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import google.generativeai as genai

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Analytics Pro: Estadística e IA", layout="wide")
plt.style.use('ggplot') # Estilo visual limpio

# Inicializar estados de sesión
if 'df' not in st.session_state:
    st.session_state.df = None

# --- SIDEBAR MEJORADO ---
with st.sidebar:
    st.title("⚙️ Panel de Control")
    modo = st.radio("Módulos del Sistema:", 
                   ["1. Gestión de Datos", "2. Análisis de Distribución", "3. Inferencia Estadística", "4. Consultoría IA"])
    st.markdown("---")
    st.caption("Proyecto Estadística U3 - 2026")

st.title("📊 Análisis Estadístico Inteligente")

# --- MODULO 1: GESTIÓN DE DATOS (MEJORADO) ---
if modo == "1. Gestión de Datos":
    st.header("📂 Adquisición de Muestra")
    
    tab1, tab2 = st.tabs(["✨ Generar Datos", "📤 Cargar Archivo"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            n = st.number_input("Tamaño de muestra (n):", 10, 1000, 100)
            media_gen = st.number_input("Media deseada:", value=50.0)
        with col2:
            std_gen = st.number_input("Desviación estándar:", value=15.0)
            if st.button("🚀 Generar Muestra Aleatoria"):
                data = np.random.normal(media_gen, std_gen, int(n))
                st.session_state.df = pd.DataFrame(data, columns=["Variable"])
                st.success("Muestra generada con éxito.")

    with tab2:
        file = st.file_uploader("Cargar CSV o Excel", type=["csv", "xlsx"])
        if file:
            df_temp = pd.read_csv(file) if file.name.endswith('csv') else pd.read_excel(file)
            st.session_state.df = df_temp.select_dtypes(include=[np.number])
            st.success("Archivo cargado. Se han seleccionado solo columnas numéricas.")

    if st.session_state.df is not None:
        st.subheader("📋 Vista de Datos")
        st.dataframe(st.session_state.df, use_container_width=True, height=250)
        st.write(f"**Resumen rápido:** {len(st.session_state.df)} registros detectados.")

# --- MODULO 2: VISUALIZACIÓN (MEJORADO) ---
elif modo == "2. Análisis de Distribución":
    st.header("📈 Caracterización Estructural")
    
    if st.session_state.df is not None:
        col_select = st.selectbox("Selecciona la variable a analizar:", st.session_state.df.columns)
        datos_v = st.session_state.df[col_select]

        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.histplot(datos_v, kde=True, color="#4e79a7", alpha=0.6, ax=ax)
            ax.set_title("Histograma y Densidad (KDE)", fontsize=12)
            st.pyplot(fig)
        with c2:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.boxplot(x=datos_v, color="#76b7b2", notch=True, ax=ax)
            ax.set_title("Diagrama de Caja (Outliers)", fontsize=12)
            st.pyplot(fig)
        
        # --- EXPLICACIONES TÉCNICAS ---
        st.markdown("---")
        exp1, exp2 = st.columns(2)
        with exp1:
            st.subheader("🧐 Interpretación de Forma")
            st.write("""
            El **Histograma** nos permite evaluar la **Normalidad**. 
            - Si la curva KDE (la línea azul) es simétrica y parece una campana, los datos son aptos para pruebas paramétricas.
            - La **Curvatura** nos indica si hay sesgo hacia la izquierda o derecha.
            """)
        with exp2:
            st.subheader("🧐 Interpretación de Dispersión")
            st.write("""
            El **Boxplot** visualiza los cuartiles. 
            - Los puntos fuera de los 'bigotes' son **Outliers** (valores atípicos).
            - Una caja muy ancha indica alta variabilidad en tus datos.
            """)
    else:
        st.warning("⚠️ No hay datos cargados. Ve al Módulo 1.")

# --- MODULO 3: PRUEBA Z (MEJORADO) ---
elif modo == "3. Inferencia Estadística":
    st.header("🧪 Contraste de Hipótesis")
    
    if st.session_state.df is not None:
        col_select = st.selectbox("Variable para la prueba:", st.session_state.df.columns)
        datos = st.session_state.df[col_select]
        
        with st.expander("⚙️ Configuración de la Prueba Z", expanded=True):
            c1, c2, c3 = st.columns(3)
            h0 = c1.number_input("Media Teórica (H₀):", value=50.0)
            sigma = c2.number_input("Sigma (σ) Poblacional:", value=15.0)
            alpha = c3.select_slider("Nivel de Significancia (α):", options=[0.01, 0.05, 0.10], value=0.05)

        # Cálculos
        x_bar = datos.mean()
        n_size = len(datos)
        z_calc = (x_bar - h0) / (sigma / np.sqrt(n_size))
        p_val = stats.norm.sf(abs(z_calc)) * 2

        # Resultados Visuales
        st.session_state.res_ia = {
            "x_bar": x_bar, "h0": h0, "n": n_size, 
            "sigma": sigma, "alpha": alpha, "z": z_calc, "p": p_val, "var_name": col_select
        }

        m1, m2, m3 = st.columns(3)
        m1.metric("Media Muestral (x̄)", f"{x_bar:.2f}")
        m2.metric("Estadístico Z", f"{z_calc:.4f}")
        m3.metric("P-Valor", f"{p_val:.4f}", delta=f"{alpha-p_val:.3f}", delta_color="inverse")

        # Explicación del Resultado
        st.markdown("### 📢 Conclusión")
        if p_val < alpha:
            st.error(f"**SE RECHAZA H₀**: El P-valor ({p_val:.4f}) es menor que α ({alpha}). Existe evidencia estadística para afirmar que la media es diferente de {h0}.")
        else:
            st.success(f"**NO SE RECHAZA H₀**: El P-valor ({p_val:.4f}) es mayor que α ({alpha}). No hay evidencia suficiente para descartar la media teórica.")
            
        st.info("""
        **¿Qué significa esto?** La Prueba Z mide cuántas desviaciones estándar se aleja tu media del valor teórico. 
        Si el P-valor es muy pequeño, la probabilidad de que tu resultado sea por 'suerte' es mínima.
        """)
    else:
        st.warning("⚠️ Genera datos en el Módulo 1.")

# --- MODULO 4: IA (MANTENIDO) ---
elif modo == "4. Consultoría IA":
    st.header("🤖 Consultoría con Gemini")
    
    api_key = st.text_input("Ingresa tu API Key de Gemini:", type="password")
    
    if 'res_ia' in st.session_state:
        if st.button("🚀 Iniciar Análisis de IA"):
            if not api_key:
                st.warning("Ingresa la API Key.")
            else:
                try:
                    genai.configure(api_key=api_key)
                    # Selección automática de modelo
                    valid_model = 'gemini-1.5-flash'
                    for m in genai.list_models():
                        if 'generateContent' in m.supported_generation_methods and 'flash' in m.name:
                            valid_model = m.name.replace('models/', '')
                            break
                            
                    model = genai.GenerativeModel(valid_model)
                    r = st.session_state.res_ia
                    prompt = f"""
                    Actúa como experto en estadística. Analiza esta prueba Z:
                    - Variable: {r['var_name']}
                    - Media muestral: {r['x_bar']:.4f}
                    - Media hipotética (H0): {r['h0']}
                    - Tamaño de muestra (n): {r['n']}
                    - Desviación estándar (sigma): {r['sigma']}
                    - Nivel de significancia (alpha): {r['alpha']}
                    - Estadístico Z calculado: {r['z']:.4f}
                    - P-valor: {r['p']:.4f}

                    ¿Se rechaza H0? Explica la decisión técnica y la interpretación estadística en español.
                    """
                    with st.spinner("Gemini está procesando el reporte..."):
                        response = model.generate_content(prompt)
                        st.markdown("---")
                        st.subheader("📋 Dictamen Profesional")
                        st.write(response.text)
                except Exception as e:
                    st.error(f"Error: {e}")
    else:
        st.warning("⚠️ Realiza la prueba en el Módulo 3.")