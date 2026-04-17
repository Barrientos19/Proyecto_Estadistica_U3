import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de la página
st.set_page_config(page_title="Proyecto de Probabilidad", layout="wide")

st.title("📊 Aplicación de Análisis Estadístico")
st.markdown("---")

# Estado de la sesión para mantener los datos
if 'df' not in st.session_state:
    st.session_state.df = None

# Menú lateral
st.sidebar.header("Configuración")
modo = st.sidebar.selectbox("Selecciona un módulo", ["Inicio", "Carga de Datos", "Visualización", "Prueba de Hipótesis"])

if modo == "Inicio":
    st.write("### Bienvenido al proyecto de la Unidad 03")
    st.info("Selecciona 'Carga de Datos' para comenzar.")

elif modo == "Carga de Datos":
    st.header("1. Carga de Datos")
    opcion = st.radio("Origen:", ["Subir CSV", "Generación Sintética"])
    
    if opcion == "Subir CSV":
        archivo = st.file_uploader("Carga tu CSV", type=['csv'])
        if archivo:
            st.session_state.df = pd.read_csv(archivo)
    else:
        n = st.number_input("n", min_value=30, value=100)
        if st.button("Generar Datos"):
            st.session_state.df = pd.DataFrame(np.random.normal(50, 10, n), columns=["Variable_Sintetica"])
            st.success("¡Datos listos!")

    if st.session_state.df is not None:
        st.write("Vista previa:", st.session_state.df.head())

elif modo == "Visualización":
    st.header("2. Visualización de Distribuciones")
    if st.session_state.df is not None:
        col = st.selectbox("Variable:", st.session_state.df.columns)
        
        # Crear los gráficos
        col1, col2 = st.columns(2)
        with col1:
            fig1, ax1 = plt.subplots()
            sns.histplot(st.session_state.df[col], kde=True, ax=ax1, color="skyblue")
            ax1.set_title("Histograma y KDE")
            st.pyplot(fig1)
        
        with col2:
            fig2, ax2 = plt.subplots()
            sns.boxplot(y=st.session_state.df[col], ax=ax2, color="lightgreen")
            ax2.set_title("Boxplot")
            st.pyplot(fig2)
            
        st.markdown("---")
        st.subheader("Análisis")
        st.write("¿Es normal? Si la curva azul parece una campana, ¡sí!")
    else:
        st.error("Carga datos primero.")