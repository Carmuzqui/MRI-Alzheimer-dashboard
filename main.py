import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Configuración de la página
st.set_page_config(page_title="Dashboard de Alzheimer", layout="wide")

# Tema oscuro con textos en blanco
st.markdown("""
    <style>
        [data-testid="stAppViewContainer"] {
            background-color: #1E1E1E;
        }
        [data-testid="stSidebar"] {
            background-color: #2D2D2D;
        }
        .stSelectbox, .stMultiSelect {
            background-color: #3D3D3D;
        }
        [data-testid="stHeader"] {
            background-color: rgba(0,0,0,0);
        }
        .stMarkdown, .stText, .stMetric, .stSelectbox, .stMultiSelect, .stRadio, .stCheckbox {
            color: #FFFFFF !important;
        }
        .stPlotlyChart {
            color: #FFFFFF;
        }
        .stSelectbox > div > div > div {
            color: #FFFFFF !important;
        }
        h1, h2, h3 {
            color: #FFFFFF !important;
        }
        [data-testid="stMetricLabel"], [data-testid="stMetricValue"] {
            color: #FFFFFF !important;
        }
    </style>
""", unsafe_allow_html=True)

# Cargar los datos
@st.cache_data
def cargar_datos():
    url_cross = "https://raw.githubusercontent.com/jddunn/dementia-progression-analysis/master/oasis_cross-sectional.csv"
    url_long = "https://raw.githubusercontent.com/multivacplatform/multivac-dl/master/data/mri-and-alzheimers/oasis_longitudinal.csv"
    df_cross = pd.read_csv(url_cross)
    df_long = pd.read_csv(url_long)
    return df_cross, df_long

df_cross, df_long = cargar_datos()

# Sidebar para navegación y configuración
st.sidebar.title("Navegación")
dataset = st.sidebar.radio("Seleccione el Dataset:", ("Transversal", "Longitudinal"))
df = df_cross if dataset == "Transversal" else df_long

menu = st.sidebar.selectbox("Seleccione la visualización:",
                            ["Resumen", "Distribución de Edad", "MMSE vs Edad", "Volumen Cerebral", "PCA", "Análisis Longitudinal"])

# Funciones para gráficos
def plot_age_distribution(df):
    fig = px.histogram(df, x="Age", color="M/F", marginal="box", title="Distribución de Edad por Género")
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='#FFFFFF'
    )
    return fig

def plot_mmse_vs_age(df):
    fig = px.scatter(df, x="Age", y="MMSE", color="CDR", title="MMSE vs Edad (por CDR)")
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='#FFFFFF'
    )
    return fig

def plot_brain_volume(df):
    fig = px.scatter(df, x="Age", y="nWBV", color="M/F", title="Volumen Cerebral vs Edad")
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='#FFFFFF'
    )
    return fig

def plot_pca(df, features):
    X = df[features].dropna()
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame(data=pca_result, columns=["PC1", "PC2"])
    if "CDR" in df.columns:
        pca_df["CDR"] = df["CDR"].iloc[X.index]
        color_var = "CDR"
    else:
        color_var = None
    fig = px.scatter(pca_df, x="PC1", y="PC2", color=color_var, title="PCA: Componentes Principales")
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='#FFFFFF'
    )
    return fig

def plot_longitudinal(df, patients):
    fig = px.line(df[df['Subject ID'].isin(patients)], x='Visit', y='MMSE',
                  color='Subject ID', title='Progresión del MMSE a lo largo del tiempo')
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='#FFFFFF'
    )
    return fig

# Contenido principal
st.title("Dashboard de Análisis de Alzheimer")

# Métricas principales
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Pacientes Únicos", df['ID'].nunique() if 'ID' in df.columns else df['Subject ID'].nunique())
with col2:
    st.metric("Edad Promedio", f"{df['Age'].mean():.1f} años")
with col3:
    st.metric("% con Demencia", f"{(df['CDR'] > 0).mean():.1%}" if 'CDR' in df.columns else "N/A")
with col4:
    st.metric("MMSE Promedio", f"{df['MMSE'].mean():.1f}" if 'MMSE' in df.columns else "N/A")

# Visualización principal basada en la selección del menú
if menu == "Resumen":
    st.subheader("Resumen Estadístico")
    st.write(df.describe())

elif menu == "Distribución de Edad":
    st.plotly_chart(plot_age_distribution(df), use_container_width=True)

elif menu == "MMSE vs Edad":
    st.plotly_chart(plot_mmse_vs_age(df), use_container_width=True)

elif menu == "Volumen Cerebral":
    st.plotly_chart(plot_brain_volume(df), use_container_width=True)

elif menu == "PCA":
    st.subheader("Análisis de Componentes Principales (PCA)")
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    pca_features = st.multiselect("Seleccione las Variables para el PCA:", numeric_columns,
                                  default=[col for col in ["Age", "MMSE", "eTIV", "nWBV"] if col in numeric_columns])
    if len(pca_features) >= 2:
        st.plotly_chart(plot_pca(df, pca_features), use_container_width=True)
    else:
        st.warning("Seleccione al menos 2 variables para realizar el PCA.")

elif menu == "Análisis Longitudinal" and dataset == "Longitudinal":
    st.subheader("Análisis Longitudinal")
    patients = st.multiselect("Seleccione pacientes para análisis:", df['Subject ID'].unique())
    if patients:
        st.plotly_chart(plot_longitudinal(df, patients), use_container_width=True)
    else:
        st.write("Seleccione al menos un paciente para ver la progresión longitudinal.")

# Nota de pie
st.markdown("---")
st.markdown("Dashboard construido con Streamlit y Plotly. Datos: OASIS Dataset.")
