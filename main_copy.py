import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import streamlit as st

def load_data():
    cross = pd.read_csv("data/oasis_cross-sectional.csv")
    long = pd.read_csv("data/oasis_longitudinal.csv")
    return cross, long
#Limpeza...
def limpar_dados_cross_sectional(df):
    df = df.drop(columns=['Hand', 'ASF', 'eTIV', 'Delay'], errors='ignore')
    # Adicionar coluna "Gender" como 1 para masculino, 0 para feminino
    df['Gender'] = (df['M/F'] == 'M').astype(int)
    df1 = df.dropna(subset = ['Educ','SES',	'MMSE','CDR'])
    return df1

def limpar_dados_longitudinais(df):
    df_clean = df.copy()

    # Selecionar apenas a primeira visita
    if 'Visit' in df_clean.columns:
        df_clean = df_clean[df_clean['Visit'] == 1]

    # Adicionar coluna "Gender" como 1 para masculino, 0 para feminino
    df_clean['Gender'] = (df_clean['M/F'] == 'M').astype(int)

    # Remover colunas irrelevantes
    cols_to_drop = ['Hand', 'ASF', 'eTIV']
    df_clean.drop(columns=cols_to_drop, errors='ignore', inplace=True)
    df_clean = df_clean.rename(columns={"EDUC":"Educ"})
    # Preencher valores faltantes
    cat_cols = ['M/F', 'Group']
    for col in cat_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])

    num_cols = ['Age', 'Educ', 'SES', 'MMSE', 'CDR', 'nWBV', 'MR Delay']
    for col in num_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    return df_clean

def limpar_dados_longitudinais_simulacao(df):
    # Cópia do dataframe para evitar warnings
    df = df.copy()
    
    # 1. Converter gênero para binário (mantendo a coluna original para referência)
    df['Gender'] = df['M/F'].map({'M': 1, 'F': 0})
    
    # 2. Remover colunas não relevantes para a análise
    cols_to_drop = ['Hand', 'ASF', 'MRI ID', 'MR Delay']
    df = df.drop(columns=cols_to_drop, errors='ignore')
    
    # 3. Tratamento de valores faltantes:
    # - Para variáveis categóricas: preencher com moda
    # - Para variáveis numéricas: preencher com mediana por grupo
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].fillna(df[col].mode()[0])
    
    for col in df.select_dtypes(include=['int64', 'float64']).columns:
        df[col] = df.groupby('Group')[col].transform(lambda x: x.fillna(x.median()))
    
    # 4. Garantir que Subject ID seja string
    df['Subject ID'] = df['Subject ID'].astype(str)
    
    # 5. Ordenar por Subject ID e Visit para análise longitudinal
    df = df.sort_values(['Subject ID', 'Visit'])
    
    return df

from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering, AgglomerativeClustering
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def rodar_clusterizacao(df, variaveis, algoritmo, n_clusters, tem_classificacao):
    df = df.copy()

    # Inclui a coluna 'Group' se ela existir e for necessária
    colunas_usadas = variaveis + (['Group'] if tem_classificacao and 'Group' in df.columns else [])
    
    # Remove entradas com NaN
    df = df[colunas_usadas].dropna()

    X = df[variaveis]

    # Padroniza os dados
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Clusterização
    if algoritmo == 'Spectral':
        cluster = SpectralClustering(n_clusters=n_clusters, assign_labels='kmeans', random_state=42)
    else:
        cluster = AgglomerativeClustering(n_clusters=n_clusters)

    labels = cluster.fit_predict(X_scaled)

    # PCA
    pca = PCA(n_components=2)
    componentes = pca.fit_transform(X_scaled)

    df_pca = pd.DataFrame(componentes, columns=['PC1', 'PC2'])
    df_pca['Cluster'] = labels

    # Adiciona a coluna Group se for o caso
    if tem_classificacao and 'Group' in df.columns:
        df_pca['Group'] = df['Group'].values

    variancia_explicada = pca.explained_variance_ratio_

    return df_pca, variancia_explicada
def exibir_menu_dados(cross_original, cross_clean, long_original, long_clean, long_clean_sim):
    st.subheader("Relatório de Limpeza de Dados")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### Dados Transversais (Sem Classificação)")
        st.write(f"**Antes da Limpeza:** {cross_original.shape[0]} registros")
        st.write(f"**Após a Limpeza:** {cross_clean.shape[0]} registros")
        st.markdown("**Pipeline:**")
        st.markdown("""
        - Criação da variável binária Gender 
        - Remoção de colunas: Hand, ASF, eTIV, M/F
        - Remoção de registros com qualquer valor faltante
        """)

    with col2:
        st.markdown("### Dados Longitudinais (Com Classificação)")
        st.write(f"**Antes da Limpeza:** {long_original.shape[0]} registros")
        st.write(f"**Após a Limpeza:** {long_clean.shape[0]} registros")
        st.markdown("**Pipeline:**")
        st.markdown("""
        - Seleção apenas da 1ª visita  
        - Criação da variável binária Gender  
        - Remoção de colunas: Hand, ASF, eTIV, M/F  
        - Preenchimento de dados categóricos com moda  
        - Preenchimento de dados numéricos com mediana
        """)

    with col3:
        st.markdown("### Dados Longitudinais (Simulação)")
        st.write(f"**Antes da Limpeza:** {long_original.shape[0]} registros")
        st.write(f"**Após a Limpeza:** {long_clean_sim.shape[0]} registros")
        st.markdown("**Pipeline:**")
        st.markdown("""
        - Criação da variável binária Gender   
        - Remoção de colunas: Hand, ASF, eTIV, M/F  
        - Preenchimento de dados categóricos com moda  
        - Preenchimento de dados numéricos com mediana   
        - Mantido todas as visitas para análise longitudinal  
        """)

def exibir_menu_dados1(cross_original, cross_clean, long_original, long_clean):
    st.subheader("Relatório de Limpeza de Dados")
    
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Dados Cross-Sectional (Sem Classificação)")
        st.write(f"**Antes da Limpeza:** {cross_original.shape[0]} registros")
        st.write(f"**Após a Limpeza:** {cross_clean.shape[0]} registros")
        st.markdown("**Pipeline:**")
        st.markdown("""
        - Remoção de colunas: `Hand`, `ASF`, `eTIV`  
        - Remoção de registros com `nWBV` faltante  
        - Preenchimento de dados categóricos com moda  
        - Preenchimento de dados numéricos com mediana
        """)

    with col2:
        st.markdown("### Dados Longitudinais (Com Classificação)")
        st.write(f"**Antes da Limpeza:** {long_original.shape[0]} registros")
        st.write(f"**Após a Limpeza:** {long_clean.shape[0]} registros")
        st.markdown("**Pipeline:**")
        st.markdown("""
        - Seleção apenas da 1ª visita  
        - Criação da variável binária `Gender`  
        - Remoção de colunas: `Hand`, `ASF`, `eTIV`  
        - Preenchimento de dados categóricos com moda  
        - Preenchimento de dados numéricos com mediana
        """)
def plotar_heatmap_clusterizacao_unico(df, cluster_col, group_col='Group'):
    """
    Plota um único heatmap com a distribuição percentual de cada grupo em cada cluster.

    Parâmetros:
    - df: DataFrame com colunas de clusterização e grupo.
    - cluster_col: nome da coluna com a clusterização.
    - group_col: nome da coluna de grupo (padrão: 'Group').
    """
    cluster_dist = pd.crosstab(df[cluster_col], df[group_col], normalize='columns') * 100

    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    sns.heatmap(cluster_dist, annot=True, fmt='.1f', cmap='Blues', vmin=0, vmax=100, ax=ax)
    ax.set_title('Distribuição percentual grupo/cluster')
    ax.set_xlabel('Grupo')
    ax.set_ylabel('Cluster')
    st.pyplot(fig)

def inspecao(cross_clean, long_clean):
    st.subheader("Inspeção de Clusters nos Dados")

    col_config, col_cross, col_long = st.columns([1, 3, 3])

    with col_config:
        st.markdown("### Configuração")

        variaveis_disponiveis = [col for col in cross_clean.columns if cross_clean[col].dtype != 'object' and col not in ['MRI ID', 'Subject ID', 'ID']]
        variaveis_default = [v for v in ['Age', 'MMSE', 'CDR', 'nWBV'] if v in variaveis_disponiveis]

        variaveis_selecionadas = st.multiselect("Variáveis para análise", variaveis_disponiveis, default=variaveis_default)

        algoritmo = st.selectbox("Algoritmo de Clusterização", ["Spectral", "Agglomerative"])

        n_clusters = st.number_input("Número de Clusters", min_value=2, max_value=10, value=3, step=1)

    if len(variaveis_selecionadas) >= 2:
        df_pca_cross, var_exp_cross = rodar_clusterizacao(cross_clean, variaveis_selecionadas, algoritmo, n_clusters, False)
        df_pca_long, var_exp_long = rodar_clusterizacao(long_clean, variaveis_selecionadas, algoritmo, n_clusters, True)
        # Unifique para encontrar limites globais
        df_pca_total = pd.concat([df_pca_cross[['PC1', 'PC2']], df_pca_long[['PC1', 'PC2']]])

        xlim = (df_pca_total['PC1'].min(), df_pca_total['PC1'].max())
        ylim = (df_pca_total['PC2'].min(), df_pca_total['PC2'].max())

        with col_cross:
            # fig1, ax1 = plt.subplots(figsize=(6, 4))
            fig1, ax1 = plt.subplots(figsize=(6, 4), constrained_layout=True)

            sns.scatterplot(data=df_pca_cross, x="PC1", y="PC2", hue="Cluster", palette="tab10", ax=ax1)
            ax1.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
            ax1.set_xlim(xlim)
            ax1.set_ylim (ylim)
            ax1.set_title(f'PCA - Não Classificado: Var. Explicada PC1: {var_exp_cross[0]:.2%}, PC2: {var_exp_cross[1]:.2%}')
            st.pyplot(fig1)

            with st.expander("Estatísticas Cross-Sectional"):
                stats_cross = df_pca_cross.groupby("Cluster")[["PC1", "PC2"]].describe()
                st.dataframe(stats_cross)
        
        with col_long:
            # Mapeia nomes longos para siglas
            df_pca_long["Grupo"] = df_pca_long["Group"].map({
                "Nondemented": "N",
                "Demented": "D",
                "Converted": "C"
                })
            df_pca_long["GrupoP"] = df_pca_long["Group"].map({
                "Nondemented": "Não Demente",
                "Demented": "Demente",
                "Converted": "Convertido"
                })
            # fig2, ax2 = plt.subplots(figsize=(6, 4))
            fig2, ax2 = plt.subplots(figsize=(6, 4), constrained_layout=True)
            sns.scatterplot(data=df_pca_long, x="PC1", y="PC2", hue="Cluster", style="Grupo", palette="tab10", ax=ax2)
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax2.set_title(f'PCA - Classificado: Var. Explicada PC1: {var_exp_long[0]:.2%}, PC2: {var_exp_long[1]:.2%}')
            ax2.set_xlim(xlim)
            ax2.set_ylim (ylim)
            st.pyplot(fig2)

            with st.expander("Distribuição por Grupo (Heatmap)"):
                col_name = "Cluster"
                if col_name in df_pca_long.columns:
                    plotar_heatmap_clusterizacao_unico(df_pca_long, cluster_col=col_name, group_col='GrupoP')
                else:
                    st.warning(f"Coluna de cluster '{col_name}' não encontrada nos dados.")

    else:
        with col_cross:
            st.info("Selecione pelo menos duas variáveis para continuar.")
        with col_long:
            st.info("Selecione pelo menos duas variáveis para continuar.")

def inspecao1(cross_clean, long_clean):
    st.subheader("Inspeção de Clusters nos Dados")

    col1, col2, col3 = st.columns([1, 3, 2])

    with col1:
        st.markdown("### Configuração")

        tipo_dado = st.radio("Base de dados", options=["Classificados", "Sem classificação"], index=0)

        if tipo_dado == "Classificados":
            df_usado = long_clean
            tem_classificacao = True
        else:
            df_usado = cross_clean
            tem_classificacao = False

        # variaveis_disponiveis = [col for col in df_usado.columns if df_usado[col].dtype != 'object' and col not in ['MRI ID', 'Subject ID', 'ID', 'Delay', 'MR Delay']]
        variaveis_disponiveis = ['Age', 'MMSE', 'CDR', 'nWBV', 'Educ', 'SES']

        variaveis_default = [v for v in ['Age', 'MMSE', 'CDR', 'nWBV'] if v in variaveis_disponiveis]

        variaveis_selecionadas = st.multiselect("Variáveis para análise", variaveis_disponiveis, default=variaveis_default)

        algoritmo = st.selectbox("Algoritmo de Clusterização", ["Spectral", "Agglomerative"])

        n_clusters = st.number_input("Número de Clusters", min_value=2, max_value=10, value=3, step=1)

    with col2:
        if len(variaveis_selecionadas) >= 2:
            df_pca, var_exp = rodar_clusterizacao(df_usado, variaveis_selecionadas, algoritmo, n_clusters, tem_classificacao)

            fig, ax = plt.subplots(figsize=(6, 4))
            if tem_classificacao and "Group" in df_pca.columns:
                sns.scatterplot(
                    data=df_pca, x="PC1", y="PC2",
                    hue="Cluster", style="Group", palette="tab10", ax=ax
                )
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            else:
                sns.scatterplot(
                    data=df_pca, x="PC1", y="PC2",
                    hue="Cluster", palette="tab10", ax=ax
                )
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

            ax.set_title(f'PCA: Variância Explicada PC1: {var_exp[0]:.2%}, PC2: {var_exp[1]:.2%}')
            st.pyplot(fig)
        else:
            st.info("Selecione pelo menos duas variáveis para continuar.")

    with col3:
        with st.expander("Estatísticas para Cluster"):
            if 'Cluster' in df_pca:
                stats_cluster = df_pca.groupby("Cluster")[["PC1", "PC2"]].describe()
                st.dataframe(stats_cluster)

        if tem_classificacao and "Group" in df_pca.columns:
            with st.expander("Estatísticas para Grupo"):
                stats_group = df_pca.groupby("Group")[["PC1", "PC2"]].describe()
                st.dataframe(stats_group)


import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

@st.cache_data
def treinar_modelo_volume(df):
    df = df.dropna(subset=['Age', 'nWBV'])
    X = df[['Age']]
    y = df['nWBV']
    modelo = LinearRegression().fit(X, y)
    return modelo

def menu_simulacao(df):
    st.header("Simulação: Projeção do Volume Cerebral ao Longo do Tempo")

    st.markdown("""
    Esta simulação permite estimar o volume cerebral (nWBV) de um indivíduo ao longo dos próximos anos, com base em sua idade atual.
    O modelo é treinado com dados reais para prever a taxa de declínio associada ao envelhecimento.
    """)

    modelo = treinar_modelo_volume(df)

    idade_atual = st.number_input("Idade atual", min_value=60, max_value=100, value=70)
    anos_futuros = st.slider("Projeção para quantos anos?", min_value=1, max_value=10, value=5)

    # Gerar projeções
    idades_futuras = np.arange(idade_atual, idade_atual + anos_futuros + 1)
    volumes_previstos = modelo.predict(idades_futuras.reshape(-1, 1))

    resultados = pd.DataFrame({
        'Ano': np.arange(2025, 2025 + anos_futuros + 1),
        'Idade': idades_futuras,
        'Volume Cerebral Estimado (nWBV)': volumes_previstos
    })

    st.subheader("Projeção do Volume Cerebral")
    st.dataframe(resultados.style.format({'Volume Cerebral Estimado (nWBV)': '{:.4f}'}))
    st.line_chart(resultados.set_index('Ano')['Volume Cerebral Estimado (nWBV)'])

def menu_inicio():
    pass
def main():
    st.set_page_config(layout="wide")  # Layout mais largo para melhor visualização
    # CSS para reduzir o espaço superior do título
    st.markdown("""
        <style>
            .block-container {
                padding-top: 1rem !important;  /* Reduz o espaço superior */
            }
        </style>
    """, unsafe_allow_html=True)
    # TÍTULO no topo em todas as abas
    st.title("Dashboard de Alzheimer")
    #carregar dados
    df_cross, df_long = load_data()
    # Cópias para evitar alterações nos dados originais
    cross_original = df_cross.copy()
    long_original = df_long.copy()
    # Limpeza dos dados
    cross_clean = limpar_dados_cross_sectional(cross_original)
    long_clean = limpar_dados_longitudinais(long_original)
    long_clean_sim = limpar_dados_longitudinais_simulacao(long_original)
    # Segunda linha: Menu interativo de abas
    aba = option_menu(
        menu_title=None,
        options=["Início","Inspeção", "Simulação", "Dados"],
        icons=["cpu","search", "cpu", "database"],
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {"padding": "0!important", "background-color": "#fafafa"},
            "icon": {"color": "#005580", "font-size": "20px"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "center",
                "--hover-color": "#eee",
                "padding": "10px",
                "margin": "0px",
            },
            "nav-link-selected": {"background-color": "#cceeff"},
        }
    )

    # Conteúdo principal baseado na aba selecionada
    if aba == "Início":
        st.subheader("Introdução")
        # Conteúdo será adicionado futuramente
        menu_inicio()
    elif aba == "Inspeção":
        st.subheader("Análise Exploratória e Visualização de Dados")
        
        inspecao(cross_clean, long_clean)

    elif aba == "Simulação":
        st.subheader("Simulação de Modelos ou Processos")
        # Conteúdo será adicionado futuramente
        menu_simulacao(df_long)

    elif aba == "Dados":
        st.subheader("Dados Brutos e Processados")
        # Conteúdo será adicionado futuramente
        exibir_menu_dados(cross_original, cross_clean, long_original, long_clean, long_clean_sim)

if __name__ == "__main__":
    main()
