import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os
import seaborn as sns
import statsmodels.api as sm

from sklearn.decomposition import PCA
from sklearn.cluster import (
    SpectralClustering, 
    AgglomerativeClustering, 
    KMeans, 
    DBSCAN
)
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import silhouette_score
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

def limpar_dados_longitudinais_visit(df, visit=1):
    df_clean = df.copy()

    # Selecionar apenas a primeira visita
    if 'Visit' in df_clean.columns:
        df_clean = df_clean[df_clean['Visit'] == visit]

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

    # Clusterização com controle de aleatoriedade
    if algoritmo == 'Spectral':
        cluster = SpectralClustering(n_clusters=n_clusters, assign_labels='kmeans', random_state=42)
    elif algoritmo == 'Agglomerative':
        cluster = AgglomerativeClustering(n_clusters=n_clusters)
    elif algoritmo == 'KMeans':
        cluster = KMeans(n_clusters=n_clusters, random_state=42)
    elif algoritmo == 'DBSCAN':
        cluster = DBSCAN(eps=0.5, min_samples=5)  # determinístico, sem seed
    elif algoritmo == 'BayesianGaussian':
        cluster = BayesianGaussianMixture(n_components=n_clusters, random_state=42)
    else:
        raise ValueError(f"Algoritmo de clusterização '{algoritmo}' não reconhecido.")

    # Aplicação do algoritmo
    labels = cluster.fit_predict(X_scaled)

    # PCA para visualização
    pca = PCA(n_components=2, random_state=42)
    componentes = pca.fit_transform(X_scaled)

    df_pca = pd.DataFrame(componentes, columns=['PC1', 'PC2'])
    df_pca['Cluster'] = labels

    # Adiciona a coluna Group se for o caso
    if tem_classificacao and 'Group' in df.columns:
        df_pca['Group'] = df['Group'].values

    variancia_explicada = pca.explained_variance_ratio_

    return df_pca, variancia_explicada

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

def limpar_dados_longitudinais_todas_visitas(df):
    """Limpa os dados longitudinais mantendo todas as visitas"""
    df_clean = df.copy()

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

def analise_clusters_por_visita(df, variaveis, algoritmo, n_clusters):
    """Realiza análise de clusterização para cada visita separadamente"""
    resultados = []
    
    # Obter lista de visitas únicas
    visitas = sorted(df['Visit'].unique())
    
    for visita in visitas:
        # Filtrar dados para a visita atual
        df_visita = df[df['Visit'] == visita].copy()
        
        # Rodar clusterização
        df_pca, _ = rodar_clusterizacao(df_visita, variaveis, algoritmo, n_clusters, True)
        
        # Calcular distribuição percentual
        if 'Group' in df_pca.columns and 'Cluster' in df_pca.columns:
            cluster_dist = pd.crosstab(df_pca['Cluster'], df_pca['Group'], normalize='index') * 100
            cluster_dist['Visita'] = visita
            resultados.append(cluster_dist)
    
    # Combinar todos os resultados
    if resultados:
        df_resultado = pd.concat(resultados)
        df_resultado = df_resultado.reset_index().melt(id_vars=['Visita', 'Cluster'], 
                                                      var_name='Group', 
                                                      value_name='Percentual')
        return df_resultado
    return None

def plotar_distribuicao_clusters1(df_resultado):
    """Plota a distribuição percentual dos grupos em cada cluster por visita"""
    if df_resultado is not None:
        # Preparar os dados
        df_plot = df_resultado.copy()
        
        # Mapear nomes completos para siglas
        group_map = {
            'Nondemented': 'ND',
            'Demented': 'D',
            'Converted': 'C'
        }
        df_plot['Group'] = df_plot['Group'].map(group_map)
        
        # Criar uma coluna combinando Cluster e Group
        df_plot['Cluster_Group'] = df_plot['Cluster'].astype(str) + ' - ' + df_plot['Group']
        
        # Ordenar as visitas
        df_plot = df_plot.sort_values('Visita')
        
        # Criar o gráfico
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Definir paleta de cores
        palette = {
            '0 - ND': '#1f77b4', '0 - D': '#ff7f0e', '0 - C': '#2ca02c',
            '1 - ND': '#aec7e8', '1 - D': '#ffbb78', '1 - C': '#98df8a',
            '2 - ND': '#c5b0d5', '2 - D': '#c49c94', '2 - C': '#dbdb8d'
        }
        
        # Plotar barras agrupadas
        sns.barplot(data=df_plot, x='Cluster_Group', y='Percentual', 
                    hue='Visita', palette='viridis', ax=ax)
        
        # Melhorar a visualização
        ax.set_title('Distribuição Percentual dos Grupos nos Clusters por Visita', pad=20)
        ax.set_xlabel('Cluster - Grupo')
        ax.set_ylabel('Percentual (%)')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        
        # Adicionar linhas de grid e ajustar limites
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.set_ylim(0, 100)
        
        # Melhorar a legenda
        ax.legend(title='Visita', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Adicionar valores nas barras
        for p in ax.patches:
            height = p.get_height()
            if height > 0:
                ax.text(p.get_x() + p.get_width()/2., height + 1,
                        f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.warning("Não foi possível gerar o gráfico de distribuição.")

def plotar_distribuicao_clusters2(df_resultado):
    """Plota a distribuição percentual dos grupos em cada cluster por visita usando linhas"""
    if df_resultado is not None:
        # Preparar os dados
        df_plot = df_resultado.copy()
        
        # Mapear nomes completos para siglas
        group_map = {
            'Nondemented': 'ND',
            'Demented': 'D',
            'Converted': 'C'
        }
        df_plot['Group'] = df_plot['Group'].map(group_map)
        
        # Criar uma coluna combinando Cluster e Group
        df_plot['Cluster_Group'] = 'Cluster ' + df_plot['Cluster'].astype(str) + ' - ' + df_plot['Group']
        
        # Ordenar os dados
        df_plot = df_plot.sort_values(['Cluster', 'Group', 'Visita'])
        
        # Criar o gráfico
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Definir paleta de cores para clusters
        cluster_colors = {
            0: '#1f77b4',  # Azul
            1: '#ff7f0e',  # Laranja
            2: '#2ca02c',  # Verde
            3: '#d62728',  # Vermelho
            4: '#9467bd'   # Roxo
        }
        
        # Estilos de linha para grupos
        line_styles = {
            'ND': '-',  # Linha sólida para ND
            'D': '--',  # Linha tracejada para D
            'C': ':'    # Linha pontilhada para C
        }
        
        # Plotar linhas para cada combinação Cluster-Group
        for cluster in df_plot['Cluster'].unique():
            for group in df_plot['Group'].unique():
                subset = df_plot[(df_plot['Cluster'] == cluster) & (df_plot['Group'] == group)]
                label = f'Cluster {cluster} - {group}'
                sns.lineplot(data=subset, x='Visita', y='Percentual', 
                            color=cluster_colors[cluster],
                            linestyle=line_styles[group],
                            marker='o', markersize=8,
                            label=label, ax=ax)
        
        # Melhorar a visualização
        ax.set_title('Evolução da Distribuição dos Grupos nos Clusters por Visita', pad=20)
        ax.set_xlabel('Número da Visita')
        ax.set_ylabel('Percentual no Cluster (%)')
        ax.set_xticks(sorted(df_plot['Visita'].unique()))
        
        # Adicionar linhas de grid e ajustar limites
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.set_ylim(0, 100)
        
        # Melhorar a legenda
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Cluster - Grupo')
        
        # Adicionar valores nos pontos
        for _, row in df_plot.iterrows():
            ax.text(row['Visita'], row['Percentual'] + 2, 
                    f"{row['Percentual']:.1f}%", 
                    ha='center', va='bottom', fontsize=8,
                    color=cluster_colors[row['Cluster']])
        
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.warning("Não foi possível gerar o gráfico de distribuição.")

def plotar_evolucao_convertido(df_resultado):
    """Plota a evolução apenas do grupo Converted nos clusters por visita"""
    if df_resultado is not None:
        # Filtrar apenas o grupo Converted
        df_convertido = df_resultado[df_resultado['Group'] == 'Converted'].copy()
        
        if not df_convertido.empty:
            # Ordenar os dados
            df_convertido = df_convertido.sort_values(['Cluster', 'Visita'])
            
            # Criar o gráfico
            fig, ax = plt.subplots(figsize=(6, 4))
            
            # Paleta de cores para clusters
            palette = sns.color_palette("husl", n_colors=len(df_convertido['Cluster'].unique()))
            
            # Plotar linhas para cada cluster
            sns.lineplot(data=df_convertido, x='Visita', y='Percentual', 
                        hue='Cluster', palette=palette,
                        style='Cluster', markers=True, dashes=False,
                        linewidth=2.5, markersize=10, ax=ax)
            
            # Melhorar a visualização
            ax.set_title('Evolução do Grupo "Converted" nos Clusters por Visita', pad=20)
            ax.set_xlabel('Número da Visita')
            ax.set_ylabel('Percentual no Cluster (%)')
            ax.set_xticks(sorted(df_convertido['Visita'].unique()))
            
            # Adicionar linhas de grid e ajustar limites
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            ax.set_ylim(0, df_convertido['Percentual'].max() + 10)
            
            # Melhorar a legenda
            ax.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Adicionar valores nos pontos
            for _, row in df_convertido.iterrows():
                ax.text(row['Visita'], row['Percentual'] + 1, 
                        f"{row['Percentual']:.1f}%", 
                        ha='center', va='bottom', fontsize=9,
                        color=palette[row['Cluster']])
            
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.warning("Não há dados disponíveis para o grupo 'Converted'.")
    else:
        st.warning("Não foi possível gerar o gráfico de evolução.")
def plot_distribuicao_cross(df_pca_cross, cross_clean):
    """Plota a distribuição de nWBV, CDR e MMSE para cada cluster dos dados transversais"""
    # Juntar os dados de clusterização com os dados originais
    df_merged = pd.concat([df_pca_cross['Cluster'], cross_clean[['nWBV', 'CDR', 'MMSE']]], axis=1)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Plot para nWBV
    sns.boxplot(data=df_merged, x='Cluster', y='nWBV', ax=axes[0])
    axes[0].set_title('Distribuição de nWBV por Cluster')
    
    # Plot para CDR
    sns.boxplot(data=df_merged, x='Cluster', y='CDR', ax=axes[1])
    axes[1].set_title('Distribuição de CDR por Cluster')
    
    # Plot para MMSE
    sns.boxplot(data=df_merged, x='Cluster', y='MMSE', ax=axes[2])
    axes[2].set_title('Distribuição de MMSE por Cluster')
    
    plt.tight_layout()
    st.pyplot(fig)

@st.cache_data

def treinar_ou_carregar_modelo(df, caminho_modelo='modelo_nwbv.pkl'):
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Selecionar colunas e eliminar NaNs
    df = df[['subject_id', 'visit', 'age', 'mmse', 'cdr', 'nwbv']].dropna()
    df['cdr'] = pd.to_numeric(df['cdr'], errors='coerce')

    # Filtrar indivíduos com pelo menos 3 visitas
    valid_subjects = df['subject_id'].value_counts()[lambda x: x >= 3].index
    df = df[df['subject_id'].isin(valid_subjects)]

    # Ordenar por tempo
    df = df.sort_values(by=['subject_id', 'age'])

    # Calcular diferenças
    df['delta_nwbv'] = df.groupby('subject_id')['nwbv'].diff()
    df['delta_tempo'] = df.groupby('subject_id')['age'].diff()

    # Remover tempo zero ou negativo
    df = df[df['delta_tempo'] > 0]

    # Calcular variação anual
    df['delta_nwbv_ano'] = df['delta_nwbv'] / df['delta_tempo']
    df['nwbv_futuro'] = df['nwbv'] + df['delta_nwbv_ano']

    # Limpeza final
    df = df.replace([np.inf, -np.inf], np.nan)
    df_model = df[['mmse', 'cdr', 'age', 'nwbv', 'nwbv_futuro']].dropna()

    # Se o modelo já existir, carrega
    if os.path.exists(caminho_modelo):
        modelo = joblib.load(caminho_modelo)
    else:
        # Treina e salva
        X = df_model[['mmse', 'cdr', 'age', 'nwbv']]
        y = df_model['nwbv_futuro']
        modelo = LinearRegression().fit(X, y)
        joblib.dump(modelo, caminho_modelo)

    return modelo

def menu_simulacao(df):
    st.header("🔮 Simulação de Volume Cerebral Futuro (nWBV)")
    
    # Carrega ou treina modelo
    modelo = treinar_ou_carregar_modelo(df)

    # Interface de entrada
    mmse = st.slider("MMSE", 0, 30, 26)
    cdr = st.selectbox("CDR", [0.0, 0.5, 1.0, 2.0])
    age = st.slider("Idade atual", 60, 100, 75)
    nwbv_atual = st.slider("nWBV atual", 0.60, 0.85, 0.72)

    anos_futuros = st.slider("🔁 Quantos anos no futuro você quer simular?", 1, 10, 1)

    if st.button("🔍 Simular volume cerebral futuro"):
        # Estimar variação anual com o modelo
        X_input = pd.DataFrame([[mmse, cdr, age, nwbv_atual]],
                               columns=['mmse', 'cdr', 'age', 'nwbv'])
        nwbv_1ano = modelo.predict(X_input)[0]

        # Delta estimado para 1 ano
        delta_1ano = nwbv_1ano - nwbv_atual

        # Extrapolar para N anos
        nwbv_Nanos = nwbv_atual + delta_1ano * anos_futuros

        st.markdown(f"### 📈 Volume cerebral previsto em **{anos_futuros} ano(s)**: **{nwbv_Nanos:.4f}**")
        st.caption("Previsão baseada em extrapolação linear da regressão treinada.")

def menu_inspecao(cross_clean, long_clean, long_raw):
    st.subheader("Inspeção de Clusters nos Dados")

    # Limpar dados longitudinais mantendo todas as visitas
    long_all_visits = limpar_dados_longitudinais_todas_visitas(long_raw)

    col_config, col_cross, col_long, col_evol = st.columns([1, 2, 2, 2])

    with col_config:
        st.markdown("Configuração")

        variaveis_disponiveis = [col for col in cross_clean.columns if cross_clean[col].dtype != 'object' and col not in ['MRI ID', 'Subject ID', 'ID']]
        variaveis_default = [v for v in ['Age', 'MMSE', 'CDR', 'nWBV'] if v in variaveis_disponiveis]

        variaveis_selecionadas = st.multiselect("Variáveis para análise", variaveis_disponiveis, default=variaveis_default)

        algoritmo = st.selectbox("Algoritmo de Clusterização", ["BayesianGaussian", "Spectral", "Agglomerative", "KMeans", "DBSCAN"])

        n_clusters = st.number_input("Número de Clusters", min_value=2, max_value=10, value=3, step=1)

    if len(variaveis_selecionadas) >= 2:
        # Análise para dados cross-section
        df_pca_cross, var_exp_cross = rodar_clusterizacao(cross_clean, variaveis_selecionadas, algoritmo, n_clusters, False)
        
        # Análise para dados longitudinais (1a visita)
        df_pca_long, var_exp_long = rodar_clusterizacao(long_clean, variaveis_selecionadas, algoritmo, n_clusters, True)
        
        # Análise para todas as visitas
        df_evolucao = analise_clusters_por_visita(long_all_visits, variaveis_selecionadas, algoritmo, n_clusters)

        # Unificar limites para visualização consistente
        df_pca_total = pd.concat([df_pca_cross[['PC1', 'PC2']], df_pca_long[['PC1', 'PC2']]])
        xlim = (df_pca_total['PC1'].min(), df_pca_total['PC1'].max())
        ylim = (df_pca_total['PC2'].min(), df_pca_total['PC2'].max())

        with col_cross:
            st.markdown("Análise Transversal - Sem Classificação")
            fig1, ax1 = plt.subplots(figsize=(6, 4), constrained_layout=True)
            sns.scatterplot(data=df_pca_cross, x="PC1", y="PC2", hue="Cluster", palette="tab10", ax=ax1)
            ax1.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
            ax1.set_xlim(xlim)
            ax1.set_ylim(ylim)
            ax1.set_title(f'PCA - Não Classificado: Var. Explicada PC1: {var_exp_cross[0]:.2%}, PC2: {var_exp_cross[1]:.2%}')
            st.pyplot(fig1)
            
            # Adicionar plot de distribuição abaixo do gráfico PCA
            st.markdown("Distribuição de variáveis por Cluster")
            plot_distribuicao_cross(df_pca_cross, cross_clean)

        with col_long:
            st.markdown("Primeira Visita Logintudinal - Com Classificação")
            
            # Definir estilo e marcadores para cada grupo
            markers = {"Demented": "X", "Nondemented": "s", "Converted": "^"}
            
            # Criar a paleta husl com o número de clusters
            cluster_palette = sns.color_palette("husl", n_colors=n_clusters)
            
            fig2, ax2 = plt.subplots(figsize=(6, 4), constrained_layout=True)
            
            sns.scatterplot(data=df_pca_long, x="PC1", y="PC2", 
                        hue="Cluster", style="Group",
                        markers=markers,
                        palette=cluster_palette,  # Usando a paleta husl
                        s=80,
                        ax=ax2)
            
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax2.set_title(f'PCA - Var. Explicada: PC1: {var_exp_long[0]:.2%}, PC2: {var_exp_long[1]:.2%}')
            ax2.set_xlim(xlim)
            ax2.set_ylim(ylim)
            st.pyplot(fig2)
            st.markdown("Distribuição Grupo/Cluster")
            plotar_heatmap_clusterizacao_unico(df_pca_long, cluster_col='Cluster', group_col='Group')

        with col_evol:
            st.markdown("Análise Longitudinal - Grupo Convertido")
            plotar_evolucao_convertido(df_evolucao)
            try:
                features = df_pca_long[['PC1', 'PC2']].values
                labels = df_pca_long['Cluster'].values
                sil_score = silhouette_score(features, labels)
                st.markdown(f"**Silhouette Score:** {sil_score:.3f}")
            except Exception as e:
                st.warning(f"Não foi possível calcular o Silhouette Score: {e}")
            st.markdown("**Contribuição das Variáveis nas Componentes Principais (PCA)**")

            try:
                X = long_clean[variaveis_selecionadas].dropna()
                X_scaled = StandardScaler().fit_transform(X)
                pca = PCA(n_components=2)
                pca.fit(X_scaled)
                componentes_df = pd.DataFrame(
                    pca.components_.T,
                    columns=['PC1', 'PC2'],
                    index=variaveis_selecionadas
                )
                st.dataframe(componentes_df.style.format("{:.2f}"))
            except Exception as e:
                st.warning(f"Erro ao calcular contribuições das variáveis no PCA: {e}")

    else:
        st.info("Selecione pelo menos duas variáveis para continuar.")

def menu_inicio(df):
    col_texto, col_grafico = st.columns([1.5, 2])

    with col_texto:
        st.markdown("""
        🔍 **1. Avaliação da Clusterização Clínica**  
        Utilizamos técnicas de clusterização para investigar se é possível **identificar grupos clínicos distintos** a partir das variáveis cognitivas e de imagem disponíveis.  

        📈 **2. Relação entre Volume Cerebral e Envelhecimento**  
        O volume cerebral total (nWBV) tende a diminuir com o envelhecimento. No entanto, essa perda pode ocorrer de forma mais acentuada em indivíduos com declínio cognitivo, por isso será explorada a relação ao longo do tempo, para **simular a trajetória do volume cerebral em função da idade**.
        """)

    with col_grafico:
        # Variável de interesse
        VAR = 'nWBV'  # Changed to match original column name

        # Cores por grupo
        colors = {
    'Nondemented': '#2196F3',  # Azul
    'Demented': '#FF9800'     # Laranja
}
        
        # Use original column names without converting to lowercase
        df = df.dropna(subset=[VAR, 'Age', 'Group', 'MR Delay', 'Subject ID'])
        
        # Remove the column name conversion since we're using original names
        # df.columns = df.columns.str.lower().str.replace(' ', '').str.replace('_', '')
        
        # Agrupar Converted com Demented
        df['group_type'] = df['Group'].replace({
            'Nondemented': 'Nondemented',
            'Demented': 'Demented',
            'Converted': 'Demented'
        })

        # Calcular idade real
        df['first_visit_age'] = df.groupby('Subject ID')['Age'].transform('first')
        df['years_since_baseline'] = df['MR Delay'] / 365.25
        df['real_age'] = df['first_visit_age'] + df['years_since_baseline']

        # Criar gráfico
        fig, ax = plt.subplots(figsize=(6, 4))
        existing_groups = [g for g in colors if g in df['group_type'].unique()]

        for group in existing_groups:
            group_data = df[df['group_type'] == group]
            sns.scatterplot(
                x='real_age', y=VAR,
                data=group_data,
                color=colors[group], ax=ax,
                alpha=0.6, s=80,
                label=f"{group} (n={len(group_data)})"
            )

            # Ajuste de regressão linear + intervalo de confiança
            try:
                X = sm.add_constant(group_data['real_age'])
                y = group_data[VAR]
                model = sm.OLS(y, X).fit()

                x_pred = np.linspace(group_data['real_age'].min(), group_data['real_age'].max(), 100)
                y_pred = model.predict(sm.add_constant(x_pred))
                conf_int = model.get_prediction(sm.add_constant(x_pred)).conf_int()

                ax.plot(x_pred, y_pred, color=colors[group], linewidth=2.5)
                ax.fill_between(x_pred, conf_int[:, 0], conf_int[:, 1], color=colors[group], alpha=0.15)
            except Exception as e:
                st.warning(f"Erro na regressão para {group}: {str(e)}")

        ax.set_title('Evolução do Volume Cerebral pela Idade')
        ax.set_ylabel('nWBV')
        ax.set_xlabel('Idade (anos)')
        ax.legend(title='Grupo Clínico')
        ax.grid(False)
        sns.despine()
        st.pyplot(fig)


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
        st.write(f"**Após a Limpeza:** {long_clean.shape[0]} registros - primeira visita")
        st.markdown("**Pipeline:**")
        st.markdown("""
        - Seleção dos sujeitos por visita  
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
    st.title("Um Estudo sobre Demência")
    #carregar dados
    df_cross, df_long = load_data()
    # Cópias para evitar alterações nos dados originais
     # Na parte onde carrega os dados, adicione:
    df_cross, df_long = load_data()
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
        menu_inicio(long_original.copy())
    elif aba == "Inspeção":
        menu_inspecao(cross_clean, long_clean, long_original)
    elif aba == "Simulação":
        menu_simulacao(df_long)

    elif aba == "Dados":
        exibir_menu_dados(cross_original, cross_clean, long_original, long_clean, long_clean_sim)

if __name__ == "__main__":
    main()
