import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import streamlit as st

def load_data():
    cross = pd.read_csv("data/oasis_cross-sectional.csv")
    long = pd.read_csv("data/oasis_longitudinal.csv")
    return cross, long

def clean_longitudinal_data(df_long):
    log = []
    original_shape = df_long.shape
    log.append(f"Nº de registros originais: {original_shape[0]}")

    long_clean = df_long.copy()
    long_clean = long_clean[long_clean['Visit'] == 1]
    log.append(f"Filtrado apenas Visit==1: {long_clean.shape[0]} registros")

    long_clean['Gender'] = (long_clean['M/F'] == 'M').astype(int)

    # Remoção de colunas desnecessárias
    drop_cols = ['Hand', 'ASF', 'eTIV']
    long_clean.drop(columns=drop_cols, errors='ignore', inplace=True)
    log.append(f"Remoção de colunas: {drop_cols}")

    # Preenchimento de valores ausentes
    cat_cols = ['M/F', 'Group']
    for col in cat_cols:
        if col in long_clean.columns:
            mode_val = long_clean[col].mode()[0]
            long_clean[col] = long_clean[col].fillna(mode_val)
            log.append(f"Preenchido valores ausentes em {col} com moda: {mode_val}")

    num_cols = ['Age', 'EDUC', 'SES', 'MMSE', 'CDR', 'nWBV', 'MR Delay']
    for col in num_cols:
        if col in long_clean.columns:
            med = long_clean[col].median()
            long_clean[col] = long_clean[col].fillna(med)
            log.append(f"Preenchido valores ausentes em {col} com mediana: {med:.2f}")

    # Verificação final
    n_missing = long_clean.isnull().sum().sum()
    log.append(f"Total de valores faltantes após limpeza: {n_missing}")

    return long_clean, log

def limpar_cross(df):
    df_clean = df.copy()

    # Remover colunas irrelevantes
    cols_to_drop = ['Hand', 'ASF', 'eTIV']
    df_clean.drop(columns=cols_to_drop, errors='ignore', inplace=True)

    # Verificar valores faltantes
    if 'nWBV' in df_clean.columns:
        df_clean = df_clean[df_clean['nWBV'].notna()]

    # Preencher dados faltantes
    cat_cols = ['M/F']
    for col in cat_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])

    num_cols = ['Age', 'Educ', 'SES', 'MMSE', 'CDR', 'nWBV', 'Delay']
    for col in num_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    return df_clean


# Função para limpar os dados longitudinais (com classificação)
def limpar_long(df):
    df_clean = df.copy()

    # Selecionar apenas a primeira visita
    if 'Visit' in df_clean.columns:
        df_clean = df_clean[df_clean['Visit'] == 1]

    # Adicionar coluna "Gender" como 1 para masculino, 0 para feminino
    df_clean['Gender'] = (df_clean['M/F'] == 'M').astype(int)

    # Remover colunas irrelevantes
    cols_to_drop = ['Hand', 'ASF', 'eTIV']
    df_clean.drop(columns=cols_to_drop, errors='ignore', inplace=True)

    # Preencher valores faltantes
    cat_cols = ['M/F', 'Group']
    for col in cat_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])

    num_cols = ['Age', 'EDUC', 'SES', 'MMSE', 'CDR', 'nWBV', 'MR Delay']
    for col in num_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    return df_clean

def clean_cross_data(df_cross):
    log = []
    original_shape = df_cross.shape
    log.append(f"Nº de registros originais: {original_shape[0]}")

    cross_clean = df_cross.copy()

    # Remoção de colunas desnecessárias
    drop_cols = ['Hand', 'ASF', 'eTIV']
    cross_clean.drop(columns=drop_cols, errors='ignore', inplace=True)
    log.append(f"Remoção de colunas: {drop_cols}")

    # Preenchimento de valores ausentes
    cat_cols = ['M/F']
    for col in cat_cols:
        if col in cross_clean.columns:
            mode_val = cross_clean[col].mode()[0]
            cross_clean[col] = cross_clean[col].fillna(mode_val)
            log.append(f"Preenchido valores ausentes em {col} com moda: {mode_val}")

    num_cols = ['Age', 'Educ', 'SES', 'MMSE', 'CDR', 'Delay']
    for col in num_cols:
        if col in cross_clean.columns:
            med = cross_clean[col].median()
            cross_clean[col] = cross_clean[col].fillna(med)
            log.append(f"Preenchido valores ausentes em {col} com mediana: {med:.2f}")

    # Remover registros com nWBV ausente
    missing_before = cross_clean['nWBV'].isnull().sum()
    cross_clean = cross_clean.dropna(subset=['nWBV'])
    log.append(f"Remoção de {missing_before} registros com nWBV ausente")

    n_missing = cross_clean.isnull().sum().sum()
    log.append(f"Total de valores faltantes após limpeza: {n_missing}")

    return cross_clean, log

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

def exibir_menu_dados(cross_original, cross_clean, long_original, long_clean):
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


def inspecao(cross_clean, long_clean):
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

        variaveis_disponiveis = [col for col in df_usado.columns if df_usado[col].dtype != 'object' and col not in ['MRI ID', 'Subject ID', 'ID']]
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

import numpy as np
import pandas as pd
from scipy.stats import shapiro
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def avaliar_normalidade(df, colunas):
    """Retorna um dicionário com a normalidade (True/False) baseada no teste de Shapiro-Wilk"""
    normalidade = {}
    for col in colunas:
        stat, p = shapiro(df[col])
        normalidade[col] = p > 0.05  # True se distribuição for normal
    return normalidade

def gerar_dados_sinteticos(df_real, colunas, n_amostras=200):
    """Gera dados sintéticos baseados nas estatísticas do df_real"""
    normalidade = avaliar_normalidade(df_real, colunas)
    dados_grupo1 = {}
    dados_grupo2 = {}

    for col in colunas:
        if normalidade[col]:
            media = df_real[col].mean()
            desvio = df_real[col].std()
            dados_grupo1[col] = np.random.normal(loc=media, scale=desvio, size=n_amostras)
            dados_grupo2[col] = np.random.normal(loc=media + 0.5 * desvio, scale=desvio, size=n_amostras)
        else:
            dados_grupo1[col] = np.random.choice(df_real[col].dropna(), size=n_amostras, replace=True)
            dados_grupo2[col] = np.random.choice(df_real[col].dropna(), size=n_amostras, replace=True)

    grupo1 = pd.DataFrame(dados_grupo1)
    grupo2 = pd.DataFrame(dados_grupo2)

    grupo1['Grupo Real'] = 0
    grupo2['Grupo Real'] = 1

    df_sim = pd.concat([grupo1, grupo2], ignore_index=True)
    return df_sim

def simular_clusterizacao(df_sim, colunas, n_clusters=2):
    """Executa PCA + clusterização nos dados sintéticos"""
    X = df_sim[colunas].copy()
    X_scaled = StandardScaler().fit_transform(X)
    
    pca = PCA(n_components=2)
    componentes = pca.fit_transform(X_scaled)

    modelo = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = modelo.fit_predict(componentes)

    df_plot = pd.DataFrame(componentes, columns=['PC1', 'PC2'])
    df_plot['Cluster'] = clusters
    df_plot['Grupo Real'] = df_sim['Grupo Real'].values

    return df_plot

def plotar_simulacao(df_plot):
    """Plota os dados com os clusters e os grupos reais"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for grupo in df_plot['Grupo Real'].unique():
        sub_df = df_plot[df_plot['Grupo Real'] == grupo]
        ax.scatter(sub_df['PC1'], sub_df['PC2'], 
                   label=f'Grupo Real {grupo}', 
                   alpha=0.5, s=60, edgecolor='k')

    for cluster in df_plot['Cluster'].unique():
        centroide = df_plot[df_plot['Cluster'] == cluster][['PC1', 'PC2']].mean()
        ax.text(centroide['PC1'], centroide['PC2'], f'C{cluster}', fontsize=12, weight='bold')

    ax.set_title("Simulação: Clusters vs Grupos Reais")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()



# def menu_simulacao(df_long_clean):
#     st.header("🔬 Menu de Simulação")

#     st.markdown("""
#     Esta aba permite simular cenários onde dois grupos sintéticos são criados com base nas estatísticas reais do dataset. 
#     Em seguida, é feita uma clusterização para observar se os grupos reais são recuperáveis pelos algoritmos de clusterização.
#     """)

#     # Seleção das variáveis
#     colunas_disponiveis = ['Age', 'EDUC', 'SES', 'MMSE']
#     colunas_selecionadas = st.multiselect("Variáveis para simulação", colunas_disponiveis, default=colunas_disponiveis)

#     # Tamanho da amostra por grupo
#     n_amostras = st.slider("Número de amostras por grupo", min_value=50, max_value=500, step=10, value=200)

#     # Número de clusters
#     n_clusters = st.slider("Número de clusters", min_value=2, max_value=6, value=2)

#     # Botão de execução
#     if st.button("Executar Simulação"):
#         if len(colunas_selecionadas) < 2:
#             st.warning("Selecione pelo menos duas variáveis para executar a simulação.")
#             return

#         # Geração e execução
#         with st.spinner("Gerando dados sintéticos e executando a clusterização..."):
#             df_sim = gerar_dados_sinteticos(df_long_clean, colunas_selecionadas, n_amostras)
#             df_plot = simular_clusterizacao(df_sim, colunas_selecionadas, n_clusters)

#             # Gráfico
#             st.subheader("Visualização dos Componentes Principais com Clusters e Grupos Reais")
#             fig, ax = plt.subplots(figsize=(8, 6))

#             for grupo in df_plot['Grupo Real'].unique():
#                 sub_df = df_plot[df_plot['Grupo Real'] == grupo]
#                 ax.scatter(sub_df['PC1'], sub_df['PC2'], label=f'Grupo Real {grupo}', alpha=0.5, s=60, edgecolor='k')

#             for cluster in df_plot['Cluster'].unique():
#                 centroide = df_plot[df_plot['Cluster'] == cluster][['PC1', 'PC2']].mean()
#                 ax.text(centroide['PC1'], centroide['PC2'], f'C{cluster}', fontsize=12, weight='bold')

#             ax.set_title("Simulação: Clusters vs Grupos Reais")
#             ax.legend()
#             ax.grid(True)
#             st.pyplot(fig)

#             # Mostrar os dados sintéticos se o usuário quiser
#             with st.expander("🔍 Ver dados sintéticos utilizados na simulação"):
#                 st.dataframe(df_sim)

#             # Métrica simples de acurácia
#             from sklearn.metrics import adjusted_rand_score
#             ari = adjusted_rand_score(df_sim['Grupo Real'], df_plot['Cluster'])
#             st.success(f"Adjusted Rand Index (ARI): {ari:.2f}")

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np

def treinar_modelo_demencia(df_long):
    # Copiar o dataframe para evitar alterações indesejadas
    df = df_long.copy()

    # Filtrar dados válidos: manter apenas linhas com 'Group' definido
    df = df[df['Group'].notna()]

    # Transformar a variável 'Group' em binária
    df['Dementia'] = df['Group'].apply(lambda x: 1 if x == 'Demented' else 0)

    # Variáveis preditoras (você pode ajustar essa lista conforme achar melhor)
    features = ['Age', 'MMSE', 'SES', 'EDUC', 'nWBV']

    # Remover linhas com dados ausentes nas colunas selecionadas
    df = df.dropna(subset=features + ['Dementia'])

    # Separar X e y
    X = df[features]
    y = df['Dementia']

    # Padronização + modelo em pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('logreg', LogisticRegression(solver='liblinear'))
    ])

    # Treinamento do modelo
    pipeline.fit(X, y)

    # (Opcional) Avaliação rápida
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    print("Relatório de classificação (avaliação rápida):")
    print(classification_report(y_test, y_pred))

    return pipeline, features
import pandas as pd
import numpy as np

def simular_incidentes_demencia(modelo, features, idade_medias=[70, 75, 80], n=500, std_idade=5, df_real=None):
    resultados = []

    # Obter estatísticas reais das demais variáveis, se fornecido
    if df_real is not None:
        df_ref = df_real.dropna(subset=features)
        medias = df_ref[features].mean()
        stds = df_ref[features].std()
    else:
        # Se não fornecido, usar valores padrão
        medias = pd.Series({feat: 0 for feat in features})
        stds = pd.Series({feat: 1 for feat in features})

    # Para cada idade média, gerar população sintética e prever demência
    for idade_media in idade_medias:
        # Gerar dados sintéticos
        dados = pd.DataFrame({
            'Age': np.random.normal(loc=idade_media, scale=std_idade, size=n),
            'MMSE': np.random.normal(loc=medias['MMSE'], scale=stds['MMSE'], size=n),
            'SES': np.random.normal(loc=medias['SES'], scale=stds['SES'], size=n),
            'EDUC': np.random.normal(loc=medias['EDUC'], scale=stds['EDUC'], size=n),
            'nWBV': np.random.normal(loc=medias['nWBV'], scale=stds['nWBV'], size=n)
        })

        # Corrigir limites (ex: MMSE não pode ser negativo)
        dados['MMSE'] = dados['MMSE'].clip(lower=0, upper=30)
        dados['SES'] = dados['SES'].clip(lower=1, upper=5)
        dados['EDUC'] = dados['EDUC'].clip(lower=0)
        dados['nWBV'] = dados['nWBV'].clip(lower=0, upper=1)

        # Prever demência
        pred = modelo.predict(dados[features])
        incidencia = np.mean(pred)

        # Armazenar resultado
        resultados.append({
            'Idade Média': idade_media,
            'População Simulada': n,
            'Prevalência Estimada (%)': round(100 * incidencia, 2)
        })

    return pd.DataFrame(resultados)
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def menu_simulacao(df_long):
    st.header("📈 Simulação: Impacto do Envelhecimento na Incidência de Demência")

    with st.expander("ℹ️ Sobre esta simulação"):
        st.markdown("""
        Esta simulação estima a prevalência de demência em populações fictícias com diferentes idades médias,
        utilizando um modelo treinado a partir do conjunto de dados longitudinal.
        """)

    # Treinar modelo
    modelo, features = treinar_modelo_demencia(df_long)

    # Parâmetros de simulação
    st.subheader("🔧 Parâmetros da Simulação")

    idade_min = int(df_long["Age"].min())
    idade_max = int(df_long["Age"].max())
    idade_medias = st.multiselect(
        "Escolha as idades médias da população simulada",
        options=list(range(idade_min, idade_max + 1, 5)),
        default=[70, 75, 80]
    )

    n_pessoas = st.slider("Tamanho da população simulada", 100, 1000, 500, step=100)
    std_idade = st.slider("Desvio padrão da idade", 1, 10, 5)

    # Rodar simulação
    if st.button("Rodar simulação"):
        st.success("Simulando populações e estimando prevalência de demência...")

        resultados = simular_incidentes_demencia(
            modelo, features,
            idade_medias=idade_medias,
            n=n_pessoas,
            std_idade=std_idade,
            df_real=df_long
        )

        st.subheader("📊 Resultados da Simulação")
        st.dataframe(resultados)

        # Gráfico
        fig, ax = plt.subplots()
        sns.barplot(data=resultados, x="Idade Média", y="Prevalência Estimada (%)", ax=ax, palette="Blues_d")
        ax.set_title("Estimativa de Prevalência de Demência por Idade Média")
        ax.set_ylim(0, 100)
        st.pyplot(fig)


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
    cross_clean = limpar_cross(cross_original)
    long_clean = limpar_long(long_original)
    
    # Segunda linha: Menu interativo de abas
    aba = option_menu(
        menu_title=None,
        options=["Inspeção", "Simulação", "Dados"],
        icons=["search", "cpu", "database"],
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
    if aba == "Inspeção":
        st.subheader("Análise Exploratória e Visualização de Dados")
        # Conteúdo será adicionado futuramente
        inspecao(cross_clean, long_clean)

    elif aba == "Simulação":
        st.subheader("Simulação de Modelos ou Processos")
        # Conteúdo será adicionado futuramente
        menu_simulacao(df_long)

    elif aba == "Dados":
        st.subheader("Dados Brutos e Processados")
        # Conteúdo será adicionado futuramente
        exibir_menu_dados(cross_original, cross_clean, long_original, long_clean)

if __name__ == "__main__":
    main()
