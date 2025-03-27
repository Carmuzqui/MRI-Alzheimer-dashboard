import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import linregress
from sklearn.metrics import r2_score

# Configurações iniciais
st.set_page_config(page_title="Dashboard Alzheimer", layout="wide", page_icon="🧠")
sns.set_palette("husl")
#plt.style.use('seaborn-v0_8')

def remove_background(ax):
    ax.set_facecolor('none')
    ax.figure.set_facecolor('none')
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    ax.tick_params(colors='white')
    if ax.get_legend() is not None:
        ax.get_legend().set_frame_on(False)  # Remove fundo da legenda
        for text in ax.get_legend().get_texts():
            text.set_color("white")  # Define cor do texto da legenda como branco
    if hasattr(ax, 'collections') and ax.collections:
        cbar = ax.collections[0].colorbar
        if cbar is not None:
            cbar.ax.yaxis.set_tick_params(color='white')  # Cor dos ticks da colorbar
            cbar.ax.yaxis.label.set_color('white')  # Cor do rótulo da colorbar
            for text in cbar.ax.get_yticklabels():
                text.set_color("white")  # Cor dos valores da colorbar

# Funções auxiliares ============================================================

@st.cache_data
def load_data():
    """Carrega e prepara os dados"""
    cross = pd.read_csv("oasis_cross-sectional.csv")
    long = pd.read_csv("oasis_longitudinal.csv")

    # Padronizar nomes das colunas
    cross.columns = cross.columns.str.strip().str.lower().str.replace('/', '_')
    long.columns = long.columns.str.strip().str.lower().str.replace('/', '_')

    return cross, long


def demographic_info(df):
    """Exibe informações demográficas básicas"""
    media_idade = df['age'].mean()
    desvio_padrao = df['age'].std()
    percent_mulheres = (df['m_f'].value_counts(normalize=True)['F'] * 100)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Média de idade", f"{media_idade:.1f} anos")
    with col2:
        st.metric("Desvio padrão idade", f"{desvio_padrao:.1f} anos")
    with col3:
        st.metric("Percentual de mulheres", f"{percent_mulheres:.1f}%")


def plot_age_distribution(df):
    """Cria gráfico de distribuição de idade em percentual"""
    fig, ax = plt.subplots(figsize=(8, 4))
    n, bins, patches = ax.hist(df['age'], bins=15, color='skyblue',
                               edgecolor='black',
                               weights=np.ones(len(df['age'])) / len(df['age']) * 100)

    ax.set_title('Distribuição de Idade dos Participantes (%)', fontsize=14)
    ax.set_xlabel('Idade (anos)', fontsize=12)
    ax.set_ylabel('Percentual de Participantes (%)', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 30)
    remove_background(ax)
    st.pyplot(fig)


def plot_scatter_age_nwbv(df, threshold):
    """Scatter plot Idade x nWBV com threshold"""
    fig, ax = plt.subplots(figsize=(8, 4))

    # Criar categoria baseada no threshold
    df['age_group'] = np.where(df['age'] >= threshold, f'≥ {threshold}', f'< {threshold}')

    # Plotar scatter
    sns.scatterplot(data=df, x='age', y='nwbv', hue='age_group',
                    alpha=0.7, ax=ax)

    # Ajuste de regressão para todos os dados
    slope_all, intercept_all, _, _, _ = linregress(df['age'], df['nwbv'])
    r2_all = r2_score(df['nwbv'], intercept_all + slope_all * df['age'])
    ax.plot(df['age'], intercept_all + slope_all * df['age'],
            color='black', linestyle='--', label=f'Todos (R²={r2_all:.2f})')

    # Ajuste de regressão apenas para idade >= threshold
    df_above = df[df['age'] >= threshold]
    if len(df_above) > 1:
        slope_above, intercept_above, _, _, _ = linregress(df_above['age'], df_above['nwbv'])
        r2_above = r2_score(df_above['nwbv'], intercept_above + slope_above * df_above['age'])
        ax.plot(df_above['age'], intercept_above + slope_above * df_above['age'],
                color='red', linestyle='--', label=f'≥ {threshold} (R²={r2_above:.2f})')

    ax.set_title(f'Idade vs Volume Cerebral (Threshold: {threshold})')
    ax.set_xlabel('Idade (anos)')
    ax.set_ylabel('Volume Cerebral Normalizado')
    ax.legend()

    # Adicionar informações de ajuste
    stats_text = f"""
    Estatísticas de Ajuste:
    - Todos os dados:
      Inclinação: {slope_all:.4f}
      Intercepto: {intercept_all:.4f}
      R²: {r2_all:.4f}
    """
    if len(df_above) > 1:
        stats_text += f"""
    - Idade ≥ {threshold}:
      Inclinação: {slope_above:.4f}
      Intercepto: {intercept_above:.4f}
      R²: {r2_above:.4f}
    """
    remove_background(ax)
    st.pyplot(fig)
    with st.expander("Estatísticas de Regressão"):
        st.code(stats_text)


def plot_boxplot_cdr_mmse(df, threshold):
    """Boxplot CDR x MMSE com threshold"""
    df['age_group'] = np.where(df['age'] >= threshold, f'≥ {threshold}', f'< {threshold}')

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(data=df, x='cdr', y='mmse', hue='age_group', ax=ax)
    ax.set_title(f'CDR vs MMSE (Threshold: {threshold})')
    ax.set_xlabel('Clinical Dementia Rating (CDR)')
    ax.set_ylabel('Mini-Mental State Examination (MMSE)')
    ax.legend(title='Faixa Etária',labelcolor= 'white')
    remove_background(ax)
    st.pyplot(fig)


def plot_violin_nwbv_cdr(df, threshold):
    """Violin plot nWBV x CDR com threshold"""
    df['age_group'] = np.where(df['age'] >= threshold, f'≥ {threshold}', f'< {threshold}')

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.violinplot(data=df, x='cdr', y='nwbv', hue='age_group',
                   split=True, inner="quart", ax=ax)
    ax.set_title(f'Volume Cerebral vs CDR (Threshold: {threshold})')
    ax.set_xlabel('Clinical Dementia Rating (CDR)')
    ax.set_ylabel('Volume Cerebral Normalizado')
    ax.legend(title='Faixa Etária')
    remove_background(ax)
    st.pyplot(fig)


def plot_scatter_mmse_age(df, threshold):
    """Scatter plot MMSE x Idade com threshold"""
    df['age_group'] = np.where(df['age'] >= threshold, f'≥ {threshold}', f'< {threshold}')

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.scatterplot(data=df, x='age', y='mmse', hue='age_group',
                    alpha=0.7, ax=ax)
    ax.set_title(f'MMSE vs Idade (Threshold: {threshold})')
    ax.set_xlabel('Idade (anos)')
    ax.set_ylabel('Mini-Mental State Examination (MMSE)')
    remove_background(ax)
    st.pyplot(fig)


def plot_violin_age_cdr(df, threshold):
    """Violin plot Idade x CDR com threshold"""
    df['age_group'] = np.where(df['age'] >= threshold, f'≥ {threshold}', f'< {threshold}')

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.violinplot(data=df, x='cdr', y='age', hue='age_group',
                   split=True, inner="quart", ax=ax)
    ax.set_title(f'Idade vs CDR (Threshold: {threshold})')
    ax.set_xlabel('Clinical Dementia Rating (CDR)')
    ax.set_ylabel('Idade (anos)')
    ax.legend(title='Faixa Etária')
    remove_background(ax)
    st.pyplot(fig)


# Interface principal ==========================================================

def motivation_section(df):
    st.header("Análise Transversal de Dados de Alzheimer")
    st.write("""
    Este dashboard explora dados de ressonância magnética (MRI) e marcadores 
    clínicos de pacientes com Alzheimer em diferentes estágios e indivíduos 
    saudáveis. A análise transversal permite comparar grupos em um único 
    momento no tempo.
    """)

    demographic_info(df)

    # Gráfico de distribuição de idade (original - sem threshold)
    plot_age_distribution(df)

    # Slider para threshold (agora colocado APÓS o gráfico de distribuição)
    min_age, max_age = int(df['age'].min()), int(df['age'].max())
    threshold = st.slider("Selecione o threshold de idade para as análises abaixo:",
                          min_value=min_age,
                          max_value=max_age,
                          value=55,
                          key="motivation_threshold")

    st.subheader("Marcadores (Análise por Threshold)")
    col1, col2 = st.columns(2)

    with col1:
        plot_scatter_age_nwbv(df, threshold)
        plot_boxplot_cdr_mmse(df, threshold)
        plot_violin_nwbv_cdr(df, threshold)

    with col2:
        plot_scatter_mmse_age(df, threshold)
        plot_violin_age_cdr(df, threshold)


def longitudinal_section(df):
    st.header("Análise Longitudinal de Dados de Alzheimer")
    st.write("""
    Esta seção examina a progressão dos marcadores clínicos e de imagem ao 
    longo do tempo em pacientes com Alzheimer e controles saudáveis, incluindo
    indivíduos que converteram de não dementes para dementes durante o estudo.
    """)

    # Informações demográficas
    demographic_info(df)

    # Create columns for age distribution and correlation matrix
    col1, col2 = st.columns(2)

    with col1:
        # Gráfico de distribuição de idade
        plot_age_distribution(df)

    with col2:
        # Matriz de correlação (excluindo ASF) com novo gradiente
        st.subheader("Matriz de Correlação")
        numeric_cols = df.select_dtypes(include=np.number).columns
        numeric_cols = numeric_cols.drop(['asf', 'visit', 'mr delay'], errors='ignore')  # Remove ASF
        corr = df[numeric_cols].corr()

        # Definir o gradiente de cores personalizado
        from matplotlib.colors import LinearSegmentedColormap
        custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', ["#b3933c","#3BA3EC"])

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap=custom_cmap, center=0, ax=ax)
        remove_background(ax)
        st.pyplot(fig)

    # Identificar indivíduos convertidos
    converted_subjects = []
    for subject in df['subject id'].unique():
        subject_data = df[df['subject id'] == subject]
        groups = subject_data['group'].unique()
        if 'Nondemented' in groups and 'Demented' in groups:
            converted_subjects.append(subject)

    # Classificar os grupos com a nova paleta de cores
    df['Group_Type'] = df['group']
    df.loc[df['subject id'].isin(converted_subjects), 'Group_Type'] = 'Converted'

    # Definir a paleta de cores personalizada para os grupos
    group_colors = {
        'Nondemented': '#b3933c',  # Cor 2
        'Converted': '#3BA3EC',  # Cor 3
        'Demented': '#d88893'  # Cor 1
    }

    # Evolução do Quadro
    st.subheader("Evolução do Quadro Clínico")

    # Configurações dos gráficos
    metrics = ['nwbv', 'cdr', 'mmse']
    titles = ['Volume Cerebral Normalizado (nWBV)',
              'Taxa de Demência Clínica (CDR)',
              'Mini Exame do Estado Mental (MMSE)']
    ylabels = ['nWBV', 'CDR', 'MMSE']

    # Criar colunas para gráficos e estatísticas
    for i, (metric, title, ylabel) in enumerate(zip(metrics, titles, ylabels)):
        col1, col2 = st.columns([3, 1])

        with col1:
            # Gráfico de evolução
            fig, ax = plt.subplots(figsize=(10, 4))

            for group in ['Nondemented', 'Converted', 'Demented']:
                group_data = df[df['Group_Type'] == group]

                # Calcular média e desvio padrão por visita
                means = group_data.groupby('visit')[metric].mean()
                stds = group_data.groupby('visit')[metric].std()
                counts = group_data.groupby('visit')[metric].count()

                # Plotar linha da média
                ax.plot(means.index, means, marker='o', color=group_colors[group],
                        label=f'{group} (n={counts.max()})')

                # Plotar área do desvio padrão
                ax.fill_between(means.index,
                                means - stds,
                                means + stds,
                                color=group_colors[group], alpha=0.2)

            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.set_xlabel('Número da Visita')
            ax.grid(False)
            ax.legend()
            remove_background(ax)
            st.pyplot(fig)

        with col2:
            # Testes estatísticos para a primeira visita
            first_visit = df[df['visit'] == 1]
            groups_data = [first_visit[first_visit['Group_Type'] == group][metric]
                           for group in ['Nondemented', 'Converted', 'Demented']]

            # Verificar normalidade
            norm_results = [stats.shapiro(group)[1] for group in groups_data]
            all_normal = all(p > 0.05 for p in norm_results)

            if all_normal:
                # ANOVA
                f_stat, p_value = stats.f_oneway(*groups_data)
                test_type = "ANOVA"
            else:
                # Kruskal-Wallis
                h_stat, p_value = stats.kruskal(*groups_data)
                test_type = "Kruskal-Wallis"

            st.markdown(f"**Teste {test_type}**")
            st.write(f"Estatística: {f_stat if all_normal else h_stat:.3f}")
            st.write(f"Valor-p: {p_value:.4f}")
            st.write("Diferença significativa" if p_value < 0.05 else "Sem diferença significativa")

    # Estatísticas resumidas
    with st.expander("Ver Estatísticas Detalhadas"):
        st.subheader("Estatísticas Resumidas por Grupo e Visita")
        for metric in metrics:
            st.write(f"\n**Métrica: {metric.upper()}**")
            stats_df = df.groupby(['Group_Type', 'visit'])[metric].agg(['mean', 'std', 'count'])
            st.dataframe(stats_df.style.format("{:.3f}"))

def metrics_section(df_cross, df_long):
    st.header("Métricas e Qualidade dos Dados")

    # Seção 1: Limpeza dos Dados
    st.subheader("1. Limpeza e Pré-processamento")
    st.markdown("""
    - **Exclusão de sujeitos com dados faltantes**: 
      Removemos todos os registros onde valores essenciais como `Age`, `MMSE`, `CDR` ou `nWBV` estavam ausentes.
      Essa abordagem garante que nossas análises sejam baseadas apenas em dados completos.
    - **Padronização**: 
      Todos os nomes de colunas foram padronizados para formato snake_case (ex: 'M/F' → 'm_f').
    """)

    # Métricas de limpeza em colunas
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Dados Transversais**")
        st.metric("Registros originais", len(df_cross))
        st.metric("Registros após limpeza", len(df_cross.dropna(subset=['age', 'mmse', 'cdr', 'nwbv'])))

    with col2:
        st.markdown("**Dados Longitudinais**")
        st.metric("Registros originais", len(df_long))
        st.metric("Registros após limpeza", len(df_long.dropna(subset=['age', 'mmse', 'cdr', 'nwbv'])))

    # Divisão visual
    st.markdown("---")

    # Seção 2: Explicação Estatística
    st.subheader("2. Testes Estatísticos")

    # Explicação sobre ANOVA
    st.markdown("""
    ### ANOVA (Análise de Variância)
    """)

    with st.expander("🔍 Clique para expandir a explicação"):
        st.markdown("""
        **O que é?**  
        A ANOVA é um teste estatístico paramétrico que compara as médias de três ou mais grupos independentes.

        **Quando usar?**  
        - Quando os dados seguem distribuição normal (teste de Shapiro-Wilk)  
        - Quando as variâncias são homogêneas (teste de Levene)  
        - Para dados intervalares/racionais  

        **Interpretação:**  
        - Valor-p < 0.05 → Diferenças significativas entre grupos  
        - Valor-p ≥ 0.05 → Sem diferenças significativas  

        **Fórmula básica:**  
        """)
        st.latex(r'''
        F = \frac{\text{Variância entre grupos}}{\text{Variância dentro dos grupos}}
        ''')

        st.markdown("""
        **Exemplo no nosso contexto:**  
        Usamos ANOVA para comparar:  
        - Volume cerebral (nWBV) entre grupos (Nondemented, Converted, Demented)  
        - Escores MMSE entre diferentes estágios de CDR  
        """)

    # Comparação com Kruskal-Wallis
    st.markdown("""
    ### ANOVA vs Kruskal-Wallis
    """)

    st.table(pd.DataFrame({
        'Característica': ['Pressupostos', 'Tipo de dados', 'Robustez'],
        'ANOVA': [
            'Normalidade, homogeneidade de variâncias',
            'Dados paramétricos',
            'Sensível a outliers'
        ],
        'Kruskal-Wallis': [
            'Nenhum pressuposto',
            'Dados não-paramétricos/ordinais',
            'Robusto a outliers'
        ]
    }))

    # Divisão visual
    st.markdown("---")

    # Seção 3: Exemplo Prático
    st.subheader("3. Aplicação no Nosso Dataset")

    # Selecionar variável para demonstração
    demo_var = st.selectbox(
        "Selecione uma variável para demonstração estatística:",
        options=['nwbv', 'mmse', 'age']
    )

    # Executar ANOVA
    from scipy.stats import f_oneway
    groups = [df_long[df_long['group'] == g][demo_var] for g in df_long['group'].unique()]

    # Verificar normalidade
    from scipy.stats import shapiro
    normal = all(shapiro(group)[1] > 0.05 for group in groups)

    if normal:
        f_val, p_val = f_oneway(*groups)
        st.success(f"✅ Dados normais (p > 0.05 no teste de Shapiro-Wilk)")
        st.metric("Resultado ANOVA",
                  f"F = {f_val:.2f}, p = {p_val:.4f}",
                  help="Valor-p < 0.05 indica diferenças significativas")
    else:
        st.warning("⚠️ Dados não-normais - Use Kruskal-Wallis")
def main():
    # Carrega os dados
    df_cross, df_long = load_data()

    # Menu lateral
    st.sidebar.title("Navegação")
    section = st.sidebar.radio("Seções", ["Motivação", "Longitudinal", "Métricas"])

    # Seção selecionada
    if section == "Motivação":
        motivation_section(df_cross)
    elif section == "Longitudinal":
        #st.warning("Implementação em progresso - esta seção será desenvolvida na próxima iteração")
        longitudinal_section(df_long)

    elif section == "Métricas":
        #st.warning("Implementação em progresso - esta seção será desenvolvida na próxima iteração")
        metrics_section(df_cross, df_long)


if __name__ == "__main__":
    main()
