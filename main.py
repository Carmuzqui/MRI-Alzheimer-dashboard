import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import linregress
from sklearn.metrics import r2_score
from streamlit_option_menu import option_menu
import qrcode
import io

from plotly.subplots import make_subplots
import plotly.graph_objects as go

# from PIL import Image

# Configurações iniciais
st.set_page_config(page_title="Dashboard Alzheimer", layout="wide", page_icon="🧠")
sns.set_palette("husl")

# Estilo CSS personalizado
st.markdown("""
<style>
    /* Estilo geral para o tema escuro */
    [data-testid="stAppViewContainer"] {
        background-color: #1E1E1E;
        color: #FFFFFF;
    }

    /* Estilo para o container do menu de opções */
    .stSelectbox, .stMultiSelect {
        background-color: transparent !important;
    }

    /* Estilo para o menu de opções */
    #MainMenu {
        background-color: transparent !important;
    }

    /* Estilo para os botões do menu */
    .nav-link {
        background-color: rgba(60, 60, 60, 0.5) !important;
        color: #FFFFFF !important;
        border-radius: 5px !important;
        margin-bottom: 5px !important;
        transition: all 0.3s ease;
    }

    /* Estilo para o botão selecionado */
    .nav-link.active {
        background-color: #02ab21 !important;
        color: #FFFFFF !important;
    }

    /* Estilo para hover nos botões */
    .nav-link:hover {
        background-color: rgba(80, 80, 80, 0.7) !important;
    }

    /* Estilo para os ícones */
    .nav-link .icon {
        color: #FFD700 !important;  /* Amarelo para os ícones */
    }

    /* Remover a borda do container do menu */
    .css-1l4firl {
        border: none !important;
        background-color: transparent !important;
    }
</style>
""", unsafe_allow_html=True)

def create_qr_code(url):
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="white", back_color="transparent")
    return img

def add_vertical_space(num_lines: int = 1):
    """Add vertical space to your Streamlit app."""
    for _ in range(num_lines):
        st.markdown('<br>', unsafe_allow_html=True)

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
    cross = pd.read_csv("data/oasis_cross-sectional.csv")
    long = pd.read_csv("data/oasis_longitudinal.csv")

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


def demographic_info_linha(df):
    """Exibe informações demográficas básicas em formato vertical"""
    media_idade = df['age'].mean()
    desvio_padrao = df['age'].std()
    percent_mulheres = (df['m_f'].value_counts(normalize=True)['F'] * 100)

    # Container para organizar as métricas verticalmente
    with st.container():
        st.metric("Média de idade", f"{media_idade:.1f} anos")
        st.metric("Desvio padrão idade", f"{desvio_padrao:.1f} anos")
        st.metric("Percentual de mulheres", f"{percent_mulheres:.1f}%")

def plot_age_distribution(df):
    """Cria gráfico de distribuição de idade em percentual, subdividido por gênero"""
    # Definir bins de 10 em 10 anos, começando do 0
    bins = list(range(0, 101, 10))  # 0, 10, 20, ..., 90, 100
    labels = [f'{i}-{i+9}' for i in range(0, 90, 10)] + ['90+']

    # Calcular a distribuição
    df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=False, include_lowest=True)
    
    # Calcular percentuais totais e por gênero
    total_distribution = df['age_group'].value_counts(normalize=True) * 100
    gender_distribution = df.groupby(['age_group', 'm_f']).size().unstack(fill_value=0)
    gender_percentages = gender_distribution.div(gender_distribution.sum(axis=1), axis=0)

     # Garantir que todas as faixas etárias estejam presentes, mesmo que com valor zero
    for label in labels:
        if label not in total_distribution.index:
            total_distribution[label] = 0
            gender_percentages.loc[label] = [0, 0]
    
    # Ordenar os dados
    total_distribution = total_distribution.sort_index()
    gender_percentages = gender_percentages.sort_index()
    
    # Calcular os valores para as barras
    male_values = total_distribution * gender_percentages['M']
    female_values = total_distribution * gender_percentages['F']

    # Criar o gráfico com Plotly
    plotly_fig = go.Figure()

    # Adicionar barras para homens
    plotly_fig.add_trace(
        go.Bar(
            x=male_values.index,
            y=male_values.values,
            name='Masculino',
            marker_color='#13ecc1',  # Azul mais intenso
            hovertemplate='Faixa Etária: %{x}<br>Total: %{customdata:.1f}%<br>Masculino: %{text:.1f}%<extra></extra>',
            text=gender_percentages['M'] * 100,
            customdata=total_distribution,
            textposition='none'  # Remove os valores das barras
        )
    )

    # Adicionar barras para mulheres
    plotly_fig.add_trace(
        go.Bar(
            x=female_values.index,
            y=female_values.values,
            name='Feminino',
            marker_color='#fa2eea',  # Rosa mais intenso
            hovertemplate='Faixa Etária: %{x}<br>Total: %{customdata:.1f}%<br>Feminino: %{text:.1f}%<extra></extra>',
            text=gender_percentages['F'] * 100,
            customdata=total_distribution,
            textposition='none'  # Remove os valores das barras
        )
    )

    # Configurar o layout
    plotly_fig.update_layout(
        title='Distribuição de Idade dos Participantes por Gênero (%)',
        xaxis_title='Faixa etária (anos)',
        yaxis_title='Percentual de participantes (%)',
        barmode='stack',
        yaxis_range=[0, max(total_distribution) * 1.1],
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        title_font=dict(size=14),
        xaxis_title_font=dict(size=12),
        yaxis_title_font=dict(size=12),
    )

    plotly_fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)', categoryorder='array', categoryarray=labels)
    plotly_fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)', tickformat='.1f')

    st.plotly_chart(plotly_fig, use_container_width=True)


# def plot_age_distribution(df):
#     """Cria gráfico de distribuição de idade em percentual"""
#     fig, ax = plt.subplots(figsize=(8, 4))
#     n, bins, patches = ax.hist(df['age'], bins=15, color='skyblue',
#                                edgecolor='black',
#                                weights=np.ones(len(df['age'])) / len(df['age']) * 100)

#     ax.set_title('Distribuição de Idade dos Participantes (%)', fontsize=14)
#     ax.set_xlabel('Idade (anos)', fontsize=12)
#     ax.set_ylabel('Percentual de Participantes (%)', fontsize=12)
#     ax.grid(axis='y', alpha=0.3)
#     ax.set_ylim(0, 30)
#     remove_background(ax)
#     st.pyplot(fig)


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
    #with st.expander("Estatísticas de Regressão"):
    #    st.code(stats_text)


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
    # Título com caixa expansiva de texto informativo
    col_title, col_expand = st.columns([0.7, 0.3])
    with col_title:
        st.header("Análise Transversal de Dados de Alzheimer")
    with col_expand:
        with st.expander("ℹ️ Informações"):
            st.write("""
            Este dashboard explora dados de ressonância magnética (MRI) e marcadores 
            clínicos de pacientes com Alzheimer em diferentes estágios e indivíduos 
            saudáveis. A análise transversal permite comparar grupos em um único 
            momento no tempo.
            """)

    # Seção superior com informações demográficas e gráfico de distribuição
    col_stats, col_graph = st.columns([0.2, 0.8])  # Ajuste as proporções conforme necessário

    with col_stats:
        # Chamada modificada para demographic_info (você precisará adaptar essa função)
        st.markdown("**Estatísticas Demográficas**")
        demographic_info_linha(df)  # Esta função precisa retornar os valores em formato vertical

    with col_graph:
        # Gráfico de distribuição de idade com mais espaço
        plot_age_distribution(df)

    # Seção do limiar de idade com slider
    st.markdown("---")
    col_threshold_label, col_threshold_slider = st.columns([0.35, 0.65])
    with col_threshold_label:
        st.subheader("Análise por CDR")
    with col_threshold_slider:
        min_age, max_age = int(df['age'].min()), int(df['age'].max())
        threshold = st.slider(
            "",
            min_value=min_age,
            max_value=max_age,
            value=55,
            key="motivation_threshold"
        )

    # Três gráficos superiores lado a lado
    col1, col2, col3 = st.columns(3)

    with col1:
        plot_violin_age_cdr(df, threshold)

    with col2:
        plot_violin_nwbv_cdr(df, threshold)

    with col3:
        plot_boxplot_cdr_mmse(df, threshold)

    # Dois gráficos médios lado a lado
    st.subheader("Análise por Idade")
    col_mid1, col_mid2 = st.columns(2)

    with col_mid1:
        plot_scatter_mmse_age(df, threshold)

    with col_mid2:
        plot_scatter_age_nwbv(df, threshold)


def longitudinal_section(df):
    st.header("Análise Longitudinal de Dados de Alzheimer")
    st.write("""
    Esta seção examina a progressão dos marcadores clínicos e de imagem ao 
    longo do tempo em pacientes com Alzheimer e controles saudáveis, incluindo
    indivíduos que converteram de não dementes para dementes durante o estudo.
    """)



    # Create columns for age distribution and correlation matrix
    col1, col2 = st.columns(2)

    with col1:
        # Gráfico de distribuição de idade
        # Informações demográficas
        demographic_info(df)
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
    with st.sidebar:
        st.title("Dashboard de Alzheimer")
        # Menu de opções
        selected = option_menu(
            menu_title=None,
            options=["Início", "Motivação", "Longitudinal", "Métricas"],
            icons=["house", "lightbulb", "graph-up", "clipboard-data"],
            menu_icon="cast",
            default_index=0,
            orientation="vertical",  # Mudado para vertical
            styles={
                "container": {"padding": "0!important", "background-color": "transparent"},
                "icon": {"color": "#FFD700", "font-size": "25px"},  # Amarelo para os ícones
                "nav-link": {
                    "font-size": "16px", 
                    "text-align": "left", 
                    "margin":"0px", 
                    "padding": "10px",
                    "--hover-color": "rgba(80, 80, 80, 0.7)"
                },
                "nav-link-selected": {"background-color": "#02ab21"},
            }
        )

    # Conteúdo principal baseado na seleção
    if selected == "Início":
                
        st.header("Bem-vindo ao análise de Alzheimer")
                
        # Criando colunas para melhor layout
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.write("Vídeo introdutório de algumas características do Alzheimer:")
            
            # Carregando o vídeo
            video_file = open('media/alzheimer.mp4', 'rb')
            video_bytes = video_file.read()
            
            # Exibindo o vídeo
            st.video(video_bytes)
            
            # Adicionando a referência alinhada à esquerda
            st.markdown("""
            <div style="font-size: 0.8em; color: gray; text-align: center; margin-top: 5px;">
            Fonte: <a href="https://www.nia.nih.gov/health/alzheimers-causes-and-risk-factors/what-happens-brain-alzheimers-disease" >National Institute on Aging</a>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            add_vertical_space(2)  # Adiciona dois espaços verticais
            st.write("Principais recursos:")
            st.write("• Análise de dados de pacientes")
            st.write("• Visualizações interativas")
            st.write("• Insights sobre Alzheimer")
            
            # Criação do QR code
            url = "https://mri-alzheimer-dashboard-k2hhdkapmydcfb8zdnvfxe.streamlit.app/"
            qr_img = create_qr_code(url)
            
            # Convertendo a imagem para bytes
            img_byte_arr = io.BytesIO()
            qr_img.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            
            st.image(img_byte_arr, caption='Escaneie para acessar o dashboard')
            

    elif selected == "Motivação":
        motivation_section(df_cross)

    elif selected == "Longitudinal":
        longitudinal_section(df_long)

    elif selected == "Métricas":
        metrics_section(df_cross, df_long)
        
    
        


if __name__ == "__main__":
    main()
