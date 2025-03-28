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
import statsmodels.api as sm
from statsmodels.formula.api import ols


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
            hovertemplate='Faixa etária: %{x}<br>Total: %{customdata:.1f}%<br>Masculino: %{text:.1f}%<extra></extra>',
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
            hovertemplate='Faixa etária: %{x}<br>Total: %{customdata:.1f}%<br>Feminino: %{text:.1f}%<extra></extra>',
            text=gender_percentages['F'] * 100,
            customdata=total_distribution,
            textposition='none'  # Remove os valores das barras
        )
    )

    # Configurar o layout
    plotly_fig.update_layout(
        title='Distribuição de idade dos participantes por gênero (%)',
        xaxis_title='Faixa etária (anos)',
        yaxis_title='Percentual de participantes (%)',
        barmode='stack',
        yaxis_range=[0, max(total_distribution) * 1.1],
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        title_font=dict(size=20),
        xaxis_title_font=dict(size=16),
        yaxis_title_font=dict(size=16),
    )

    plotly_fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)', categoryorder='array', categoryarray=labels)
    plotly_fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)', tickformat='.1f')

    st.plotly_chart(plotly_fig, use_container_width=True)


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
            color='yellow', linestyle='--', label=f'Todos (R²={r2_all:.2f})')

    # Ajuste de regressão apenas para idade >= threshold
    df_above = df[df['age'] >= threshold]
    if len(df_above) > 1:
        slope_above, intercept_above, _, _, _ = linregress(df_above['age'], df_above['nwbv'])
        r2_above = r2_score(df_above['nwbv'], intercept_above + slope_above * df_above['age'])
        ax.plot(df_above['age'], intercept_above + slope_above * df_above['age'],
                color='red', linestyle='--', label=f'≥ {threshold} (R²={r2_above:.2f})')

    ax.set_title(f'Volume cerebral vs Idade (Limiar: {threshold})')
    ax.set_xlabel('Idade (anos)')
    ax.set_ylabel('Volume cerebral normalizado (nWBV)')
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

    fig, ax = plt.subplots(figsize=(8, 5))
    # sns.boxplot(data=df, x='cdr', y='mmse', hue='age_group', ax=ax)
    sns.boxplot(data=df, x='cdr', y='mmse', hue='age_group', ax=ax,
                flierprops={'markerfacecolor': 'white', 'markeredgecolor': 'white'},
                whiskerprops={'color': 'white'},
                medianprops={'color': 'white'},
                capprops={'color': 'white'})
    ax.set_title(f'MMSE vs CDR (Limiar: {threshold})', fontsize=18)
    ax.set_xlabel('Taxa de demência clínica (CDR)', fontsize=16)
    ax.set_ylabel('Mini-Exame do estado mental (MMSE)', fontsize=16)
    # ax.legend(title='Faixa etária',labelcolor= 'white')
    legend = ax.legend(title='Faixa etária', labelcolor='white', title_fontsize='10')
    legend.get_title().set_color('white')  # Define a cor do título da legenda como branco
    remove_background(ax)
    st.pyplot(fig)


def plot_violin_nwbv_cdr(df, threshold):
    """Violin plot nWBV x CDR com threshold"""
    df['age_group'] = np.where(df['age'] >= threshold, f'≥ {threshold}', f'< {threshold}')

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.violinplot(data=df, x='cdr', y='nwbv', hue='age_group',
                   split=True, inner="quart", ax=ax)
    ax.set_title(f'Volume cerebral vs CDR (Limiar: {threshold})', fontsize=18)
    ax.set_xlabel('Taxa de demência clínica (CDR)', fontsize=16)
    ax.set_ylabel('Volume cerebral normalizado (nWBV)', fontsize=16)
    # ax.legend(title='Faixa etária')
    legend = ax.legend(title='Faixa etária', labelcolor='white', title_fontsize='10')
    legend.get_title().set_color('white')  # Define a cor do título da legenda como branco
    remove_background(ax)
    st.pyplot(fig)


def plot_scatter_mmse_age(df, threshold):
    """Scatter plot MMSE x Idade com threshold"""
    df['age_group'] = np.where(df['age'] >= threshold, f'≥ {threshold}', f'< {threshold}')

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.scatterplot(data=df, x='age', y='mmse', hue='age_group',
                    alpha=0.7, ax=ax)
    ax.set_title(f'MMSE vs Idade (Limiar: {threshold})')
    ax.set_xlabel('Idade (anos)')
    ax.set_ylabel('Mini-Exame do estado mental (MMSE)')
    legend = ax.legend(title='Faixa etária', labelcolor='white', title_fontsize='10')
    legend.get_title().set_color('white')  # Define a cor do título da legenda como branco
    remove_background(ax)
    st.pyplot(fig)


def plot_violin_age_cdr(df, threshold):
    """Violin plot Idade x CDR com threshold"""
    df['age_group'] = np.where(df['age'] >= threshold, f'≥ {threshold}', f'< {threshold}')

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.violinplot(data=df, x='cdr', y='age', hue='age_group',
                   split=True, inner="quart", ax=ax)
    ax.set_title(f'Idade vs CDR (Limiar: {threshold})', fontsize=18)
    ax.set_xlabel('Taxa de demência clínica (CDR)', fontsize=16)
    ax.set_ylabel('Idade (anos)', fontsize=16)
    # ax.legend(title='Faixa etária')
    legend = ax.legend(title='Faixa etária', labelcolor='white', title_fontsize='10')
    legend.get_title().set_color('white')  # Define a cor do título da legenda como branco
    remove_background(ax)
    st.pyplot(fig)


# Interface principal ==========================================================

def motivation_section(df):
    # Título com caixa expansiva de texto informativo
    col_title, col_expand = st.columns([0.7, 0.3])
    with col_title:
        st.header("Análise transversal de dados de Alzheimer")
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
        st.markdown("**Estatísticas demográficas**")
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
            "Limiar",
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
    st.subheader("Análise por idade")

    # Remova os valores NaN da coluna 'cdr'
    df_cdr = df.dropna(subset=['cdr'])

    # Obtenha os valores únicos de CDR do DataFrame, excluindo NaN
    cdr_options = sorted(df_cdr['cdr'].unique())

    # CSS personalizado para juntar as caixas de verificação
    st.markdown("""
    <style>
        div.row-widget.stRadio > div{flex-direction:row;}
        div.row-widget.stRadio > div > label{margin-right: 5px; padding-right: 5px;}
    </style>
    """, unsafe_allow_html=True)

    # Crie um container para as caixas de seleção
    st.write("Selecione os valores de CDR para filtrar:")

    # Crie as caixas de seleção horizontalmente com menos espaço entre elas
    cols = st.columns(len(cdr_options) + 1)  # +1 para dar um pouco de espaço extra no final
    selected_cdrs = {}
    for idx, cdr in enumerate(cdr_options):
        with cols[idx]:
            selected_cdrs[cdr] = st.checkbox(f"CDR {cdr}", value=True, key=f"cdr_{cdr}")

    # Filtrar o DataFrame baseado nas seleções
    filtered_df = df[df['cdr'].isin([cdr for cdr, selected in selected_cdrs.items() if selected])]

    # Verifique se algum CDR foi selecionado
    if filtered_df.empty:
        st.warning("Por favor, selecione pelo menos um valor de CDR para visualizar os gráficos.")
    else:
        # Dois gráficos médios lado a lado
        col_mid1, col_mid2 = st.columns(2)

        with col_mid1:
            plot_scatter_mmse_age(filtered_df, threshold)

        with col_mid2:
            plot_scatter_age_nwbv(filtered_df, threshold)
            
        
        
        
        
        


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
        st.subheader("Matriz de correlação")
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
####################################################################################
    st.subheader("Evolução do Quadro Clínico")

    # Configurações dos gráficos
    metrics = ['nwbv', 'cdr', 'mmse']
    titles = ['Volume Cerebral Normalizado (nWBV)',
              'Taxa de Demência Clínica (CDR)',
              'Mini Exame do Estado Mental (MMSE)']
    ylabels = ['nWBV', 'CDR', 'MMSE']

    # Criar 3 colunas para os gráficos
    cols = st.columns(3)

    for i, (metric, title, ylabel) in enumerate(zip(metrics, titles, ylabels)):
        with cols[i]:
            # Gráfico de evolução
            fig, ax = plt.subplots(figsize=(8, 4))

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

            # Caixa expansiva para estatísticas
            with st.expander(f"Estatísticas - {title}"):
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

                if p_value < 0.05:
                    st.success("Diferença significativa (p < 0.05)")
                else:
                    st.info("Sem diferença significativa")
#################################################################################
    st.subheader("Análise Longitudinal do Volume Cerebral")

    # Verificar e padronizar os nomes dos grupos
    #st.write("Grupos únicos encontrados:", df['Group_Type'].unique())

    # Verificar se estamos usando os nomes corretos dos grupos
    group_names = {
        'Nondemented': ['Nondemented', 'NonDemented', 'Control'],
        'Demented': ['Demented', 'Demented/Converted', 'Dementia']
    }

    # Padronizar nomes dos grupos
    df['Group_Type'] = df['Group_Type'].str.strip()
    for standardized, variants in group_names.items():
        for variant in variants:
            df.loc[df['Group_Type'].str.lower() == variant.lower(), 'Group_Type'] = standardized

    # Definir variável de análise
    VAR = 'nwbv'
    if VAR not in df.columns:
        st.error(f"Coluna '{VAR}' não encontrada no DataFrame")
        return

    # Criar coluna de tempo se não existir
    if 'mr delay' in df.columns:
        df['Years_Since_Baseline'] = df['mr delay'] / 365.25
    else:
        st.error("Coluna 'mr delay' não encontrada")
        return

    # Criar layout
    VAR = 'nwbv'

    # Processar dados
    df = df.copy()
    df['group_type'] = df['group'].replace({'Demented': 'Demented', 'Converted': 'Demented'})

    # Calcular idade real (idade na primeira visita + anos desde baseline)
    df['first_visit_age'] = df.groupby('subject id')['age'].transform('first')
    df['years_since_baseline'] = df['mr delay'] / 365.25
    df['real_age'] = df['first_visit_age'] + df['years_since_baseline']

    col1, col2 = st.columns([6, 4])

    with col1:
        # Configurar estilo
        sns.set_style("whitegrid")
        colors = {
            'Nondemented': '#b3933c',  # Amarelo/ouro
            'Demented': '#d88893'  # Vermelho claro
        }

        # Criar figura
        fig, ax = plt.subplots(figsize=(10, 6))

        # Verificar quais grupos existem após padronização
        existing_groups = [g for g in colors.keys() if g in df['group_type'].unique()]

        if not existing_groups:
            st.error("Nenhum grupo válido encontrado após padronização")
            return

        for group in existing_groups:
            group_data = df[df['group_type'] == group]

            if len(group_data) < 3:  # Mínimo de pontos para regressão
                st.warning(f"Dados insuficientes para {group} (n={len(group_data)})")
                continue

            # Plotar pontos
            sns.scatterplot(
                x='real_age',
                y=VAR,
                data=group_data,
                color=colors[group],
                ax=ax,
                alpha=0.6,
                s=80,
                label=f"{group} (n={len(group_data)})"
            )

            # Ajustar regressão
            try:
                X = sm.add_constant(group_data['real_age'])
                y = group_data[VAR]
                model = sm.OLS(y, X).fit()

                # Gerar predições
                x_pred = np.linspace(
                    group_data['real_age'].min(),
                    group_data['real_age'].max(),
                    100
                )
                y_pred = model.predict(sm.add_constant(x_pred))
                conf_int = model.get_prediction(sm.add_constant(x_pred)).conf_int()

                # Plotar linha e intervalo
                ax.plot(x_pred, y_pred, color=colors[group], linewidth=2.5)
                ax.fill_between(
                    x_pred, conf_int[:, 0], conf_int[:, 1],
                    color=colors[group], alpha=0.15
                )
            except Exception as e:
                st.warning(f"Erro na regressão para {group}: {str(e)}")

        ax.set_title(f'Evolução do {VAR.upper()} por Idade', pad=20)
        ax.set_ylabel(VAR.upper())
        ax.set_xlabel('Idade (anos)')
        ax.legend(title='Grupo Clínico', frameon=True)
        ax.grid(False)
        sns.despine()
        remove_background(ax)
        st.pyplot(fig)

    with col2:
        with st.expander("**📊 Resultados Estatísticos**", expanded=True):
            # ANCOVA (ANOVA com covariável de idade)
            try:
                # Garantir que o nome do grupo está consistente
                df['group_type'] = df['group_type'].replace({'Demented/Converted': 'Demented'})

                model_ancova = ols(f'{VAR} ~ C(group_type) + real_age', data=df).fit()
                anova_table = sm.stats.anova_lm(model_ancova, typ=2)

                # Coeficientes - agora usando os nomes corretos conforme saída do modelo
                coef = model_ancova.params
                se = model_ancova.bse
                r2 = model_ancova.rsquared

                # Verificar qual é a referência do grupo
                if 'C(group_type)[T.Nondemented]' in coef:
                    # Se a referência é Demented
                    dementia_coef = coef['C(group_type)[T.Nondemented]']
                    dementia_se = se['C(group_type)[T.Nondemented]']
                    coef_text = f"Nondemented (ref: Demented):\n{dementia_coef:.4f} ± {dementia_se:.4f}"
                else:
                    # Se a referência é Nondemented (caso contrário)
                    dementia_coef = coef['C(group_type)[T.Demented]']
                    dementia_se = se['C(group_type)[T.Demented]']
                    coef_text = f"Demented (ref: Nondemented):\n{dementia_coef:.4f} ± {dementia_se:.4f}"

                st.markdown("**Teste ANCOVA (Grupo + Idade)**")
                st.markdown(f"""
                            **Efeito do Grupo**  
                            F: {anova_table['F']['C(group_type)']:.1f}  
                            p: {anova_table['PR(>F)']['C(group_type)']:.4f}

                            **Efeito da Idade**  
                            F: {anova_table['F']['real_age']:.1f}  
                            p: {anova_table['PR(>F)']['real_age']:.4f}
                            """)

                if anova_table['PR(>F)']['C(group_type)'] < 0.05:
                    st.success("Diferença significativa entre grupos (p < 0.05)")
                else:
                    st.warning("Sem diferença significativa entre grupos")
            except Exception as e:
                st.error(f"Erro na análise estatística: {str(e)}")
                st.text("Detalhes do modelo:")
                st.text(model_ancova.summary() if 'model_ancova' in locals() else "Modelo não pôde ser criado")
        with st.expander("**📊 Regressão Linear Múltipla**", expanded=True):
            # ANCOVA (ANOVA com covariável de idade)
            try:
                # Garantir que o nome do grupo está consistente
                df['group_type'] = df['group_type'].replace({'Demented/Converted': 'Demented'})

                model_ancova = ols(f'{VAR} ~ C(group_type) + real_age', data=df).fit()
                anova_table = sm.stats.anova_lm(model_ancova, typ=2)

                # Coeficientes - agora usando os nomes corretos conforme saída do modelo
                coef = model_ancova.params
                se = model_ancova.bse
                r2 = model_ancova.rsquared

                # Verificar qual é a referência do grupo
                if 'C(group_type)[T.Nondemented]' in coef:
                    # Se a referência é Demented
                    dementia_coef = coef['C(group_type)[T.Nondemented]']
                    dementia_se = se['C(group_type)[T.Nondemented]']
                    coef_text = f"Nondemented (ref: Demented):\n{dementia_coef:.4f} ± {dementia_se:.4f}"
                else:
                    # Se a referência é Nondemented (caso contrário)
                    dementia_coef = coef['C(group_type)[T.Demented]']
                    dementia_se = se['C(group_type)[T.Demented]']
                    coef_text = f"Demented (ref: Nondemented):\n{dementia_coef:.4f} ± {dementia_se:.4f}"

                st.markdown(f"""
                **Modelo de Regressão**

                **R² = {r2:.3f}**

                **Intercepto:**  
                {coef['Intercept']:.4f} ± {se['Intercept']:.4f}

                **Nondemented (ref: Demented):**  
                {coef['C(group_type)[T.Nondemented]']:.4f} ± {se['C(group_type)[T.Nondemented]']:.4f}

                **Idade:**  
                {coef['real_age']:.4f} ± {se['real_age']:.4f}
                """)

            except Exception as e:
                st.error(f"Erro na análise estatística: {str(e)}")
                st.text("Detalhes do modelo:")
                st.text(model_ancova.summary() if 'model_ancova' in locals() else "Modelo não pôde ser criado")
def metrics_section(df_cross, df_long):
    st.header("Métricas e Qualidade dos Dados")

    # Seção 1: Limpeza dos Dados
    st.subheader("1. Limpeza e Pré-processamento")
    st.markdown("""
    - **Exclusão de sujeitos com dados faltantes**: 
      Removemos todos os registros onde valores essenciais como idade, volume cerebral, ídice de CDR e MMSE estavam ausentes.""")
    # Dados iciciais e finais em cada um dos bancos de dados:
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
        ### Kruskal-Wallis
        """)
    with st.expander("🔍 Clique para expandir a explicação"):
        st.markdown("""
        **O que é?**  
        O teste de Kruskal-Wallis é um teste estatístico não paramétrico que compara as distribuições de três ou mais grupos independentes.

        **Quando usar?**  
        - Quando os dados **não** seguem distribuição normal (teste de Shapiro-Wilk)  
        - Quando há heterogeneidade de variâncias (teste de Levene)  
        - Para dados ordinais ou quando há outliers que podem afetar a ANOVA  

        **Interpretação:**  
        - Valor-p < 0.05 → Pelo menos um grupo difere significativamente  
        - Valor-p ≥ 0.05 → Nenhuma diferença significativa detectada  

        **Fórmula básica:**  
        """)
        st.latex(r'''
        H = \frac{12}{N(N+1)} \sum \frac{R_i^2}{n_i} - 3(N+1)
        ''')
    st.markdown("""
            ### Regressão Linear Múltipla
            """)
    with st.expander("🔍 Clique para expandir a explicação"):
        st.markdown("""
        **O que foi feito?**  
        Ajustamos um modelo de **Regressão Linear Múltipla** para analisar a relação entre o volume cerebral normalizado (**nWBV**) e dois fatores:
        - O tempo desde a linha de base 
        - A presença de demência

        **Como foi feito?**  
        - Os grupos **Demente** e **Convertido** foram unificados em um único grupo: **Demente/Convertido**  
        - Criamos uma variável binária (**Dementia**) para indicar se um indivíduo pertence a esse grupo (1) ou não (0)  
        - Ajustamos um modelo de regressão linear com:""")
        st.latex(r'''
            nWBV = \beta_0 + \beta_1 (\text{Years Since Baseline}) + \beta_2 (\text{Dementia}) + \varepsilon
            ''')

        st.markdown("""
            - O modelo ajuda a entender o impacto do tempo e da demência na atrofia cerebral (redução de nWBV).  
            """)

def main():
    # Carrega os dados
    df_cross, df_long = load_data()
    
    # Menu lateral
    with st.sidebar:
        st.title("Dashboard de Alzheimer")
        # Menu de opções
        selected = option_menu(
            menu_title=None,
            options=["Início", "Transversal", "Longitudinal", "Métricas"], #Motivação
            icons=["house", "layers", "graph-up", "clipboard-data"],     #lightbulb
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
            

    elif selected == "Transversal":  #Motivação
        motivation_section(df_cross)

    elif selected == "Longitudinal":
        longitudinal_section(df_long)

    elif selected == "Métricas":
        metrics_section(df_cross, df_long)


if __name__ == "__main__":
    main()
