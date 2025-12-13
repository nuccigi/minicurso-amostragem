import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import norm

from data import generate_population
from sampling_functions import (
    sample_simple_random,
    sample_stratified
)
from calculations import sample_size_mean
from plots import plot_distribution

# ================================
# FUNÇÃO AUXILIAR – FORMATAÇÃO PT-BR
# ================================

def fmt_br(x, dec=2):
    return f"{x:,.{dec}f}".replace(",", "X").replace(".", ",").replace("X", ".")

# ================================
# CONFIGURAÇÃO DO APP
# ================================

st.set_page_config(
    page_title="Simulador de Amostragem – Laplace",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# LOGO
col_logo1, col_logo2 = st.columns([6, 1])
with col_logo2:
    st.image("assets/logo.png", width=120)

st.title("Simulador de Amostragem – Minicurso Laplace")
st.write("Explore conceitos de tamanho de amostra e métodos de seleção.")

laplace_colors = ["#0f2f56", "#22576d", "#296872", "#045243"]

# ============================================================
# 1. EXEMPLO DE POPULAÇÃO
# ============================================================

st.markdown("<br><br>", unsafe_allow_html=True)
st.header("1. Exemplo de população")

N = st.slider("Tamanho da população", 1_000, 50_000, 10_000)
population = generate_population(N)

VAR_INTERESSE = "renda"

# 👉 LIMITES GLOBAIS PARA COMPARAÇÃO DOS GRÁFICOS
x_min = population[VAR_INTERESSE].min()
x_max = population[VAR_INTERESSE].max()

col1, col2 = st.columns(2)

# ---------------------------
# Tabela e estatísticas da população
# ---------------------------
with col1:
    st.subheader("Primeiras linhas do exemplo de população")
    st.dataframe(population.head().reset_index(drop=True))

    pop_mean = population[VAR_INTERESSE].mean()
    pop_sd = population[VAR_INTERESSE].std()

    pop_stats_df = pd.DataFrame({
        "Estatística": [
            "Tamanho (N)",
            f"Média da {VAR_INTERESSE}",
            f"Desvio-padrão da {VAR_INTERESSE}"
        ],
        "Valor": [
            f"{N:,}".replace(",", "."),
            fmt_br(pop_mean),
            fmt_br(pop_sd),
        ]
    })

    st.markdown("**Estatísticas do exemplo de população (variável de interesse)**")
    st.table(pop_stats_df)

# ---------------------------
# Gráfico da população
# ---------------------------
with col2:
    st.subheader(f"Distribuição da {VAR_INTERESSE} na população")
    fig_pop = plot_distribution(
        population[VAR_INTERESSE],
        f"Distribuição da {VAR_INTERESSE}",
        palette=laplace_colors,
        xlim=(x_min, x_max)
    )
    st.pyplot(fig_pop)

# ============================================================
# 2. CÁLCULO DO TAMANHO AMOSTRAL
# ============================================================

st.markdown("<br><br>", unsafe_allow_html=True)
st.header("2. Cálculo do tamanho amostral")

st.markdown("**Fórmula utilizada (estimativa da média populacional):**")
st.latex(r"n = \frac{z_{\alpha/2}^2 \cdot \sigma^2}{E^2}")

st.markdown("---")

E = st.slider(
    "Margem de erro (E) – mesma unidade da renda",
    min_value=100,
    max_value=3000,
    value=500,
    step=100
)

conf = st.selectbox(
    "Nível de confiança",
    [0.90, 0.95, 0.99],
    index=1
)

z_valor = norm.ppf((1 + conf) / 2)

n_calc, _ = sample_size_mean(
    E=E,
    sigma=pop_sd,
    conf=conf,
    N=N
)

n_final = min(n_calc, N)

st.success(f"Tamanho recomendado da amostra (n) = **{n_final:,}**".replace(",", "."))

# ============================================================
# 3. SELEÇÃO DA AMOSTRA
# ============================================================

st.markdown("<br><br>", unsafe_allow_html=True)
st.header("3. Seleção da amostra")

method = st.selectbox(
    "Método de amostragem",
    ["Aleatória simples", "Estratificada (sexo)"]
)

n = st.slider(
    "Tamanho da amostra (n)",
    min_value=10,
    max_value=N,
    value=min(n_final, N)
)

if method == "Aleatória simples":
    sample = sample_simple_random(population, n)
else:
    sample = sample_stratified(population, "sexo", n)

# ============================================================
# 4. GRÁFICO + ESTATÍSTICAS DA AMOSTRA
# ============================================================

st.subheader("Distribuição da amostra")

col_a1, col_a2 = st.columns(2)

with col_a1:
    fig_sample = plot_distribution(
        sample[VAR_INTERESSE],
        f"Distribuição da {VAR_INTERESSE} na amostra (n={n})",
        palette=laplace_colors,
        xlim=(x_min, x_max)  # 👉 MESMA ESCALA DA POPULAÇÃO
    )
    st.pyplot(fig_sample)

with col_a2:
    sample_mean = sample[VAR_INTERESSE].mean()
    sample_sd = sample[VAR_INTERESSE].std()

    sample_stats_df = pd.DataFrame({
        "Estatística": [
            "Tamanho (n)",
            f"Média da {VAR_INTERESSE}",
            f"Desvio-padrão da {VAR_INTERESSE}"
        ],
        "Valor": [
            f"{n:,}".replace(",", "."),
            fmt_br(sample_mean),
            fmt_br(sample_sd),
        ]
    })

    st.markdown("**Estatísticas da amostra (variável de interesse)**")
    st.table(sample_stats_df)

