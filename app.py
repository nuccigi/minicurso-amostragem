import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import norm

from data import generate_population
from sampling_functions import (
    sample_simple_random,
    sample_stratified
)
from calculations import (
    sample_size_mean,
    standard_error,
    confidence_interval
)
from plots import plot_distribution

# ================================
# CONFIGURAÇÃO DO APP
# ================================

st.set_page_config(
    page_title="Simulador de Amostragem – Laplace",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# LOGO NO CANTO SUPERIOR DIREITO
col_logo1, col_logo2 = st.columns([6, 1])
with col_logo2:
    st.image("assets/logo.png", width=120)

st.title("Simulador de Amostragem – Minicurso Laplace")
st.write("Explore conceitos de tamanho de amostra, métodos de seleção e erros amostrais.")

# Paleta Laplace
laplace_colors = ["#0f2f56", "#22576d", "#296872", "#045243"]

# ============================================================
# 1. EXEMPLO DE POPULAÇÃO
# ============================================================

st.markdown("<br><br>", unsafe_allow_html=True)
st.header("1. Exemplo de população")

N = st.slider("Tamanho da população", 1000, 50000, 10000)
population = generate_population(N)

VAR_INTERESSE = "renda"

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
        "População": [
            f"{N:,}",
            f"{pop_mean:,.2f}",
            f"{pop_sd:,.2f}",
        ]
    })

    st.markdown("**Estatísticas do exemplo de população (variável de interesse)**")
    st.table(pop_stats_df.reset_index(drop=True))

    # ---------------------------
    # Proporção de sexo na população
    # ---------------------------
    pop_gender_prop = (
        population["sexo"]
        .value_counts(normalize=True)
        .rename("Proporção")
        .reset_index()
    )
    pop_gender_prop.columns = ["Sexo", "Proporção"]
    pop_gender_prop["Proporção"] = (pop_gender_prop["Proporção"] * 100).round(2)

    st.markdown("**Proporção de sexo no exemplo de população**")
    st.table(pop_gender_prop.reset_index(drop=True))

# ---------------------------
# Gráfico da população
# ---------------------------
with col2:
    st.subheader(f"Distribuição da {VAR_INTERESSE} na população")
    fig = plot_distribution(
        population[VAR_INTERESSE],
        f"Distribuição da {VAR_INTERESSE}",
        palette=laplace_colors
    )
    st.pyplot(fig)

# ============================================================
# 2. CÁLCULO DO TAMANHO AMOSTRAL
# ============================================================

st.markdown("<br><br>", unsafe_allow_html=True)
st.header("2. Cálculo do tamanho amostral")

st.markdown("**Fórmula utilizada (estimativa da média populacional):**")
st.latex(r"n = \frac{z_{\alpha/2}^2 \cdot \sigma^2}{E^2}")

st.markdown("---")

E_pct = st.slider("Margem de erro (E) em %", 1, 20, 5)
E = E_pct / 100

conf = st.selectbox(
    "Nível de confiança",
    [0.90, 0.95, 0.99],
    index=1
)

usar_correcao = st.checkbox("Usar correção para população finita", value=True)

if usar_correcao:
    N_input = st.number_input(
        "Tamanho da população (N)",
        min_value=1,
        value=int(N),
        step=1
    )
else:
    N_input = None

n_recomendado, n0 = sample_size_mean(
    E=E,
    sigma=pop_sd,
    conf=conf,
    N=N_input
)

z_valor = norm.ppf((1 + conf) / 2)

st.info(
    f"""
    **Parâmetros utilizados:**
    - Variável de interesse: {VAR_INTERESSE}
    - Desvio-padrão populacional (σ): {pop_sd:.2f}
    - Nível de confiança: {int(conf*100)}%
    - Valor crítico z: {z_valor:.2f}
    - Margem de erro (E): {E:.3f}
    """
)

st.success(f"Tamanho recomendado da amostra (n) = **{n_recomendado}**")

# ============================================================
# 3. SELEÇÃO DA AMOSTRA
# ============================================================

st.markdown("<br><br>", unsafe_allow_html=True)
st.header("3. Seleção da amostra")

method = st.selectbox(
    "Método de amostragem",
    ["Aleatória simples", "Estratificada (sexo)"]
)

n_max = min(5000, N)
n = st.slider(
    "Tamanho da amostra (n)",
    min_value=10,
    max_value=int(n_max),
    value=int(min(n_recomendado, n_max))
)

if method == "Aleatória simples":
    sample = sample_simple_random(population, n)

elif method == "Estratificada (sexo)":
    sample = sample_stratified(population, "sexo", n)

# ============================================================
# 4. ESTATÍSTICAS DA AMOSTRA + GRÁFICO
# ============================================================

st.subheader("Distribuição da amostra")

col_a1, col_a2 = st.columns(2)

with col_a1:
    fig_sample = plot_distribution(
        sample[VAR_INTERESSE],
        f"Distribuição da {VAR_INTERESSE} na amostra (n={len(sample)})",
        palette=laplace_colors
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
        "Amostra": [
            len(sample),
            f"{sample_mean:,.2f}",
            f"{sample_sd:,.2f}",
        ]
    })

    st.markdown("**Estatísticas da amostra (variável de interesse)**")
    st.table(sample_stats_df.reset_index(drop=True))

# ============================================================
# 5. PROPORÇÃO DE SEXO NA AMOSTRA (ESTRATIFICADA)
# ============================================================

if method == "Estratificada (sexo)":
    st.markdown("### Proporção de sexo na amostra (estratificada)")

    sample_gender_prop = (
        sample["sexo"]
        .value_counts(normalize=True)
        .rename("Proporção")
        .reset_index()
    )
    sample_gender_prop.columns = ["Sexo", "Proporção"]
    sample_gender_prop["Proporção"] = (sample_gender_prop["Proporção"] * 100).round(2)

    st.table(sample_gender_prop.reset_index(drop=True))

