import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image

from data import generate_population
from sampling_functions import (
    sample_simple_random,
    sample_stratified
)
from calculations import (
    sample_size_error,
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
# 1. GERAR POPULAÇÃO ARTIFICIAL
# ============================================================

st.markdown("<br><br>", unsafe_allow_html=True)
st.header("1. Gerar população artificial")

N = st.slider("Tamanho da população", 1000, 50000, 10000)
population = generate_population(N)

VAR_INTERESSE = "renda"

col1, col2 = st.columns(2)

# ---------------------------
# Tabela e estatísticas
# ---------------------------
with col1:
    st.subheader("Primeiras linhas da população")
    st.dataframe(population.head().reset_index(drop=True))

    pop_mean = population[VAR_INTERESSE].mean()
    pop_sd = population[VAR_INTERESSE].std()

    pop_stats_df = pd.DataFrame({
        "Estatística": ["Tamanho (N)", "Média", "Desvio-padrão"],
        "População": [
            f"{N:,}",
            f"{pop_mean:,.2f}",
            f"{pop_sd:,.2f}",
        ]
    })

    st.markdown("**Estatísticas da população (variável de interesse)**")
    st.table(pop_stats_df.reset_index(drop=True))

    # ---------------------------
    # Proporção de sexo na POP
    # ---------------------------
    pop_gender_prop = (
        population["sexo"]
        .value_counts(normalize=True)
        .rename("Proporção")
        .reset_index()
    )
    pop_gender_prop.columns = ["Sexo", "Proporção"]
    pop_gender_prop["Proporção"] = (pop_gender_prop["Proporção"] * 100).round(2)

    st.markdown("**Proporção de sexo na população**")
    st.table(pop_gender_prop.reset_index(drop=True))

# ---------------------------
# Gráfico população
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

col_f1, col_f2 = st.columns(2)
with col_f1:
    st.markdown("**Populações infinitas (AAS com reposição)**")
    st.latex(r"n_0 = \frac{1}{E_0^2}")
with col_f2:
    st.markdown("**Populações finitas (correção)**")
    st.latex(r"n = \frac{N \cdot n_0}{N + n_0}")

st.markdown("---")

E0_pct = st.slider("Erro amostral máximo permitido (E₀) em %", 1, 20, 5)
E0 = E0_pct / 100

usar_correcao = st.checkbox("Usar correção para população finita", value=True)

if usar_correcao:
    N_input = st.number_input(
        "Tamanho da população (N)",
        min_value=1,
        value=int(N),
        step=1
    )
    n_recomendado, n0 = sample_size_error(E0, N_input)
else:
    n_recomendado, n0 = sample_size_error(E0, None)

st.info(f"Primeira aproximação (n₀) = {n0:.1f}")
st.success(f"Tamanho recomendado (n) = **{n_recomendado}**")

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
# 3B. ESTATÍSTICAS DA AMOSTRA + GRÁFICO
# ============================================================

st.subheader("Distribuição da amostra (mesmo visual da população)")

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
        "Estatística": ["Tamanho (n)", "Média", "Desvio-padrão"],
        "Amostra": [
            len(sample),
            f"{sample_mean:,.2f}",
            f"{sample_sd:,.2f}",
        ]
    })

    st.markdown("**Estatísticas da amostra**")
    st.table(sample_stats_df.reset_index(drop=True))

# ============================================================
# 3C. PROPORÇÃO DE SEXO NA AMOSTRA (APENAS PARA ESTRATIFICADA)
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

