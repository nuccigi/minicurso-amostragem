import numpy as np

# ---- NOVO: fórmula do seu slide ----
def sample_size_error(E0, N=None):
    """
    E0: erro amostral máximo permitido (em escala decimal, ex: 0.05 para 5%)
    N:  tamanho da população (se None, assume população infinita)
    Retorna:
      - n (inteiro, arredondado para cima)
      - n0 (primeira aproximação, sem correção)
    """

    # primeira aproximação (população infinita)
    n0 = 1 / (E0 ** 2)

    if N is None:
        n = n0
    else:
        # correção para população finita
        n = (N * n0) / (N + n0)

    return int(np.ceil(n)), n0


# As funções abaixo podem ficar como já estavam:
from scipy.stats import norm

def standard_error(sd, n):
    return sd / np.sqrt(n)

def confidence_interval(mean, se, conf=0.95):
    z = norm.ppf((1 + conf) / 2)
    return mean - z*se, mean + z*se

