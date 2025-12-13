import numpy as np
from scipy.stats import norm

def sample_size_mean(E, sigma, conf=0.95, N=None):
    """
    E     : margem de erro (escala decimal, ex: 0.05)
    sigma : desvio-padrão populacional da variável de interesse
    conf  : nível de confiança (ex: 0.95, 0.99)
    N     : tamanho da população (None → população infinita)

    Retorna:
      - n (inteiro, arredondado para cima)
      - n0 (tamanho sem correção finita)
    """

    z = norm.ppf((1 + conf) / 2)

    # fórmula do slide
    n0 = (z**2 * sigma**2) / (E**2)

    if N is not None:
        n = (N * n0) / (N + n0)
    else:
        n = n0

    return int(np.ceil(n)), n0


def standard_error(sd, n):
    return sd / np.sqrt(n)


def confidence_interval(mean, se, conf=0.95):
    z = norm.ppf((1 + conf) / 2)
    return mean - z * se, mean + z * se

