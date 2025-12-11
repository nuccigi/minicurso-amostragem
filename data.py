import numpy as np
import pandas as pd

def generate_population(N=10000, seed=42):
    np.random.seed(seed)

    population = pd.DataFrame({
        "idade": np.random.normal(40, 12, N).astype(int),
        "renda": np.random.lognormal(mean=8.5, sigma=0.5, size=N),
        "sexo": np.random.choice(["F", "M"], size=N),
        "regiao": np.random.choice(["Norte", "Sul", "Leste", "Oeste"], size=N)
    })
    return population
