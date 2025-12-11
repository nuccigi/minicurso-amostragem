import numpy as np
import pandas as pd

def sample_simple_random(df, n, seed=42):
    return df.sample(n=n, random_state=seed)

def sample_systematic(df, n, seed=42):
    np.random.seed(seed)
    N = len(df)
    k = N // n
    start = np.random.randint(0, k)
    indices = np.arange(start, start + k*n, k)
    indices = indices[indices < N]
    return df.iloc[indices]

def sample_stratified(df, strata_col, n, seed=42):
    np.random.seed(seed)
    strata = df[strata_col].unique()
    out = []
    for s in strata:
        group = df[df[strata_col] == s]
        n_s = int((len(group) / len(df)) * n)
        out.append(group.sample(n=n_s, random_state=seed))
    return pd.concat(out)
