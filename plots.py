import matplotlib.pyplot as plt
import seaborn as sns

DEFAULT_PALETTE = ["#0f2f56", "#22576d", "#296872", "#045243"]

def plot_distribution(data, title="", palette=None, xlim=None):

    if palette is None:
        palette = DEFAULT_PALETTE

    color = palette[0]  # cor principal Laplace

    fig, ax = plt.subplots(figsize=(6, 4))

    sns.histplot(
        data,
        kde=True,
        ax=ax,
        color=color,
        edgecolor="white",
        alpha=0.85
    )

    ax.set_title(title, fontsize=12, color=palette[0])
    ax.set_xlabel("")
    ax.set_ylabel("Frequência")

    # AQUI: eixo sincronizado
    if xlim is not None:
        ax.set_xlim(xlim)

    sns.despine()

    return fig
