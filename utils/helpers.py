import matplotlib.pyplot as plt
import seaborn as sns


def set_plot_style():
    """
    Applies a consistent and clean visual style for matplotlib and seaborn plots.

    This function configures:
    - The seaborn theme (`whitegrid`) with a muted color palette
    - Matplotlib default style and custom `rcParams` for:
        - Grid appearance and transparency
        - Axis spines removal (cleaner look)
        - Font sizes for titles, labels, and ticks
        - Figure size and resolution
        - Axis label padding
        - Tick visibility and alignment

    Recommended to call at the beginning of a script or notebook 
    to standardize all plots in your analysis.

    Note:
        You can uncomment `"figure.dpi": 300` if you want to change screen rendering resolution.
    """
    sns.set_theme(style="whitegrid", palette="muted")

    plt.style.use("default")
    plt.rcParams.update(
        {
            "axes.edgecolor": "white",
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.color": "grey",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.spines.bottom": False,
            "axes.titlesize": 10,
            "axes.labelsize": 10,
            "axes.labelpad": 15,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            # "figure.dpi": 300,
            "savefig.dpi": 300,
            "figure.figsize": (12, 6),
            "xtick.bottom": False,
            "ytick.left": False,
        }
    )