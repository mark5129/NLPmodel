import matplotlib.pyplot as plt


def set_style():
    # Use a base style (optional)
    plt.style.use("ggplot")  # Or "seaborn-whitegrid", etc.
    
    # Set global rcParams
    plt.rcParams.update({
        "figure.figsize": (10, 6),
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        #"axes.grid": True,
        #"grid.alpha": 0.3,
        #"grid.linestyle": "--",
        "axes.facecolor": "white",  # Set axes background to white
        "axes.edgecolor": "black",  # Set axes edge color
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "font.family": "serif",
        "savefig.dpi": 300,
        "figure.autolayout": True
    })