import matplotlib.pyplot as plt
import seaborn as sns

def density(data, bins=50, color="darkblue", linewidth=1):
    """
    :param data: list
    :param bins: int
    :param color: str
    :param linewidth: int
    :return: None
    """
    sns.distplot(data, hist=True, kde=True,
                 bins=bins, color=color,
                 hist_kws={'edgecolor': 'black'},
                 kde_kws={'linewidth': linewidth})

    plt.show()