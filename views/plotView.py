import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
from production.codes.models import plotModel

def save_plot(train_plt_data, test_plt_data, symbols, episode, saved_path, dt_str, dpi=500, linewidth=0.2, title=None,
              figure_size=(28, 12), fontsize=9):
    """
    :param train_plt_data: {pd.Dataframe}
    :param test_plt_data: {pd.Dataframe}
    :param symbols: [str]
    :param saved_path: str, file to be saved
    :param episode: int
    :param dt_str: str
    :param dpi: resolution of image
    :param linewidth: line width in plots
    :param figure_size: tuple to indicate the size of figure
    :param show_inputs: Boolean
    """
    # prepare figure
    fig = plt.figure(figsize=figure_size, dpi=dpi)
    fig.suptitle(title, fontsize=fontsize*4)
    gs = fig.add_gridspec(plotModel.get_total_height(train_plt_data), 1)  # slice into grid with different size

    # graph list
    for i in range(len(train_plt_data)):
        df = pd.concat([train_plt_data[i]['df'], test_plt_data[i]['df']], axis=0)
        grid_step = train_plt_data[i]['height'] # height for each grid
        plt.subplot(gs[(i*grid_step):(i*grid_step+grid_step), :])
        for col_name in df:
            plt.plot(df.index, df[col_name].values, linewidth=linewidth, label=col_name)

        # text
        train_start_index, test_start_index = train_plt_data[i]['df'].index[0], test_plt_data[0]['df'].index[0]
        if train_plt_data[i]['text'] != None and test_plt_data[i]['text'] != None:
            plt.text(train_start_index, df.iloc[:,0].quantile(0.1), "Train \n" + train_plt_data[i]['text'], fontsize=fontsize)   # calculate the quantile 0.1 to get the y-position
            plt.text(test_start_index, df.iloc[:,0].quantile(0.1), "Test \n" + test_plt_data[i]['text'], fontsize=fontsize)     # calculate the quantile 0.1 to get the y-position

        # equation
        if train_plt_data[i]['equation'] != None and test_plt_data[i]['equation'] != None:
            plt.text(train_plt_data[i]['df'].index.mean(), df.iloc[:,0].quantile(0.1), train_plt_data[i]['equation'], fontsize=fontsize)
        plt.legend()

    full_path = saved_path + plotModel.get_coin_NN_plot_image_name(dt_str, symbols, episode)
    plt.savefig(full_path)  # save in higher resolution image
    plt.clf()                                                                              # clear the plot data

def density(ret_list, bins=50, color="darkblue", linewidth=1):
    """
    :param ret_list: list
    :param bins: int
    :param color: str
    :param linewidth: int
    :return: None
    """
    sns.distplot(ret_list, hist=True, kde=True,
                 bins=bins, color=color,
                 hist_kws={'edgecolor': 'black'},
                 kde_kws={'linewidth': linewidth})
    plt.show()