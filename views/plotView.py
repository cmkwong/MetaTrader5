import os
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
from production.codes.backtest import plotPre

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

def save_plot(train_plt_data, test_plt_data, symbols, episode, saved_path, dt_str, dpi=500, linewidth=0.2, title=None,
              figure_size=(28, 12), fontsize=9, bins=100, setting='', hist_range=None):
    """
    :param setting:
    :param bins:
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
    plt.figtext(0.1,0.9, setting, fontsize=fontsize*2)
    gs = fig.add_gridspec(plotPre.get_total_height(train_plt_data), 1)  # slice into grid with different size
    # for histogram range
    if hist_range != None: hist_range = (hist_range[0]+1, hist_range[1]-1) # exclude the range, see note (51a)

    # graph list
    for i in range(len(train_plt_data)):

        # subplot setup
        grid_step = train_plt_data[i]['height']  # height for each grid
        plt.subplot(gs[(i * grid_step):(i * grid_step + grid_step), :])

        # dataframe
        if test_plt_data == None:
            if type(train_plt_data[i]['df']) == pd.DataFrame:
                df = train_plt_data[i]['df']
                for col_name in df.columns:
                    plt.plot(df.index, df[col_name].values, linewidth=linewidth, label=col_name)
        else:
            if type(train_plt_data[i]['df']) == pd.DataFrame and type(test_plt_data[i]['df']) == pd.DataFrame:
                df = pd.concat([train_plt_data[i]['df'], test_plt_data[i]['df']], axis=0)
                for col_name in df.columns:
                    plt.plot(df.index, df[col_name].values, linewidth=linewidth, label=col_name)

        # histogram (pd.Series)
        if test_plt_data == None:
            if type(train_plt_data[i]['hist']) == pd.Series:
                plt.hist(train_plt_data[i]['hist'], bins=bins, label="{} {}".format("Train", train_plt_data[i]['hist'].name), range=hist_range)
        else:
            if type(train_plt_data[i]['hist']) == pd.Series and type(test_plt_data[i]['hist']) == pd.Series:
                plt.hist(train_plt_data[i]['hist'], bins=bins, label="{} {}".format("Train", train_plt_data[i]['hist'].name), range=hist_range)
                plt.hist(test_plt_data[i]['hist'], bins=bins, label="{} {}".format("Test", test_plt_data[i]['hist'].name), range=hist_range)

        # text
        if test_plt_data == None:
            if type(train_plt_data[i]['text']) == str:
                train_start_index = train_plt_data[i]['df'].index[0]
                plt.text(train_start_index, df.iloc[:, 0].quantile(0.01), "Train \n" + train_plt_data[i]['text'], fontsize=fontsize * 0.7)
        else:
            if type(train_plt_data[i]['text']) == str and type(test_plt_data[i]['text']) == str:
                train_start_index, test_start_index = train_plt_data[i]['df'].index[0], test_plt_data[0]['df'].index[0]
                plt.text(train_start_index, df.iloc[:,0].quantile(0.01), "Train \n" + train_plt_data[i]['text'], fontsize=fontsize*0.7)   # calculate the quantile 0.1 to get the y-position
                plt.text(test_start_index, df.iloc[:,0].quantile(0.01), "Test \n" + test_plt_data[i]['text'], fontsize=fontsize*0.7)     # calculate the quantile 0.1 to get the y-position

        # equation
        if type(train_plt_data[i]['equation']) == str:
            plt.text(train_plt_data[i]['df'].index.mean(), df.iloc[:,0].quantile(0.1), train_plt_data[i]['equation'], fontsize=fontsize*2)

        plt.legend()

    full_path = os.path.join(saved_path, plotPre.get_coin_NN_plot_image_name(dt_str, symbols, episode))
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