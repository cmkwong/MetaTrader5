import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
from production.codes.models import plotModel

def save_plot(train_plt_df, test_plt_df, symbols, episode, saved_path, dt_str, dpi=500, linewidth=0.2, title=None,
              figure_size=(28, 12), show_inputs=False):
    """
    :param train_plt_df: pd.Dataframe
    :param test_plt_df: pd.Dataframe
    :param symbols: [str]
    :param saved_path: str, file to be saved
    :param episode: int
    :param dt_str: str
    :param dpi: resolution of image
    :param linewidth: line width in plots
    :param figure_size: tuple to indicate the size of figure
    """
    # concat the plotting dataframe
    plt_df = pd.concat((train_plt_df,test_plt_df))
    time_index = plt_df.index
    split_index = test_plt_df.index[0]

    # prepare figure
    fig = plt.figure(figsize=figure_size, dpi=dpi)
    gs = fig.add_gridspec(10,1) # slice into grid with different size

    # switch to index of plot 1 in 2x1 plot
    plt.subplot(gs[0:7,:])
    plt.title(title)
    target_symbol = symbols[-1] # get the name
    full_path = saved_path + plotModel.get_plot_image_name(dt_str, symbols, episode)
    if show_inputs:
        for symbol in symbols[:-1]:
            plt.plot(time_index, plt_df[symbol].values, linestyle="k--", linewidth=linewidth)
    plt.plot(time_index, plt_df[target_symbol].values, color='blue', linewidth=linewidth)
    plt.plot(time_index, plt_df['predict'].values, color='red', linewidth=linewidth)
    plt.plot(time_index, plt_df['spread'].values, color='darkorange', linewidth=linewidth)    # plot spread
    plt.axhline(y=0, linewidth=0.1, linestyle="--", color="dimgray")  # y=0 reference line
    plt.axvline(x=split_index, linewidth=0.1, linestyle="--", color="darkgrey")  # testing start index

    # switch to index of plot 2 in 2x1 plot
    plt.subplot(gs[8:10,:])
    plt.plot(time_index, plt_df['z_score'].values, color='darkorange', linewidth=linewidth/2)
    plt.axhline(y=0, linewidth=0.1, linestyle="--", color="dimgray")                            # y=0 reference line
    plt.axvline(x=split_index, linewidth=0.1, linestyle="--", color="darkgrey")                # testing start index

    plt.savefig(full_path)                                                                      # save in higher resolution image
    plt.clf()

def _save_plot_simple_subplot(plt_df, symbols, episode, saved_path, dt_str, splite_index, dpi, linewidth, title=None, show_inputs=True):
    """
    :param plt_df: dataframe, 3 parts: inputs column, target column, predicted column
    :param symbols: [str]
    :param saved_path: str, file to be saved
    :param episode: int
    :param dt_str: str
    :param dpi: resolution of image
    :param linewidth: line width in plots
    """
    plt.figure(figsize=(14, 6), dpi=dpi)

    # switch to index of plot 1 in 2x1 plot
    plt.subplot(211)
    plt.title(title)
    target_symbol = symbols[-1] # get the name
    full_path = saved_path + plotModel.get_plot_image_name(dt_str, symbols, episode)
    if show_inputs:
        for symbol in symbols[:-1]:
            plt.plot(plt_df[symbol].index, plt_df[symbol].values, linestyle="k--", linewidth=linewidth)
    plt.plot(plt_df[target_symbol].index, plt_df[target_symbol].values, color='blue', linewidth=linewidth)
    plt.plot(plt_df['predict'].index, plt_df['predict'].values, color='red', linewidth=linewidth)
    plt.plot(plt_df['spread'].index, plt_df['spread'].values, color='darkorange', linewidth=linewidth)    # plot spread
    plt.axhline(y=0, linewidth=0.1, linestyle="--", color="dimgray")  # y=0 reference line
    plt.axvline(x=splite_index, linewidth=0.1, linestyle="--", color="darkgrey")  # testing start index

    # switch to index of plot 2 in 2x1 plot
    plt.subplot(212)
    plt.plot(plt_df['z_score'].index, plt_df['z_score'].values, color='darkorange', linewidth=linewidth/2)
    plt.axhline(y=0, linewidth=0.1, linestyle="--", color="dimgray")                            # y=0 reference line
    plt.axvline(x=splite_index, linewidth=0.1, linestyle="--", color="darkgrey")                # testing start index

    plt.savefig(full_path)                                                                      # save in higher resolution image
    plt.clf()                                                                                   # clear the plot data

def get_price_plot(train_prices_df, test_prices_df, model, episode, seq_len, symbols, saved_path, dt_str, dpi=500, linewidth=0.2):
    # data prepare
    train_plt_df = plotModel.get_plotting_data(train_prices_df, model, seq_len)
    test_plt_df = plotModel.get_plotting_data(test_prices_df, model, seq_len)
    # plot graph - prices
    save_plot(train_plt_df, test_plt_df, symbols, episode, saved_path, dt_str, dpi=dpi, linewidth=linewidth)

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