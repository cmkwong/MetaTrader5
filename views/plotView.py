import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
from production.codes.models import plotModel, coinNNModel

def save_plot(train_plt_data, test_plt_data, symbols, episode, saved_path, dt_str, dpi=500, linewidth=0.2, title=None,
              figure_size=(28, 12)):
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
    fig.suptitle(title, fontsize=20)
    gs = fig.add_gridspec(10, 1)  # slice into grid with different size

    # graph 1
    for i in range(len(train_plt_data)):
        df = pd.concat([train_plt_data[i]['df'], test_plt_data[i]['df']], axis=0)
        plt.subplot(gs[(i*2):(i*2+2), :])
        for col_name in df:
            plt.plot(df.index, df[col_name].values, linewidth=linewidth, label=col_name)

        # text
        train_start_index, test_start_index = train_plt_data[i]['df'].index[0], test_plt_data[0]['df'].index[0]
        if train_plt_data[i]['text'] != None and test_plt_data[i]['text'] != None:
            plt.text(train_start_index, df.iloc[:,0].quantile(0.1), "Train \n" + train_plt_data[i]['text'], fontsize=9)   # calculate the quantile 0.1 to get the y-position
            plt.text(test_start_index, df.iloc[:,0].quantile(0.1), "Test \n" + test_plt_data[i]['text'], fontsize=9)     # calculate the quantile 0.1 to get the y-position

        # equation
        if train_plt_data[i]['equation'] != None and test_plt_data[i]['equation'] != None:
            plt.text(train_plt_data[i]['df'].index.mean(), df.iloc[:,0].quantile(0.1), train_plt_data[i]['equation'], fontsize=9)
        plt.legend()

    full_path = saved_path + plotModel.get_coin_plot_image_name(dt_str, symbols, episode)
    plt.savefig(full_path)  # save in higher resolution image
    plt.clf()

def save_plot2(train_plt_df, test_plt_df, symbols, episode, saved_path, dt_str, dpi=500, linewidth=0.2, title=None,
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
    :param show_inputs: Boolean
    """
    # concat the plotting dataframe
    plt_df = pd.concat((train_plt_df, test_plt_df))
    time_index = plt_df.index
    train_start_index = train_plt_df.index[0]
    test_start_index = test_plt_df.index[0]

    # prepare figure
    fig = plt.figure(figsize=figure_size, dpi=dpi)
    gs = fig.add_gridspec(10,1) # slice into grid with different size

    # switch to index of plot 1 in 2x1 plot
    plt.subplot(gs[0:7,:])
    plt.title(title)
    if show_inputs:
        for symbol in symbols[:-1]:
            plt.plot(time_index, plt_df[symbol].values, linestyle="k--", linewidth=linewidth, label=symbol)
    plt.plot(time_index, plt_df[symbols[-1]].values, color='blue', linewidth=linewidth, label=symbols[-1])
    plt.plot(time_index, plt_df['predict'].values, color='red', linewidth=linewidth, label='predict')
    plt.plot(time_index, plt_df['spread'].values, color='darkorange', linewidth=linewidth, label='spread')      # plot spread
    plt.axhline(y=0, linewidth=0.1, linestyle="--", color="dimgray")                                            # y=0 reference line
    plt.axvline(x=test_start_index, linewidth=0.1, linestyle="--", color="darkgrey")                            # testing start index
    plt.legend()

    # add ADF test result
    adf_result_text = plotModel.get_ADF_text_result(train_plt_df['spread'].values)
    plt.text(train_start_index, 0.5, "Train \n" + adf_result_text, fontsize=12)
    adf_result_text = plotModel.get_ADF_text_result(test_plt_df['spread'].values)
    plt.text(test_start_index,0.5, "Test \n" + adf_result_text, fontsize=12)

    # switch to index of plot 2 in 2x1 plot
    plt.subplot(gs[8:10,:])
    plt.plot(time_index, plt_df['z_score'].values, color='darkorange', linewidth=linewidth/2, label='z_score')
    plt.axhline(y=0, linewidth=0.1, linestyle="--", color="dimgray")                                # y=0 reference line
    plt.axvline(x=test_start_index, linewidth=0.1, linestyle="--", color="darkgrey")                # testing start index
    plt.legend()

    # save plot
    full_path = saved_path + plotModel.get_coin_plot_image_name(dt_str, symbols, episode)
    plt.savefig(full_path)                                                                      # save in higher resolution image
    plt.clf()

def save_plot_simple_subplot__discard(plt_df, symbols, episode, saved_path, dt_str, splite_index, dpi, linewidth, title=None, show_inputs=True):
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
    full_path = saved_path + plotModel.get_coin_plot_image_name(dt_str, symbols, episode)
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