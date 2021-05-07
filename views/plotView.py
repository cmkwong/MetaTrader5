import seaborn as sns
from matplotlib import pyplot as plt
from production.codes.models import plotModel

def save_plot(df_plt, symbols, episode, saved_path, dt_str, splite_index, dpi=500, linewidth=0.2, title=None,
              figure_size=(28, 12), show_inputs=True):
    """
    :param figure_size:
    :param df_plt: dataframe, 3 parts: inputs column, target column, predicted column
    :param symbols: [str]
    :param saved_path: str, file to be saved
    :param episode: int
    :param dt_str: str
    :param dpi: resolution of image
    :param linewidth: line width in plots
    """
    fig = plt.figure(figsize=figure_size, dpi=dpi)
    gs = fig.add_gridspec(10,1) # slice the grid into tenth parts

    # switch to index of plot 1 in 2x1 plot
    plt.subplot(gs[0:7,:])
    plt.title(title)
    target_symbol = symbols[-1] # get the name
    full_path = saved_path + plotModel.get_plot_image_name(dt_str, symbols, episode)
    if show_inputs:
        for symbol in symbols[:-1]:
            plt.plot(df_plt[symbol].index, df_plt[symbol].values, style="k--", linewidth=linewidth)
    plt.plot(df_plt[target_symbol].index, df_plt[target_symbol].values, color='blue', linewidth=linewidth)
    plt.plot(df_plt['predict'].index, df_plt['predict'].values, color='red', linewidth=linewidth)
    plt.plot(df_plt['spread'].index, df_plt['spread'].values, color='darkorange', linewidth=linewidth)    # plot spread
    plt.axhline(y=0, linewidth=0.1, linestyle="--", color="dimgray")  # y=0 reference line
    plt.axvline(x=splite_index, linewidth=0.1, linestyle="--", color="darkgrey")  # testing start index

    # switch to index of plot 2 in 2x1 plot
    plt.subplot(gs[8:10,:])
    plt.plot(df_plt['z_score'].index, df_plt['z_score'].values, color='darkorange', linewidth=linewidth/2)
    plt.axhline(y=0, linewidth=0.1, linestyle="--", color="dimgray")                            # y=0 reference line
    plt.axvline(x=splite_index, linewidth=0.1, linestyle="--", color="darkgrey")                # testing start index

    plt.savefig(full_path)                                                                      # save in higher resolution image
    plt.clf()

def _save_plot_simple_subplot(df_plt, symbols, episode, saved_path, dt_str, splite_index, dpi, linewidth, title=None, show_inputs=True):
    """
    :param df_plt: dataframe, 3 parts: inputs column, target column, predicted column
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
            plt.plot(df_plt[symbol].index, df_plt[symbol].values, style="k--", linewidth=linewidth)
    plt.plot(df_plt[target_symbol].index, df_plt[target_symbol].values, color='blue', linewidth=linewidth)
    plt.plot(df_plt['predict'].index, df_plt['predict'].values, color='red', linewidth=linewidth)
    plt.plot(df_plt['spread'].index, df_plt['spread'].values, color='darkorange', linewidth=linewidth)    # plot spread
    plt.axhline(y=0, linewidth=0.1, linestyle="--", color="dimgray")  # y=0 reference line
    plt.axvline(x=splite_index, linewidth=0.1, linestyle="--", color="darkgrey")  # testing start index

    # switch to index of plot 2 in 2x1 plot
    plt.subplot(212)
    plt.plot(df_plt['z_score'].index, df_plt['z_score'].values, color='darkorange', linewidth=linewidth/2)
    plt.axhline(y=0, linewidth=0.1, linestyle="--", color="dimgray")                            # y=0 reference line
    plt.axvline(x=splite_index, linewidth=0.1, linestyle="--", color="darkgrey")                # testing start index

    plt.savefig(full_path)                                                                      # save in higher resolution image
    plt.clf()                                                                                   # clear the plot data

def get_price_plot(train_prices_matrix, test_prices_matrix, model, episode, seq_len, symbols, saved_path, dt_str, dpi=500, linewidth=0.2):
    # data prepare
    train_plt_data = plotModel.get_plotting_data(train_prices_matrix, model, seq_len)
    test_plt_data = plotModel.get_plotting_data(test_prices_matrix, model, seq_len)
    # combine into df
    df_plt = plotModel.concatenate_plotting_df(train_plt_data, test_plt_data, symbols)
    # plot graph - prices
    save_plot(df_plt, symbols, episode, saved_path, dt_str, len(train_plt_data['inputs']), dpi=dpi, linewidth=linewidth)

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