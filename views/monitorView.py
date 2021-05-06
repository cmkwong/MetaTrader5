from production.codes.models import monitorModel, mt5Model
import matplotlib.pyplot as plt

def get_plot_title(start, end, timeframe_str):
    start_str = mt5Model.get_time_string(start)
    end_str = mt5Model.get_time_string(end)
    title = "{} : {}, {}".format(start_str, end_str, timeframe_str)
    return title

def get_plot_image_name(dt_str, symbols, episode):
    symbols_str = ''
    for symbol in symbols:
        symbols_str += '_' + symbol
    name = "{}-{}-episode-{}.jpg".format(dt_str, episode, symbols_str)
    return name

def save_plot(df_plt, symbols, episode, saved_path, dt_str, splite_index, dpi, linewidth, title=None, show_inputs=True):
    """
    :param df_plt: dataframe, 3 parts: inputs column, target column, predicted column
    :param symbols: [str]
    :param saved_path: str, file to be saved
    :param episode: int
    :param dt_str: str
    :param dpi: resolution of image
    :param linewidth: line width in plots
    """
    target_symbol = symbols[-1] # get the name
    full_path = saved_path + get_plot_image_name(dt_str, symbols, episode)
    if show_inputs:
        for symbol in symbols[:-1]:
            df_plt[symbol].plot(kind='line', style="k--", linewidth=linewidth, legend=True)
    df_plt[target_symbol].plot(kind='line', color='blue', linewidth=linewidth, legend=True)
    df_plt['predict'].plot(kind='line', color='red', linewidth=linewidth, legend=True)
    df_plt['spread'].plot(kind='line', color='darkorange', linewidth=linewidth, legend=True)    # plot spread
    plt.axhline(y=0, linewidth=0.1, linestyle="--", color="dimgray")                            # y=0 reference line
    plt.axvline(x=splite_index, linewidth=0.1, linestyle="--", color="darkgrey")            # testing start point
    plt.title(title)
    plt.savefig(full_path, dpi=dpi)                                                             # save in higher resolution image
    plt.clf()                                                                                   # clear the plot data

def get_price_plot(train_prices_matrix, test_prices_matrix, model, episode, seq_len, symbols, saved_path, dt_str, dpi=500, linewidth=0.2):
    # data prepare
    train_plt_data = monitorModel.get_plotting_data(train_prices_matrix, model, seq_len)
    test_plt_data = monitorModel.get_plotting_data(test_prices_matrix, model, seq_len)
    # combine into df
    df_plt = monitorModel.get_plotting_df(train_plt_data, test_plt_data, symbols)
    # plot graph - prices
    save_plot(df_plt, symbols, episode, saved_path, dt_str, len(train_plt_data['inputs']), dpi=dpi, linewidth=linewidth)

def loss_status(writer, loss, episode, mode='train'):
    """
    :param writer: SummaryWriter from pyTorch
    :param loss: float
    :param episode: int
    :param mode: string "train" / "test"
    """
    writer.add_scalar("{}-episode_loss".format(mode), loss, episode)
    print("{}. {} loss: {:.6f}".format(episode, mode, loss))