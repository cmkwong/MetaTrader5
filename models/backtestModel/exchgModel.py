from production.codes.models.backtestModel import indexModel, priceModel

def modify_exchg_q2d(exchg_q2d, signal):
    """
    note 79a
    :param exchg_q2d: pd.DataFrame
    :param signal: pd.Series
    :return:
    """
    exchg_q2d_copy = exchg_q2d.copy()
    start_index, end_index = indexModel.get_action_start_end_index(signal.reset_index(drop=True))
    for s, e in zip(start_index, end_index):
        exchg_q2d_copy.iloc[s:e,:] = exchg_q2d.iloc[s,:].values
    return exchg_q2d_copy

def get_exchange_symbols(symbols, all_symbols_info, deposit_currency='USD', exchg_type='q2d'):
    """
    Find all the currency pair related to and required currency and deposit symbol
    :param symbols: [str] : ["AUDJPY", "AUDUSD", "CADJPY", "EURUSD", "NZDUSD", "USDCAD"]
    :param all_symbols_info: dict with nametuple
    :param deposit_currency: str: USD/GBP/EUR, main currency for deposit
    :param exchg_type: str, 'q2d' = quote to deposit OR 'b2d' = base to deposit
    :return: [str], get required exchange symbol in list: ['USDJPY', 'AUDUSD', 'USDJPY', 'EURUSD', 'NZDUSD', 'USDCAD']
    """
    symbol_names = list(all_symbols_info.keys())
    exchange_symbols = []
    target_symbol = None
    for symbol in symbols:
        if exchg_type == 'b2d': target_symbol = symbol[:3]
        elif exchg_type == 'q2d': target_symbol = symbol[3:]
        if target_symbol != deposit_currency:  # if the symbol not relative to required deposit currency
            test_symbol_1 = target_symbol + deposit_currency
            test_symbol_2 = deposit_currency + target_symbol
            if test_symbol_1 in symbol_names:
                exchange_symbols.append(test_symbol_1)
                continue
            elif test_symbol_2 in symbol_names:
                exchange_symbols.append(test_symbol_2)
                continue
            else: # if not found the relative pair with respect to deposit currency, raise the error
                raise Exception("{} has no relative currency with respect to deposit {}.".format(target_symbol, deposit_currency))
        else: # if the symbol already relative to deposit currency
            exchange_symbols.append(symbol)
    return exchange_symbols

def modify_exchange_rate(symbols, exchange_symbols, exchange_rate_df, deposit_currency, exchg_type):
    """
    :param symbols:             ['AUDJPY', 'AUDUSD', 'CADJPY', 'EURUSD', 'NZDUSD', 'USDCAD']
    :param exchange_symbols:    ['USDJPY', 'AUDUSD', 'USDJPY', 'EURUSD', 'NZDUSD', 'USDCAD'] all is related to deposit currency
    :param exchange_rate_df: pd.DataFrame, the price from excahnge_symbols
    :param deposit_currency: "USD" / "GBP" / "EUR"
    :param exchg_type: str, 'q2d' = quote to deposit OR 'b2d' = base to deposit
    :return: pd.DataFrame with cols name: ['JPYUSD', 'USD', 'JPYUSD', 'USD', 'USD', 'CADUSD']
    """
    symbol_new_names = []
    for i, exch_symbol in enumerate(exchange_symbols):
        base, quote = exch_symbol[:3], exch_symbol[3:]
        if exchg_type == 'q2d': # see note 38a
            if quote == deposit_currency:
                if symbols[i] != exch_symbol:
                    symbol_new_names.append("{}to{}".format(exch_symbol[:3], exch_symbol[3:]))
                elif symbols[i] == exch_symbol:
                    symbol_new_names.append(deposit_currency)
                    exchange_rate_df.iloc[:, i] = 1.0
            elif base == deposit_currency:
                symbol_new_names.append("{}to{}".format(exch_symbol[3:], exch_symbol[:3]))
                exchange_rate_df.iloc[:, i] = 1 / exchange_rate_df.iloc[:, i].values
        elif exchg_type == 'b2d':
            if base == deposit_currency:
                if symbols[i] != exch_symbol:
                    symbol_new_names.append("{}to{}".format(exch_symbol[3:], exch_symbol[:3]))
                    exchange_rate_df.iloc[:, i] = 1 / exchange_rate_df.iloc[:, i].values
                elif symbols[i] == exch_symbol:
                    symbol_new_names.append(deposit_currency)
                    exchange_rate_df.iloc[:, i] = 1.0
            elif quote == deposit_currency:
                symbol_new_names.append("{}to{}".format(exch_symbol[:3], exch_symbol[3:]))

    return exchange_rate_df, symbol_new_names

def get_exchange_df(symbols, all_symbols_info, deposit_currency, timeframe, timezone, ohlc, count, exchg_type, col_names, start=None, end=None):
    """
    :param col_names:
    :param symbols: [str]
    :param all_symbols_info: mt5.symbols_info object
    :param deposit_currency: str, USD / GBP / EUR
    :param timeframe: mt5.timeFrame
    :param timezone: str "Hongkong"
    :param start: (2010,1,1,0,0)
    :param end: (2020,1,1,0,0)
    :param ohlc: 'str', eg: '1000'
    :param count: int
    :param exchg_type: q2d = quote exchange to deposit, b2d = base exchange to deposit
    :param col_names: list, the name assigned to column names
    :return: pd.DataFrame
    """
    exchange_symbols = get_exchange_symbols(symbols, all_symbols_info, deposit_currency, exchg_type=exchg_type)
    exchange_rate_df = priceModel._get_prices_df(exchange_symbols, timeframe, timezone, start, end, ohlc=ohlc, count=count)  # just need the open price
    exchange_rate_df, modified_names = modify_exchange_rate(symbols, exchange_symbols, exchange_rate_df,
                                                                    deposit_currency, exchg_type=exchg_type)
    exchange_rate_df.columns = col_names  # assign temp name
    return exchange_rate_df, modified_names

