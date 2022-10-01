import MetaTrader5 as mt5

def print_terminal_info():
    # request connection status and parameters
    print(mt5.terminal_info())
    # request account info
    print(mt5.account_info())
    # get loader on MetaTrader 5 version
    print(mt5.version())