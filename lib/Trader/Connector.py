import MetaTrader5 as mt5

class MetaTrader_Connector:

    def __init__(self):
        self.connect_server()

    def connect_server(self):
        # connect to MetaTrader 5
        if not mt5.initialize():
            print("initialize() failed")
            mt5.shutdown()
        else:
            print("MetaTrader Connected")

    def print_terminal_info(self):
        # request connection status and parameters
        print(mt5.terminal_info())
        # request account info
        print(mt5.account_info())
        # get data on MetaTrader 5 version
        print(mt5.version())