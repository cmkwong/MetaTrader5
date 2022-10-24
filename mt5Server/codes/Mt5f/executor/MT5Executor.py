import MetaTrader5 as mt5

"""
This is pure executor that will not store any status of trades
The status will be stored in NodeJS server, including:

self-defined:
    StrategyId

MetaTrader:
    positionId
    
"""

class MT5Executor:
    def __init__(self, type_filling='ioc'):
        self.type_filling = type_filling

    def request_format(self, symbol, actionType, sl, tp, deviation, lot):
        """
        :param strategy_id: str, belong to specific strategy
        :param lots: [float]
        :param close_pos: Boolean, if it is for closing position, it will need to store the position id for reference
        :return: requests, [dict], a list of request
        """

        # type of filling
        tf = None
        if self.type_filling == 'fok':
            tf = mt5.ORDER_FILLING_FOK
        elif self.type_filling == 'ioc':
            tf = mt5.ORDER_FILLING_IOC
        elif self.type_filling == 'return':
            tf = mt5.ORDER_FILLING_RETURN

        # building request format
        if actionType == 'long':
            action_type = mt5.ORDER_TYPE_BUY  # int = 0
            price = mt5.symbol_info_tick(symbol).ask
        elif actionType == 'short':
            action_type = mt5.ORDER_TYPE_SELL  # int = 1
            price = mt5.symbol_info_tick(symbol).bid
            # lot = -lot
        else:
            raise Exception("The lot cannot be 0")  # if lot equal to 0, raise an Error
        request = {
            'action': mt5.TRADE_ACTION_DEAL,
            'symbol': symbol,
            'volume': float(lot),
            'type': action_type,
            'price': price,
            'sl': sl,
            'tp': tp,
            'deviation': deviation,  # indeed, the deviation is useless when it is marketing order, note 73d
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": tf,
        }
        return request

    def request_execute(self, request):
        """
        :param request: request
        :return: Boolean
        """
        result = mt5.order_send(request)  # sending the request
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print("order_send failed, symbol={}, retcode={}".format(request['symbol'], result.retcode))
            return False
        print(
            f"Action: {request['type']}; by {request['symbol']} {result.volume:.2f} lots at {result.price:.5f} ( ptDiff={((request['price'] - result.price) * 10 ** mt5.symbol_info(request['symbol']).digits):.1f} ({request['price']:.5f}(request.price) - {result.price:.5f}(result.price) ))")
        return True


