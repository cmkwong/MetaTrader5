from production.codes import config
from datetime import datetime, timedelta
import MetaTrader5 as mt5
import pytz

def get_txt2timeframe(timeframe_txt):
    timeframe_dicts = {"M1": mt5.TIMEFRAME_M1, "M2": mt5.TIMEFRAME_M2, "M3": mt5.TIMEFRAME_M3, "M4": mt5.TIMEFRAME_M4,
                      "M5": mt5.TIMEFRAME_M5, "M6": mt5.TIMEFRAME_M6, "M10": mt5.TIMEFRAME_M10,
                      "M12": mt5.TIMEFRAME_M12,
                      "M15": mt5.TIMEFRAME_M15, "M20": mt5.TIMEFRAME_M20, "M30": mt5.TIMEFRAME_M30,
                      "H1": mt5.TIMEFRAME_H1,
                      "H2": mt5.TIMEFRAME_H2, "H3": mt5.TIMEFRAME_H3, "H4": mt5.TIMEFRAME_H4, "H6": mt5.TIMEFRAME_H6,
                      "H8": mt5.TIMEFRAME_H8, "H12": mt5.TIMEFRAME_H12, "D1": mt5.TIMEFRAME_D1, "W1": mt5.TIMEFRAME_W1,
                      "MN1": mt5.TIMEFRAME_MN1}
    return timeframe_dicts[timeframe_txt]

def get_timeframe2txt(mt5_timeframe_txt):
    timeframe_dicts = {mt5.TIMEFRAME_M1: "M1", mt5.TIMEFRAME_M2: "M2", mt5.TIMEFRAME_M3: "M3", mt5.TIMEFRAME_M4: "M4",
                      mt5.TIMEFRAME_M5: "M5", mt5.TIMEFRAME_M6: "M6", mt5.TIMEFRAME_M10: "M10",
                      mt5.TIMEFRAME_M12: "M12",
                      mt5.TIMEFRAME_M15: "M15", mt5.TIMEFRAME_M20: "M20", mt5.TIMEFRAME_M30: "M30",
                      mt5.TIMEFRAME_H1: "H1",
                      mt5.TIMEFRAME_H2: "H2", mt5.TIMEFRAME_H3: "H3", mt5.TIMEFRAME_H4: "H4", mt5.TIMEFRAME_H6: "H6",
                      mt5.TIMEFRAME_H8: "H8", mt5.TIMEFRAME_H12: "H12", mt5.TIMEFRAME_D1: "D1", mt5.TIMEFRAME_W1: "D1",
                      mt5.TIMEFRAME_MN1: "MN1"}
    return timeframe_dicts[mt5_timeframe_txt]

def get_utc_time_from_broker(time, timezone):
    """
    :param time: tuple (year, month, day, hour, mins) eg: (2010, 10, 30, 0, 0)
    :param timezone: Check: set(pytz.all_timezones_set) - (Etc/UTC)
    :return: datetime format
    """
    dt = datetime(time[0], time[1], time[2], hour=time[3], minute=time[4]) + timedelta(hours=2, minutes=0)
    utc_time = pytz.timezone(timezone).localize(dt)
    return utc_time

def get_current_utc_time_from_broker(timezone):
    """
    :param time: tuple (year, month, day, hour, mins) eg: (2010, 10, 30, 0, 0)
    :param timezone: Check: set(pytz.all_timezones_set) - (Etc/UTC)
    :return: datetime format
    """
    now = datetime.today()
    dt = datetime(now.year, now.month, now.day, hour=now.hour, minute=now.minute) + timedelta(hours=config.BROKER_TIME_BETWEEN_UTC, minutes=0)
    utc_time = pytz.timezone(timezone).localize(dt)
    return utc_time

def get_time_string(tt):
    """
    :param tt: time_tuple: tuple (yyyy,m,d,h,m)
    :return: string
    """
    time_string = str(tt[0]) + '-' + str(tt[1]).zfill(2) + '-' + str(tt[2]).zfill(2) + '-' + str(tt[3]).zfill(2) + '-' + str(tt[4]).zfill(2)
    return time_string

def get_current_time_string():
    now = datetime.today()
    end_str = get_time_string((now.year, now.month, now.day, now.hour, now.minute))
    return end_str