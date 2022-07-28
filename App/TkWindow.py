import tkinter as tk
from tkcalendar import Calendar
from datetime import datetime, date
import inspect

from TkInitWidget import TkInitWidget
from TkWidget import TkWidget


class TkWindow(TkWidget):
    def __init__(self):
        super(TkWindow, self).__init__()

    def createFrame(self, root, Widgets, label=None):
        """
        :param widgetDict: {'myCalendar': {'wtype': 'Calendar', label: 'Start', 'value': '2022-01-02', 'pos': (0,0,0) },
                            'myLabel': {'wtype': 'Label', label: 'Status', 'value': 'Error Occurred', 'pos': (0,1,0) },
                            'inputField1': {'wtype': 'Entry', label: 'MyAge', value: '', 'style': { width: 50, borderwidth: 3}, 'pos': (0,0,0) },
                            'dropdown1': {'wtype': 'OptionMenu', label: 'This is my Dropdown', value: [], variable: tkVar, 'pos': (0,0,0) },
                            'button1': {'wtype': 'Button', label: 'Click Me', value: fn, 'style': { width: 20}, 'pos': (0,0,0) }
                           }
        :return: frame
        """
        # create frame
        if label:
            frame = tk.LabelFrame(root, text=label)
        else:
            frame = tk.Frame(root)
        # assign the widget onto frame
        for ele in Widgets:
            self.getWidget(frame, ele)
        return frame

    def openTopWindowByFrame(self, root, getFrameCallbacks: list, title='tk window', windowSize='400x400', **kwargs):
        subRoot = tk.Toplevel(root)
        subRoot.title(title)
        subRoot.geometry(windowSize)
        for getFrameCallback in getFrameCallbacks:
            frame = getFrameCallback(subRoot, **kwargs)
            frame.pack()
        return subRoot
