import tkinter as tk
from datetime import datetime
from tkcalendar import Calendar
import collections

class TkWindow:
    def __init__(self):
        self.widgets = {}
        # constant
        self.DROPDOWN = 'dropdown'
        self.LABEL = 'label'
        self.TEXTFIELD = 'textfield'
        self.BUTTON = 'button'
        self.CALENDAR = 'calendar'

    def getWidget(self, frame, ele):
        """
        display widget onto frame, and save the widget into widgets = {}
        :param frame: tk.Frame
        :param ele:
        :return:
        """
        # get the element type
        elementType = ele.type
        # get the position
        row, column, columnspan = ele.pos
        # display label
        if elementType != self.BUTTON:
            tk.Label(frame, text=ele.label).grid(row=row, column=column)  # label field
        # get the style
        style = ele.style
        if not style:
            style = {}
        # define widget
        widget = None
        if elementType == self.CALENDAR:
            d = datetime.strptime(ele.value, '%Y-%m-%d')  # date value
            widget = Calendar(frame, selectmode='day', year=d.year, month=d.month, day=d.day, **style)
        elif elementType == self.LABEL:
            widget = tk.Label(frame, text=ele.value, **style)
        elif elementType == self.TEXTFIELD:
            widget = tk.Entry(frame, **style)
        elif elementType == self.DROPDOWN:
            widget = tk.OptionMenu(frame, ele.var, *ele.value)
        elif elementType == self.BUTTON:
            widget = tk.Button(frame, text=ele.label, command=ele.command, **style)

        # display the widget
        widget.grid(row=row, column=column + 1)
        return widget

    def createFrame(self, parent, frameLabel, Elements):
        """
        :param widgetDict: {'myCalendar': {'wtype': 'Calendar', label: 'Start', 'value': '2022-01-02', 'pos': (0,0,0) },
                            'myLabel': {'wtype': 'Label', label: 'Status', 'value': 'Error Occurred', 'pos': (0,1,0) },
                            'inputField1': {'wtype': 'Entry', label: 'MyAge', value: '', 'style': { width: 50, borderwidth: 3}, 'pos': (0,0,0) },
                            'dropdown1': {'wtype': 'OptionMenu', label: 'This is my Dropdown', value: [], variable: tkVar, 'pos': (0,0,0) },
                            'button1': {'wtype': 'Button', label: 'Click Me', value: fn, 'style': {width: 20}, 'pos': (0,0,0) }
                           }
        :return: frame
        """
        # create frame
        frame = tk.LabelFrame(parent, text=frameLabel)
        # assign the widget onto frame
        for ele in Elements:
            id = ele.id
            self.widgets[id] = self.getWidget(frame, ele)
        return frame

    def openWindow(self, parent, getFrameCallbacks:list, windowSize='400x200'):
        window = tk.Toplevel(parent)
        window.geometry(windowSize)
        for getFrameCallback in getFrameCallbacks:
            frame = getFrameCallback(window)
            frame.pack()
        return window