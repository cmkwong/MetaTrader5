import tkinter as tk
from tkcalendar import Calendar
from datetime import datetime, date
import inspect

from common import InitWidget

class TkWidget:
    def __init__(self):
        self.widgets = {}  # widgets[category][id] => widget
        self.variables = {}
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
            tk.Label(frame, text=ele.label).grid(row=row, column=column, padx=5, pady=5)  # label field
        # get the style
        style = ele.style
        if not style:
            style = {}
        # define widget and variable
        widget, variable = None, None
        if elementType == self.CALENDAR:
            if ele.default:
                d = datetime.strptime(ele.default, '%Y-%m-%d')  # date value
            else:
                d = date.today()
            widget = Calendar(frame, selectmode='day', year=d.year, month=d.month, day=d.day, **style)
            variable = widget
        elif elementType == self.LABEL:
            widget = tk.Label(frame, text=ele.value, **style)
            variable = widget
        elif elementType == self.TEXTFIELD:
            widget = tk.Entry(frame, **style)
            if ele.default:
                widget.insert(tk.END, ele.default)
            variable = widget
        elif elementType == self.DROPDOWN:
            widget = tk.OptionMenu(frame, ele.var, *ele.value)
            if ele.default:
                ele.var.set(ele.default)
            widget = ele.var
        elif elementType == self.BUTTON:
            widget = tk.Button(frame, text=ele.label, command=ele.command, **style)
            variable = widget

        # display the widget
        widget.grid(row=row, column=column + 1, padx=5, pady=5)
        return widget, variable

    def get_params_initWidgets(self, class_object, cat):
        """
        :param class_object: class / function
        :return: [widget]
        """
        # params details from object
        initWidgets = []
        sig = inspect.signature(class_object)
        for r, param in enumerate(sig.parameters.values()):
            initWidgets.append(InitWidget(cat=cat, id=param.name, type=self.TEXTFIELD, label=param.name, default=param.default, pos=(r, 0, 1)))
        return initWidgets

    def getWidgetValue(self, cat, id):
        pass
