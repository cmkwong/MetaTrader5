import tkinter as tk
from tkinter import scrolledtext
from tkcalendar import Calendar, DateEntry
from datetime import datetime, date
import inspect

from TkInitWidget import TkInitWidget

class TkWidgetLabel:
    # constant
    DROPDOWN = 'OptionMenu'
    LABEL = 'Label'
    TEXTFIELD = 'Entry'
    BUTTON = 'Button'
    CALENDAR = 'DateEntry'
    SCROLLEDTEXT = 'ScrolledText'

class TkWidget(TkWidgetLabel):
    def __init__(self):
        super(TkWidget, self).__init__()
        self.widgets = {}  # widgets[category][id] => widget
        self.variables = {}

    def _storeWidgetVariable(self, widget, variable, ele):
        cat = ele.cat
        if cat not in self.widgets.keys():
            self.widgets[cat] = {}
            self.variables[cat] = {}
        id = ele.id
        self.widgets[cat][id], self.variables[cat][id] = widget, variable

    def getWidget(self, root, ele):
        """
        display widget onto root, and save the widget into widgets = {}
        :param root: tk.root / tk.Frame
        :param ele:
        :return:
        """
        # get the element type
        elementType = ele.type
        # get the position
        row, column, columnspan = ele.pos
        # display label
        if elementType != self.BUTTON:
            tk.Label(root, text=ele.label).grid(row=row, column=column, padx=5, pady=5)  # label field
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
            widget = DateEntry(root, selectmode='day', year=d.year, month=d.month, day=d.day, **style)
            variable = widget
        elif elementType == self.LABEL:
            widget = tk.Label(root, text=ele.value, **style)
            variable = widget
        elif elementType == self.TEXTFIELD:
            widget = tk.Entry(root, **style)
            if ele.default:
                widget.insert(tk.END, ele.default)
            variable = widget
        elif elementType == self.DROPDOWN:
            variable = tk.StringVar(root)
            widget = tk.OptionMenu(root, variable, *ele.value)
            if ele.default:
                variable.set(ele.default)
        elif elementType == self.BUTTON:
            widget = tk.Button(root, text=ele.label, command=ele.onClick, **style)
            variable = widget
        elif elementType == self.SCROLLEDTEXT:
            widget = scrolledtext.ScrolledText(root, undo=False, **style)
            variable = widget
        # display the widget
        widget.grid(row=row, column=column + 1, padx=5, pady=5)
        self._storeWidgetVariable(widget, variable, ele)

    def get_params_initWidgets(self, class_object, cat):
        """
        :param class_object: class / function
        :return: [widget]
        """
        # params details from object
        initWidgets = []
        sig = inspect.signature(class_object)
        for r, param in enumerate(sig.parameters.values()):
            initWidgets.append(TkInitWidget(cat=cat, id=param.name, type=self.TEXTFIELD, label=param.name, default=param.default, pos=(r, 0, 1)))
        return initWidgets

    def getWidgetValue(self, cat, id):
        value = None
        widgetType = type(self.widgets[cat][id]).__name__
        if widgetType == self.CALENDAR:
            value = self.variables[cat][id].get_date()
        elif widgetType == self.LABEL:
            pass
        elif widgetType == self.TEXTFIELD:
            value = self.variables[cat][id].get()
        elif widgetType == self.DROPDOWN:
            value = self.variables[cat][id].get()
        elif widgetType == self.BUTTON:
            pass
        return value

