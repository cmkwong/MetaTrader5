import tkinter as tk
from datetime import datetime
from tkcalendar import Calendar

class TkWindow:
    def __init__(self, root):
        self.root = root
        self.widgets = {}

    def displayWidget(self, frame, id, elementProperty):
        """
        display widget onto frame, and save the widget into widgets = {}
        :param frame: tk.Frame
        :param elementProperty:
        :return:
        """
        # get the element type
        elementType = elementProperty['wtype']
        # get the position
        row, column, columnspan = elementProperty['pos']
        # display label
        if elementType != 'Button':
            tk.Label(frame, text=elementProperty['label']).grid(row=row, column=column)  # label field
        # get the style
        style = {}
        if 'style' in elementProperty.keys():
            style = elementProperty['style']
        # define widget
        widget = None
        if elementType == 'Calendar':
            d = datetime.strptime(elementProperty['value'], '%Y-%m-%d')  # date value
            widget = Calendar(frame, selectmode='day', year=d.year, month=d.month, day=d.day, **style)
        elif elementType == 'Label':
            widget = tk.Label(frame, text=elementProperty['value'], **style)
        elif elementType == 'Entry':
            widget = tk.Entry(frame, **style)
        elif elementType == 'OptionMenu':
            widget = tk.OptionMenu(frame, elementProperty['variable'], *elementProperty['value'], **style)
        elif elementType == 'Button':
            widget = tk.Button(frame, text=elementProperty['label'], command=elementProperty['value'], **style)

        # display the widget
        widget.grid(row=row, column=column + 1)

        # save widgets
        self.widgets[id] = widget

    def createFrame(self, frameLabel, elementsDict):
        """
        :param widgetDict: {'id': {'myCalendar', wtype: 'Calendar', label: 'Start', 'value': '2022-01-02', 'pos': (0,0,0) },
                            'id': {'myLabel', wtype: 'Label', label: 'Status', 'value': 'Error Occurred', 'pos': (0,1,0) },
                            'id': {'inputField1', wtype: 'Entry', label: 'MyAge', value: '', 'style': { width: 50, borderwidth: 3}, 'pos': (0,0,0) },
                            'id': {'dropdown1', wtype: 'OptionMenu', label: 'This is my Dropdown', value: [], variable: tkVar, 'pos': (0,0,0) },
                            'id': {'button1', wtype: 'Button', label: 'Click Me', value: fn, 'style': {width: 20}, 'pos': (0,0,0) }
                           }
        :return: frame
        """
        # create frame
        frame = tk.LabelFrame(self.root, text=frameLabel)
        # assign the widget
        for id, elementProperty in elementsDict.items():
            self.displayWidget(frame, id, elementProperty)
        return frame

