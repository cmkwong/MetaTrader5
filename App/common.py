import collections
from dataclasses import dataclass
from typing import Union

@dataclass
class InitWidget:
    cat: str    # widget category
    id: str     # widget id
    type: str   # widget type: button, textfield, ...
    value: any = None   # widget value
    var: any = None     # variable is needed, mainly for dropdown
    command: any = None # fn for button click, if widget type is button
    label: str = ''     # label beside to widget
    default: any = None # default value, for textfield and dropdown
    pos: tuple = (0, 0, 0)  # position
    style: dict = None      # style

# fields = ['id', 'type', 'value', 'var', 'command', 'label', 'pos', 'style']
# ElementC = collections.namedtuple('ElementC', fields, defaults=(None,) * len(fields))