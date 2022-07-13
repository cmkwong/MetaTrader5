import collections
from dataclasses import dataclass
from typing import Union

@dataclass
class InitWidget:
    id: str
    type: str
    value: any = None
    var: any = None
    command: any = None
    label: str = ''
    default: any = None
    pos: tuple = (0, 0, 0)
    style: dict = None

# fields = ['id', 'type', 'value', 'var', 'command', 'label', 'pos', 'style']
# ElementC = collections.namedtuple('ElementC', fields, defaults=(None,) * len(fields))