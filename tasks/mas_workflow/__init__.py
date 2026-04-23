from mas.mas import MetaMAS
from .autogen import AutoGen
from .macnet import MacNet
from .dylan import DyLAN
from .strategy import StrategyMAS

MAS = {
    'autogen': AutoGen,
    'macnet': MacNet,
    'dylan': DyLAN,
    'strategy': StrategyMAS,
}

def get_mas(mas_type: str) -> MetaMAS:

    if MAS.get(mas_type) is None:
        raise ValueError('Unsupported mas type.')
    return MAS.get(mas_type)() 