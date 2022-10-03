"""
===========
eda
===========
"""

from utils import make_whole_data
from ai4water.eda import EDA

data = make_whole_data(['tetx_coppml', 'sul1_coppml', 'aac_coppml'])

eda = EDA(data=data )

eda.correlation()