__version__ = "0.0.1"
import importlib.util

from zenithml import data
from zenithml import utils
from zenithml import metrics
from zenithml import nvt
from zenithml import preprocess as pp
from zenithml.ray import runner as ray_runner

if importlib.util.find_spec("torch") is not None:
    from zenithml import torch

# if importlib.util.find_spec("tensorflow") is not None:
#     from zenithml import tf
