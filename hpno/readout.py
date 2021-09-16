""" Hierarchical Message Passing. """

# =============================================================================
# IMPORTS
# =============================================================================
import numpy as np
import torch
import dgl
from typing import Callable
from .layer import HierarchicalPathNetworkLayer

# =============================================================================
# MODULE CLASSES
# =============================================================================
class GraphReadout(torch.nn.Module):
    """ Graph level readout.

    Parameters
    ----------


    """
