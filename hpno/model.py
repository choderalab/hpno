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
class HierarchicalPathNetwork(torch.nn.Module):
    """ Hierarchical Path Network.

    Parameters
    ----------
    units : `int`
        (Default value = 32)
        Units in hidden layer.

    Attributes
    ----------
    hyno_layer_

    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: int,
        depth: int,
        activation: Callable=torch.nn.SiLU(),
    ):
        super(HierarchicalPathNetwork, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.depth = depth
        self.activation = activation

        for idx in range(depth):
            _in_features = in_features if idx == 0 else hidden_features
            _out_features = out_features if idx == depth-1 else hidden_features
            _activation = torch.nn.Identity() if idx == depth-1 else activation
            setattr(
                self,
                "hpno_layer_%s" % idx,
                HierarchicalPathNetworkLayer(
                    in_features=_in_features,
                    out_features=_out_features,
                    hidden_features=hidden_features,
                    activation=_activation,
                )
            )

    def forward(self, graph, feat):
        """
        Parameters
        ----------
        graph : dgl.DGLHeteroGraph
            Input graph.

        feat : torch.Tensor
            Input feature.

        Attributes
        ----------
        d_up_ : `torch.nn.Linear`
            Upstream message passing.

        d_down_ : `torch.nn.Linear`
            Downsteam message passing.
        """
        graph = graph.local_var()
        for idx in range(self.depth):
            feat = getattr(self, "hpno_layer_%s" % idx)(graph, feat)
        return feat
