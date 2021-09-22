""" Hierarchical Message Passing. """

# =============================================================================
# IMPORTS
# =============================================================================
import numpy as np
import torch
import dgl
from typing import Callable, Union
from .layer import HierarchicalPathNetworkLayer
from .readout import GraphReadout

# =============================================================================
# MODULE CLASSES
# =============================================================================
class HierarchicalPathNetwork(torch.nn.Module):
    """ Hierarchical Path Network.

    Parameters
    ----------
    in_features : int
        Input features.

    out_features : int
        Output features.

    hidden_features : int
        Hidden features.

    max_level : int
        Maximum level of hierarchical message passing.

    activation : Callable
        Activation function for layer.

    max_level : int
        Maximum level of message passing.

    ring : bool
        If true, include ring information.

    readout : Union[None, Callable]
        If not None, apply readout.

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
        max_level: int=4,
        ring: bool=False,
        activation: Callable=torch.nn.SiLU(),
        readout: Union[None, Callable]=None,
    ):
        super(HierarchicalPathNetwork, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.depth = depth
        self.activation = activation
        self.max_level = max_level
        self.ring = ring
        self.readout = readout

        self.in_linear = torch.nn.Sequential(
            torch.nn.Linear(in_features, hidden_features),
            activation,
            torch.nn.Linear(hidden_features, hidden_features)
        )

        for idx in range(depth):
            _in_features = hidden_features
            _out_features = out_features if idx == depth-1 else hidden_features
            _activation = torch.nn.Identity() if idx == depth-1 else activation

            setattr(
                self,
                "hpno_layer_%s" % idx,
                HierarchicalPathNetworkLayer(
                    in_features=_in_features,
                    out_features=_out_features,
                    activation=_activation,
                    max_level=max_level,
                    ring=ring,
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
        feat = self.in_linear(feat)
        for idx in range(self.depth):
            feat = getattr(self, "hpno_layer_%s" % idx)(graph, feat)

        if self.readout is not None:
            feat = self.readout(graph, feat)
        return feat
