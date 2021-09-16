""" Hierarchical Message Passing. """

# =============================================================================
# IMPORTS
# =============================================================================
import numpy as np
import torch
import dgl
from typing import Callable, Union
from .layer import HierarchicalPathNetworkLayer

# =============================================================================
# MODULE CLASSES
# =============================================================================
class GraphReadout(torch.nn.Module):
    """ Graph level readout.

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

    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: int,
        max_level: int=4,
        activation: Callable=torch.nn.SiLU(),
    ):
        super(GraphReadout, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.max_level = max_level
        self.activation = activation
        self.upward_layer = HierarchicalPathNetworkLayer(
            in_features=in_features,
            out_features=out_features,
            hidden_features=hidden_features,
        )

        # hack the layer by removing all the downward steps
        for name, child in self.upward_layer.named_children():
            print(name, child)
            if "down" in name:
                setattr(self.upward_layer, name, torch.nn.Identity())

        self.summarize_layer = torch.nn.Sequential(
            torch.nn.Linear(
                in_features + (max_level - 1) * hidden_features,
                hidden_features,
            ),
            activation,
            torch.nn.Linear(
                hidden_features,
                out_features,
            )
        )

        print(self)

    def forward(self, graph, feat):
        """ Forward pass.

        Parameters
        ----------
        graph : dgl.DGLHeteroGraph
            Input graph.

        feat : torch.Tensor
            Input features.

        Returns
        -------
        torch.Tensor : graph-level feature

        """
        graph = graph.local_var()

        # apply activation first because the original features are without
        feat = self.activation(feat)
        graph.nodes['n1'].data['h'] = feat

        # apply the upward pass
        graph = self.upward_layer.upward(graph)

        # graph all levels
        for idx in range(1, self.max_level+1):
            graph.multi_update_all(
                etype_dict={
                    'n%s_in_g' % idx: (
                        # msg_func
                        dgl.function.copy_src(
                            src='h',
                            out='m%s' % idx,
                        ),

                        # reduce_func
                        dgl.function.sum(
                            msg='m%s' % idx,
                            out='h%s' % idx,
                        ),

                    ) for pos_idx in range(2)
                },
                cross_reducer='sum'
            )

        # summarize all levels
        graph.apply_nodes(
            lambda node: {
                "h": self.summarize_layer(
                    torch.cat(
                        [
                            node.data["h%s" % idx]
                            for idx in range(1, self.max_level+1)
                        ],
                        dim=-1,
                    )
                )
            },
            ntype="g",
        )

        # grab feature
        feat = graph.nodes['g'].data['h']
        return feat
