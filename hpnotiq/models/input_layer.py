""" Input net. """

# =============================================================================
# IMPORTS
# =============================================================================
import numpy as np
import torch
import dgl

# =============================================================================
# MODULE CLASSES
# =============================================================================
class InputLayer(torch.nn.Module):
    """ Input layer.

    Parameters
    ----------
    in_features : `int`
        (Default value=117)

    out_features : `int`
        (Default value=32)
    """
    def __init__(self, in_features=117, out_features=32, max_level=4):
        super(InputLayer, self).__init__()
        # bookkeeping
        self.in_features = in_features
        self.out_features = out_features
        self.max_level = max_level


        self.d = torch.nn.Linear(
            in_features=self.in_features,
            out_features=self.out_features
        )


    def forward(self, g):
        g.apply_nodes(
            lambda nodes: {
                'h': self.d(nodes.data['h0'])
            },
            ntype='n1'
        )

        # here we assume that all the nodes have the same number of
        # hidden dimension
        for idx in range(2, self.max_level+1):
            g.apply_nodes(
                lambda nodes: {
                    'h': torch.zeros(
                        g.number_of_nodes(ntype='n%s' % idx),
                        self.out_features
                    )
                },
                ntype='n%s' % idx,
            )

        return g
