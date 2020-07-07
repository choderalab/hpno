""" Output Layer. """

# =============================================================================
# IMPORTS
# =============================================================================
import numpy as np
import torch
import dgl

# =============================================================================
# MODULE CLASSES
# =============================================================================
class Pool(torch.nn.Module):
    """ Output layer.

    Parameters
    ----------
    in_features : `int`
        (Default value=32)

    """
    def __init__(
            self, fn='sum', in_features=32, out_features=1, activation='Tanh'):

        super(Pool, self).__init__()
        self.fn = fn
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation

        # readout net
        self.d = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=self.in_features,
                out_features=self.in_features,
            ),
            getattr(torch.nn, activation)(),
            torch.nn.Linear(
                in_features=self.in_features,
                out_features=self.out_features,
            )
        )

    def forward(self, g):
        """ Forward Pass.

        Parameters
        ----------
        g : `dgl.HeteroGraph`
            Input graph.

        """
        g.multi_update_all(
            {
                'n1_in_g': (
                    dgl.function.copy_src(src='h', out='m'),
                    getattr(dgl.function, self.fn)(msg='m', out='h')
                )
            },
            cross_reducer='sum'
        )

        g.apply_nodes(
            func=lambda node: {
                'y_hat': self.d(node.data['h'])
            },
            ntype='g'
        )
        
        return g
