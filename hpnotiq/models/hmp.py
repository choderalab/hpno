""" Hierarchical Message Passing. """

# =============================================================================
# IMPORTS
# =============================================================================
import numpy as np
import torch
import dgl


# =============================================================================
# MODULE CLASSES
# =============================================================================
class HierarchicalMessagePassing(torch.nn.Module):
    """ Hierarchical Message Passing.

    Parameters
    ----------
    units : `int`
        (Default value = 32)
        Units in hidden layer.



    Attributes
    ----------
    d_up_ : `torch.nn.Linear`
        Upstream message passing.

    d_down_ : `torch.nn.Linear`
        Downsteam message passing.



    """
    def __init__(self, units=32, activation='Tanh', max_level=4):
        super(HierarchicalMessagePassing, self).__init__()

        # bookkeeping
        self.units = units
        self.max_level = max_level
        self.activation = activation()

        # up
        for idx in range(2, max_level+1): # hard-code 2 for neighbor
            setattr(
                self,
                'd_up_%s' % idx,
                torch.nn.Sequential(
                    torch.nn.Linear(
                        # hard-code 3:
                        # (previous_state, left, right)
                        3 * units,
                        units,
                    ),
                    getattr(torch.nn, self.activation)(),
                )
            )

        # down
        for idx in range(1, max_level): # hard-code 1 for neighbor
            setattr(
                self,
                'd_down_%s' % idx,
                torch.nn.Sequential(
                    torch.nn.Linear(
                        # hard-code 2:
                        # (previous_state, current_state)
                        2 * units,
                        units,
                    ),
                    getattr(torch.nn, self.activation)(),
                )
            )

    def forward(self, g):
        """ Forward pass.

        Parameters
        ----------
        g : `dgl.DGLHeteroGraph`
            Input graph.

        Returns
        -------
        g : `dgl.DGLHeteroGraph`
            Output graph.

        """

        # up
        for idx in range(2, self.max_level+1):
            g.multi_update_all(
                etype_dict={
                    'n%s_as_%s_in_n%s' % (idx-1, pos_idx, idx): (

                        # msg_func
                        dgl.function.copy_src(
                            src='h',
                            out='m%s' % pos_idx
                        ),

                        # reduce_func
                        dgl.function.sum(
                            msg='m%s' % pos_idx,
                            out='h%s' % pos_idx,
                        ),

                    ) for pos_idx in range(2)
                },
                cross_reducer='sum'
            )


            g.apply_nodes(
                func=lambda nodes: {
                    'h':  getattr(self, 'd_up_%s' % idx)(
                        torch.cat(
                            [nodes.data['h']]
                            + [
                                nodes.data['h%s' % pos_idx]
                                for pos_idx in range(2)
                            ],
                            dim=-1
                        )
                    )
                },
                ntype='n%s' % idx,
            )

        # down
        for idx in range(self.max_level, 1, -1):
            g.multi_update_all(
                etype_dict={
                    'n%s_has_%s_n%s' % (idx, pos_idx, idx-1): (
                        dgl.function.copy(src='h', out='m'),
                        dgl.function.sum(msg='m', out='h_down')
                    ) for pos_idx in range(2)
                },
                cross_reducer='sum'
            )

            g.apply_nodes(
                func=lambda nodes: {
                    'h': getattr(self, 'd_down_%s' % (idx-1))(
                        torch.cat(
                            [nodes.data['h']] + [nodes.data['h_down']]
                            for pos_idx in range(2)
                        )
                    )
                },
                cross_reducer='sum'
            )

        return g
