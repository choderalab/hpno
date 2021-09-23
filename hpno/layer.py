""" Hierarchical Message Passing. """

# =============================================================================
# IMPORTS
# =============================================================================
import torch
import dgl
from typing import Union, Callable

# =============================================================================
# MODULE CLASSES
# =============================================================================

class HierarchicalPathNetworkLayer(torch.nn.Module):
    """ Hierarchical Path Network Layer.

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

    ring : bool
        If true, include ring information.

    Attributes
    ----------
    d_up_ : `torch.nn.Linear`
        Upstream message passing.

    d_down_ : `torch.nn.Linear`
        Downsteam message passing.

    """
    def __init__(
            self,
            in_features: int,
            out_features: int,
            max_level: int=4,
            activation: Callable=torch.nn.SiLU(),
            ring: bool=False,
        ):
        super(HierarchicalPathNetworkLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.max_level = max_level
        self.ring = ring
        self.activation = activation
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(3*in_features, out_features),
            activation,
        )

    def upward(self, graph):
        """ Upward pass.

        Parameters
        ----------
        graph : dgl.DGLHeteroGraph
            Input graph.

        Returns
        -------
        dgl.DGLHeteroGraph: Output graph.

        """
        graph = graph.local_var()

        # add log == multiply
        for idx in range(2, self.max_level+1):
            graph.multi_update_all(
                etype_dict={
                    'n%s_in_n%s' % (idx-1, idx): (

                        # msg_func
                        dgl.function.copy_src(
                            src='h',
                            out='m',
                        ),

                        # reduce_func
                        dgl.function.sum(
                            msg='m',
                            out='h',
                        ),

                    )
                },
                cross_reducer='sum'
            )

        for idx in range(1, self.max_level+1):
            graph.nodes['n%s' % idx].data['h_softmax'] = graph.nodes['n%s' % idx].data['h'].softmax(dim=-1)

        return graph

    def downward(self, graph):
        """
        Parameters
        ----------
        graph : dgl.DGLHeteroGraph
            Input graph.

        Returns
        -------
        dgl.DGLHeteroGraph: Output graph.

        """
        graph = graph.local_var()
        graph.apply_nodes(
            lambda node: {'h_softmax': node.data['h'].softmax(dim=-1)},
            ntype='n%s' % self.max_level
        )
        for idx in range(self.max_level, 2, -1):
            graph.multi_update_all(
                etype_dict={
                    'n%s_has_n%s' % (idx, idx-1): (
                        dgl.function.copy_src(src='h_softmax', out='m'),
                        dgl.function.sum(msg='m', out='h_down'),
                        lambda node: {
                            'h_softamx': (
                                node.data['h'] + node.data['h_down']
                            ).softmax(dim=-1)
                        },
                    )
                },
                cross_reducer='sum'
            )

        graph.update_all(
            dgl.function.copy_src(src='h_softmax', out='m'),
            dgl.function.sum(msg='m', out='h_down'),
            etype='n2_has_n1',
        )

        graph.update_all(
            dgl.function.copy_src(src='h', out='m'),
            dgl.function.sum(msg='m', out='h2'),
            etype='n2_has_n1',
        )

        graph.apply_nodes(
            func=lambda nodes: {
                'h': self.linear(
                        torch.cat(
                            [
                                nodes.data['h'],
                                nodes.data['h2'],
                                nodes.data['h_down'],
                            ],
                            dim=-1,
                        ),
                ),
            },
            ntype='n1',
        )

        return graph

    def forward(self, graph, feat):
        """ Forward pass.

        Parameters
        ----------
        graph : dgl.DGLHeteroGraph
            Input graph.

        feat :  torch.Tensor
            Input features.

        Returns
        -------
        dgl.DGLHeteroGraph: Output graph.

        """
        graph = graph.local_var()
        graph.nodes['n1'].data['h'] = feat
        graph = self.upward(graph)
        graph = self.downward(graph)
        feat = graph.nodes['n1'].data['h']
        return feat
