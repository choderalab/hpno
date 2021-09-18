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
            hidden_features: Union[int, None]=None,
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

        if hidden_features is None:
            hidden_features = in_features

        # up
        for idx in range(2, max_level+1):
            if idx == 2:
                _in_features = in_features
            else:
                _in_features = hidden_features

            setattr(
                self,
                'd_up_%s' % idx,
                torch.nn.Sequential(
                    torch.nn.Linear(
                        # hard-code 3:
                        # (previous_state, left, right, is_ring)
                        2 * _in_features + int(ring),
                        hidden_features,
                    ),
                    activation,
                )
            )

        # down
        for idx in range(1, max_level): # hard-code 1 for neighbor
            if idx == 1:
                _out_features = out_features
                _in_features = in_features
            else:
                _out_features = hidden_features
                _in_features = hidden_features

            setattr(
                self,
                'd_down_%s' % idx,
                torch.nn.Sequential(
                    torch.nn.Linear(
                        _in_features + hidden_features,
                        _out_features,
                    ),
                    activation,
                )
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
        for idx in range(2, self.max_level+1):
            graph.multi_update_all(
                etype_dict={
                    'n%s_as_%s_in_n%s' % (idx-1, pos_idx, idx): (

                        # msg_func
                        dgl.function.copy_src(
                            src='h',
                            out='m%s' % pos_idx,
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

            graph.apply_nodes(
                func=lambda nodes: {
                    'h':  getattr(self, 'd_up_%s' % idx)(
                        torch.cat(
                            [
                                nodes.data['is_ring']
                                for _ in range(int(self.ring))
                            ]
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
        for idx in range(self.max_level, 1, -1):

            graph.multi_update_all(
                etype_dict={
                    'n%s_has_%s_n%s' % (idx, pos_idx, idx-1): (
                        dgl.function.copy_src(src='h', out='m'),
                        dgl.function.sum(msg='m', out='h_down')
                    ) for pos_idx in range(2)
                },
                cross_reducer='sum'
            )

            graph.apply_nodes(
                func=lambda nodes: {
                    'h': getattr(self, 'd_down_%s' % (idx-1))(
                        torch.cat(
                            [
                                nodes.data['h'],
                                nodes.data['h_down']
                            ],
                            dim=-1,
                        )
                    )
                },
                ntype='n%s' % (idx-1),
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


class StochasticHierarchicalPathNetworkLayer(HierarchicalPathNetworkLayer):
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
            hidden_features: Union[int, None]=None,
            max_level: int=4,
            activation: Callable=torch.nn.SiLU(),
            ring=False,
        ):
        assert ring is False, "For stochastic version, no ring is supported."
        super(StochasticHierarchicalPathNetworkLayer, self).__init__(
            in_features=in_features,
            out_features=out_features,
            hidden_features=hidden_features,
            max_level=max_level,
            activation=activation,
            ring=False,
        )

    def trim(self, graph):
        graph = graph.local_var()
        desired_number_of_nodes = graph.number_of_nodes("n2")
        for level in range(3, self.max_level+1):
            number_of_nodes = graph.number_of_nodes("n%s" % level)
            node_indices_to_delete = torch.randperm(
                number_of_nodes
            )[desired_number_of_nodes:]
            graph.remove_nodes(
                nids=node_indices_to_delete,
                ntype="n%s" % level
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
        graph = self.trim(graph)
        feat = super().forward(graph, feat)
        return feat
