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

        # final
        self.final_linear = torch.nn.Sequential(
            torch.nn.Linear(in_features+2*hidden_features, hidden_features),
            activation,
            torch.nn.Linear(hidden_features, out_features),
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

        graph.multi_update_all(
            {
                "n2_has_%s_n1" % pos_idx: (
                    dgl.function.copy_src(src='h', out='m'),
                    dgl.function.sum(msg='m', out='h2'),
                )
                for pos_idx in range(1)
            },
            cross_reducer="sum",
        )

        for idx in range(self.max_level, 2, -1):
            graph.multi_update_all(
                {
                    "n%s_has_%s_n%s" % (idx, pos_idx, (idx-1)): (
                        dgl.function.copy_src("h", "m"),
                        dgl.function.sum("m", "h"),
                    )
                    for pos_idx in range(1)
                },
                cross_reducer="mean"
            )

        graph.multi_update_all(
            {
                "n2_has_%s_n1" % pos_idx: (
                    dgl.function.copy_src("h", "m"),
                    dgl.function.sum("m", "h_down"),
                )
                for pos_idx in range(1)
            },
            cross_reducer="sum"
        )

        for field in ['h', 'h2', 'h_down']:
            print(graph.nodes['n1'].data[field].shape)

        graph.apply_nodes(
            func=lambda nodes: {
                'h': self.final_linear(
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
