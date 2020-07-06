""" Constructing hypergraph from homograph. """

# =============================================================================
# IMPORTS
# =============================================================================
import numpy as np
import torch
import dgl

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def get_indices_from_adjacency_matrix(a, max_size=4):
    """ Read the relationship indices from adjacency matrix.

    Parameters
    ----------
    a : `torch.SparseTensor`
        Input adjacency matrix.

    max_size : `int`
        Highest level of hypernodes.

    Returns
    -------
    idxs : `dict`
        Dictionary of indices of paths with various sizes.

    """
    # check input signature
    assert isinstance(max_size, int)
    assert max_size >= 2
    assert isinstance(a, torch.Tensor) or isinstance(a, torch.SparseTensor)

    # initialize output
    idxs = {}

    # loop through the levels
    for level in range(3, max_size + 1):

        # get the indices that is the basis of the level
        base_idxs = idxs["n%s" % (level - 1)]

        # enumerate all the possible pairs at base level
        base_pairs = torch.cat(
            [
                base_idxs[None, :, :].repeat(base_idxs.shape[0], 1, 1),
                base_idxs[:, None, :].repeat(1, base_idxs.shape[0], 1),
            ],
            dim=-1,
        ).reshape(-1, 2 * (level - 1))

        mask = 1.0
        # filter to get the ones that share some indices
        for idx_pos in range(level - 2):
            mask *= torch.eq(
                base_pairs[:, idx_pos + 1], base_pairs[:, idx_pos + level - 1]
            )

        mask *= 1 - 1 * torch.eq(base_pairs[:, 0], base_pairs[:, -1])

        mask = mask > 0.0

        # filter the enumeration to be output
        idxs_level = torch.cat(
            [
                base_pairs[mask][:, : (level - 1)],
                base_pairs[mask][:, -1][:, None]
            ],
            dim=-1,
        )

        # put results in output dictionary
        idxs["n%s" % level] = idxs_level

    return idxs


# =============================================================================
# MODULE FUNCTIONS
# =============================================================================
def from_homograph(g, max_size=4):
    """ Constructing hypergraph from homograph.

    Parameters
    ----------
    g : `dgl.DGLGraph`
        Input graph.

    max_size : `int`
        (Default value = 4)
        Highest level of hypernodes.

    Returns
    -------
    hg : `dgl.DGLHeteroGraph`
        Output graph.

    """
    # ==============
    # initialization
    # ==============
    # initialize hypergraph as a dictionary
    hg = {}

    # ========
    # indexing
    # ========
    # get adjacency matrix
    a = g.adjacency_matrix()

    # get indices
    idxs = get_indices_from_adjacency_matrix(a)

    # build a mapping between indices and the ordering
    idxs_to_ordering = {}
    for term in ["n%s" % level for level in range(2, max_size)]:
        idxs_to_ordering[term] = {
            tuple(subgraph_idxs): ordering
            for (ordering, subgraph_idxs) in enumerate(list(idxs[term]))
        }

    # NOTE:
    # here we define all the possible
    # 'has' and 'in' relationships.
    # TODO:
    # we'll test later to see if this adds too much overhead
    for small_idx in range(1, max_size+1): # child
        for big_idx in range(small_idx + 1, max_size+1): # parent
            for pos_idx in range(big_idx - small_idx + 1): # position

                # `in` relationship
                hg[ # (source, relationship, destination)
                    (
                        "n%s" % small_idx,
                        "n%s_as_%s_in_n%s" % (small_idx, pos_idx, big_idx),
                        "n%s" % big_idx,
                    )
                ] = np.stack( # use `np.array` here but convert to list later
                    [
                        np.array(
                            [
                                idxs_to_ordering["n%s" % small_idx][tuple(x)]
                                for x in idxs["n%s" % big_idx][
                                    :, pos_idx : pos_idx + small_idx
                                ]
                            ]
                        ),
                        np.arange(idxs["n%s" % big_idx].shape[0]),
                    ],
                    axis=1,
                )

                # define the same for `has` relationship
                hg[
                    (
                        "n%s" % big_idx,
                        "n%s_has_%s_n%s" % (big_idx, pos_idx, small_idx),
                        "n%s" % small_idx,
                    )
                ] = np.stack(
                    [
                        np.arange(idxs["n%s" % big_idx].shape[0]),
                        np.array(
                            [
                                idxs_to_ordering["n%s" % small_idx][tuple(x)]
                                for x in idxs["n%s" % big_idx][
                                    :, pos_idx : pos_idx + small_idx
                                ]
                            ]
                        ),
                    ],
                    axis=1,
                )

    # convert all to python `List`
    hg = dgl.heterograph({key: list(value) for key, value in hg.items()})

    # include indices in the nodes themselves
    for term in ["n%s" % level for level in range(1, max_size+1)]:
        hg.nodes[term].data["idxs"] = torch.tensor(idxs[term])

    # NOTE: only relationships are handled here, not data
    # assignment is needed if data is in homograph
    return hg
