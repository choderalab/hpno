""" Metrics to evaluate and train models.

"""
# =============================================================================
# IMPORTS
# =============================================================================
import torch

# =============================================================================
# MODULE FUNCTIONS
# =============================================================================
def mse(y, y_hat):
    """ Mean squarred error. """
    assert y.numel() == y_hat.numel()
    return torch.nn.functional.mse_loss(
        y.flatten(),
        y_hat.flatten()
    )

def rmse(y, y_hat):
    """ Rooted mean squarred error. """
    assert y.numel() == y_hat.numel()
    return torch.sqrt(
        torch.nn.functional.mse_loss(
            y.flatten(),
            y_hat.flatten()
        )
    )

def r2(y, y_hat):
    """ R2 """
    ss_tot = (y - y.mean()).pow(2).sum()
    ss_res = (y_hat - y).pow(2).sum()
    return 1 - torch.div(ss_res, ss_tot)
