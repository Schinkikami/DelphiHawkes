import torch
from torch import Tensor


def inverse_softplus(x: Tensor):
    # Computes the inverse of the softplus function,
    # using the numerically stable log(expm1(x)) implementation
    # (sadly torch does not provide logexpm1)
    return torch.log(torch.expm1(x))
