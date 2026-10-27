import numpy as np


def l1_loss(input: np.ndarray, target: np.ndarray, reduction: str = "mean") -> float | np.ndarray:
    """
    input, target: same shape, any shape.
    reduction: "mean" (default), "sum", or "none".

    Mirrors torch.nn.functional.l1_loss: mean absolute error, not
    squared error. See Theory for why this changes MSE Loss's
    behavior around outliers.
    """
    absolute_error=np.abs(input-target)

    if reduction =="mean":
        return absolute_error.mean()
    elif reduction =='sum':
        return absolute_error.sum()
    elif reduction =="none":
        return absolute_error
    else:
        raise ValueError("Invalid Reduction")
