import numpy as np


def bce_loss(p: np.ndarray, y: np.ndarray) -> float:
    """
    p: predicted probabilities, shape (n_samples,)
    y: true labels (0 or 1), shape (n_samples,)
    Returns a single scalar.
    """
    # TODO: Implement Binary Cross-Entropy, as derived in Theory.
    # Clip p away from exactly 0 or 1 before taking a log, and return
    # a plain Python float, not a NumPy scalar.
    scaler=np.clip(p,10**-12,1-10**-12)
    loss=-(y*np.log(scaler)+(1-y)*np.log(1-scaler))
    return float(np.mean(loss))
