"""Phase 7 fixed-model evaluation; test outputs never change fit or label mapping."""

import numpy as np


def predict_gaussian(model, X):
    """Independent Gaussian MAP labels (T,) using supervised emissions and pi.

    The same supervised initial distribution is applied at every window. A,
    neighbouring observations and decoded labels have no effect on this baseline.
    Supervised emission state IDs already equal the fixed activity label IDs.
    """
    with np.errstate(divide='ignore'):
        scores = model._log_emission(X) + np.log(model.pi)
    return np.argmax(scores, axis=1).astype(np.int64)
