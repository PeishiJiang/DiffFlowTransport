"""Performance metrics."""

# Author: Peishi Jiang
# Email: shixijps@gmail.com

import numpy as np
import hydroeval as he

from sklearn.metrics import r2_score
from jaxtyping import Array


def compute_metrics(
    pred: Array,
    true: Array,
    mask_naninf: bool = False,
):
    """
    Computing a bunch of evaluation metrics
    """
    pred, true = np.array(pred), np.array(true)  # pyright: ignore
    assert pred.shape == true.shape

    if mask_naninf:
        mask = np.isfinite(true)
        pred = pred[mask]
        true = true[mask]

    # Relative square error
    def func_rse(pred, true):
        upper = np.sum((pred - true) ** 2)
        lower = np.sum((true - true.mean()) ** 2)
        return upper / lower

    rse = func_rse(pred, true)
    mare = he.evaluator(he.mare, pred, true)[0]
    rmse = he.evaluator(he.rmse, pred, true)[0]
    kge = he.evaluator(he.kge, pred, true)[0][0]
    mkge_all = he.evaluator(he.kgeprime, pred, true).flatten()
    mkge = mkge_all[0]
    cc = mkge_all[1]
    beta = mkge_all[2]
    alpha = mkge_all[3]
    nse = he.evaluator(he.nse, pred, true)[0]
    r2 = r2_score(pred, true)
    # try:
    # except:
    #     print(pred.mean(), pred.max(), pred.min(), np.isnan(true).sum())
    #     print(true.mean(), true.max(), true.min(), np.isnan(true).sum())
    #     raise Exception("Wrong.")
    return {
        "rse": rse,
        "mare": mare,
        "rmse": rmse,
        "mse": rmse**2,
        "r2": r2,
        "kge": kge,
        "nse": nse,
        "mkge": mkge,
        "cc": cc,
        "alpha": alpha,
        "beta": beta,
    }