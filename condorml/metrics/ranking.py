from typing import List, Dict, Callable, Optional

import dask
import numpy as np
import pandas as pd
from dask import dataframe as dd


def mrr(r):
    if np.max(r) == 0:
        return 0.0
    else:
        return 1 / (1 + np.argmax(r))


def dcg_at_k(r, k, method=0):
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError("method must be 0 or 1.")
    return 0.0


def ndcg_at_k(r, k, method=0):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.0
    return dcg_at_k(r, k, method) / dcg_max


DEFAULT_RANKING_METRICS = {"mrr": mrr}


def compute_ranking_metrics(
    predictions_path: str,
    label_cols: List[str],
    id_var: str,
    metric_fns: Optional[Dict[str, Callable]] = None,
    score_col: str = "score",
):
    from ray.util.dask import ray_dask_get
    from tempfile import tempdir

    dask.config.set(scheduler=ray_dask_get)
    metric_fns = metric_fns or DEFAULT_RANKING_METRICS

    def compute_metrics(request):
        _df = pd.DataFrame(request)
        # _df[score_col] = np.random.rand()#] * len(_df[score_col])
        _df = _df.sort_values(by=[score_col], ascending=False)
        metric_results = {}
        for label_col in label_cols:
            reward = list(_df[label_col].values)
            for metric_name, metric_fn in metric_fns.items():
                metric_results[f"{metric_name}_{label_col}"] = metric_fn(reward)
        return metric_results

    with dask.config.set({"temporary_directory": tempdir}):

        # Read the prediction from GCS
        ddf = dd.read_parquet(predictions_path)

        # Group by request ID and compute the metrics
        ddf = ddf.groupby(id_var)
        ddf = ddf.apply(lambda req: compute_metrics(request=req))
        ddf = ddf.reset_index()
        metrics_df = ddf.compute()

        return metrics_df[0].apply(pd.Series)
