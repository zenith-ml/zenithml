from typing import Optional, Callable, List

import dask.dataframe as dd
import ray
import torch
from dask import delayed
from ray.util import ActorPool

from zenithml.data import ParquetDataset
from zenithml.torch import BaseTorchModel


@ray.remote(num_gpus=1)  # type: ignore
class BatchInference(object):
    def __init__(
        self,
        model_dir,
        predictions_loc,
        preprocessor_loc,
        working_dir,
        worker_id,
        side_cols=None,
        map_fn=None,
        batch_size=10000,
    ):
        self.model = BaseTorchModel.load(model_dir)
        self.preprocessor_loc = preprocessor_loc
        self.working_dir = working_dir
        self.predictions_loc = predictions_loc
        self.side_cols = side_cols
        self.batch_size = batch_size
        self.map_fn = map_fn
        self.idx = 0
        self.worker_id = worker_id

    def predict(self, filename: str, overwrite: bool = True):
        import cudf
        import cupy as cp

        @delayed
        def predict_wrapper(_model, x, y, side_df=None, outcome_var=None):
            df = side_df.copy()
            if outcome_var not in df.columns:
                df[outcome_var] = cp.fromDlpack(torch.utils.dlpack.to_dlpack(y))
            df["score"] = cp.fromDlpack(torch.utils.dlpack.to_dlpack(_model(x)))
            return df

        ds = ParquetDataset(data_loc=filename, working_dir=self.working_dir)
        ds.load_preprocessor(self.preprocessor_loc)
        if self.side_cols is not None:
            outcome_variable = ds.preprocessor.outcome_variable
            side_cols = [col for col in self.side_cols if col != outcome_variable]
            dataloader = ds.to_torch(
                batch_size=self.batch_size,
                transform_data=True,
                side_cols=side_cols,
                map_fn=self.map_fn,
            )

            dd_res = dd.from_delayed(
                [
                    predict_wrapper(
                        self.model,
                        x,
                        y,
                        df,
                        outcome_variable,
                    )
                    for x, y, df in dataloader
                ]
            )
        else:
            dataloader = ds.to_torch(
                batch_size=self.batch_size,
                transform_data=True,
                side_cols=self.side_cols,
                map_fn=self.map_fn,
            )
            dd_res = dd.from_delayed(
                [
                    predict_wrapper(
                        self.model,
                        x,
                        y,
                        cudf.DataFrame(
                            data={ds.preprocessor.outcome_variable: cp.fromDlpack(torch.utils.dlpack.to_dlpack(y))}
                        ),
                    )
                    for x, y in dataloader  # type: ignore
                ]
            )
        self.idx += 1
        dd_res.to_parquet(path=str(self.predictions_loc) + f"/{self.worker_id}-{self.idx}", overwrite=overwrite)


def batch_inference(
    files,
    parallelization: int,
    model_loc: str,
    predictions_loc: str,
    preprocessor_loc: str,
    working_dir: str,
    side_cols: Optional[List[str]] = None,
    map_fn: Optional[Callable] = None,
):
    pool = ActorPool(
        [
            BatchInference.remote(  # type: ignore
                model_dir=model_loc,
                preprocessor_loc=preprocessor_loc,
                predictions_loc=predictions_loc,
                working_dir=working_dir,
                side_cols=side_cols,
                map_fn=map_fn,
                worker_id=w,
            )
            for w in range(parallelization)
        ]
    )
    _ = [_ for _ in pool.map(lambda inferrer, f: inferrer.predict.remote(f), files)]
