[![Tests](https://github.com/zenith-ml/zenithml/actions/workflows/tests.yml/badge.svg)](https://github.com/zenith-ml/zenithml/actions/workflows/tests.yml)
[![Docker](https://github.com/zenith-ml/zenithml/actions/workflows/docker.yml/badge.svg)](https://github.com/zenith-ml/zenithml/actions/workflows/docker.yml)
[![Docs](https://github.com/zenith-ml/zenithml/actions/workflows/docs.yml/badge.svg)](https://github.com/zenith-ml/zenithml/actions/workflows/docs.yml)
[![codecov](https://codecov.io/gh/zenith-ml/zenithml/branch/main/graph/badge.svg?token=7JQCAEKFRH)](https://codecov.io/gh/zenith-ml/zenithml)
[![PyPI version](https://badge.fury.io/py/zenith.svg)](https://badge.fury.io/py/zenithml)

# Zenith ML
Zenith ML is provides an end-to-end framework for
developing machine learning models. 
It provides feature processors, data loaders,
ML models for recommendations and search applications, 
and evaluation methodologies for rapid experimentation at scale.
Zenith ML is framework-agnostic and current has built-in
support for PyTorch and Keras/Tensorflow. 

The goal of the library is to provide a simple interface for 
training recommendation & search models.

### Feature Preprocessing & Model Training

```python
from pathlib import Path
import torch
import zenithml as zm

working_dir = "./"
model_loc=Path(working_dir) / "model"
preprocessor_loc=Path(working_dir) / "preprocessor"
predictions_loc=Path(working_dir) / "predictions"

# Data Loading & Feature Transformation
data = zm.data.load_dataset(name="movielens").train
data.analyze_transform(preprocessor_loc=preprocessor_loc)


# Model Training
torch_ds = data.to_torch(batch_size=512)
model = zm.torch.recsys.LinearClassifier(preprocess_layers=data.preprocessor.get_preprocess_layers())
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
model.trainer(
    ds=torch_ds,
    model=model,
    optimizer=optimizer,
    config={},
    model_dir=model_loc,
)
```

### Batch Prediction & Evaluation

```python
# Batch Inference
test_data =zm.data.load_dataset(name="movielens").test 
zm.torch.batch_inference(
        files=zm.utils.fs.ls(test_data.dataset_loc),
        parallelization=1,
        model_loc=model_loc,
        preprocessor_loc=preprocessor_loc,
        predictions_loc=predictions_loc,
        working_dir=working_dir
)


# Evaluation
metric_df = zm.metrics.compute_ranking_metrics(
    predictions_path=predictions_loc,
    label_cols=["ratings"],
    id_var="user_id",
    metric_fns={"mrr": zm.metrics.mrr},
    score_col="score",
)


```

The library intends to provide a wrapper around NVIDA's 
[NVTabular](https://github.com/NVIDIA-Merlin/NVTabular) for 
feature transformation at scale using GPU. In addition, 
it also enables users to add custom analyzers & transformers. 
And, also provides examples for infra setup and parallel 
processing using [Ray](https://www.ray.io/).


