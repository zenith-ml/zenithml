from nvtabular import Workflow, Dataset

from nvtabular.workflow.workflow import _transform_ddf


class CustomNVTWorkflow(Workflow):
    def transform(self, dataset: Dataset, additional_cols=None) -> Dataset:
        self._clear_worker_cache()

        if not self.output_schema:
            self.fit_schema(dataset.schema)

        if additional_cols:
            ddf = dataset.to_ddf(columns=self._input_columns() + additional_cols)
        else:
            ddf = dataset.to_ddf(columns=self._input_columns())

        return Dataset(
            _transform_ddf(ddf, self.output_node, self.output_dtypes, additional_cols),
            client="auto",
            cpu=dataset.cpu,
            base_dataset=dataset.base_dataset,
            schema=self.output_schema,
        )
