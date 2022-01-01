import nvtabular as nvt

from condorml.preprocess.base_transformer import _merge_nvt_ops


class NVTAnalyzer:
    @staticmethod
    def fit(nvt_ds, input_group_dict, client, dask_working_dir):

        ops = []
        for group_name, vals in input_group_dict.items():
            for _feature in vals:
                ops.append(_feature.nvt_analyzer(dask_working_dir=dask_working_dir))

        nvt_ops = _merge_nvt_ops(ops)
        nvt_workflow = nvt.Workflow(nvt_ops, client=client)
        nvt_workflow.fit(nvt_ds)

        for group_name, vals in input_group_dict.items():
            for _feature in vals:
                pass
        return nvt_workflow
