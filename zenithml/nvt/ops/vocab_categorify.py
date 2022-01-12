from nvtabular.ops import Categorify


class VocabCategorify(Categorify):
    def fit(self, col_selector, ddf):
        import dask

        return dask.delayed(lambda: None)()

    def fit_finalize(self, categories):
        pass
