import numpy as np
import tensorflow as tf
from nvtabular.dispatch import annotate, is_series_or_dataframe_object, HAS_GPU
from nvtabular.loader.tensorflow import KerasSequenceLoader
from toolz.itertoolz import sliding_window

from condorml.preprocess import Preprocessor

from_dlpack = tf.experimental.dlpack.from_dlpack


def df_split(chunk, split_idx):
    if HAS_GPU:
        return [chunk.iloc[start:stop] for start, stop in sliding_window(2, np.cumsum([0] + split_idx))]
    else:
        return [_df for _df in np.split(chunk, split_idx) if not _df.empty]


class NVTKerasDataset(KerasSequenceLoader):
    @classmethod
    def from_preprocessor(
        cls,
        paths_or_dataset,
        batch_size,
        preprocessor: Preprocessor,
        side_cols=None,
        engine=None,
        shuffle=True,
        seed_fn=None,
        buffer_size=0.1,
        device=None,
        parts_per_chunk=1,
        reader_kwargs=None,
        global_size=None,
        global_rank=None,
        drop_last=False,
        sparse_as_dense=True,
    ):

        cont_cols, cat_cols, sparse_names, sparse_max = preprocessor.get_nvt_cols()
        # TODO: Switch from RaggedTensor to SparseTensor in EmbeddingLayer and fix this
        cat_cols += sparse_names
        label_names = preprocessor.outcome_variable
        return cls(
            paths_or_dataset=paths_or_dataset,
            batch_size=batch_size,
            label_names=[label_names] if isinstance(label_names, str) else label_names,
            cat_names=cat_cols,
            cont_names=cont_cols,
            side_cols=side_cols,
            engine=engine,
            shuffle=shuffle,
            seed_fn=seed_fn,
            buffer_size=buffer_size,
            device=device,
            parts_per_chunk=parts_per_chunk,
            reader_kwargs=reader_kwargs,
            global_size=global_size,
            global_rank=global_rank,
            drop_last=drop_last,
            sparse_names=None,
            sparse_max=None,
            sparse_as_dense=False,
        )

    def __init__(
        self,
        paths_or_dataset,
        batch_size,
        label_names=None,
        cat_names=None,
        cont_names=None,
        side_cols=None,
        engine=None,
        shuffle=True,
        seed_fn=None,
        buffer_size=0.1,
        device=None,
        parts_per_chunk=1,
        reader_kwargs=None,
        global_size=None,
        global_rank=None,
        drop_last=False,
        sparse_names=None,
        sparse_max=None,
        sparse_as_dense=False,
    ):
        self.side_cols = side_cols
        super().__init__(
            paths_or_dataset=paths_or_dataset,
            batch_size=batch_size,
            label_names=label_names,
            feature_columns=None,
            cat_names=cat_names,
            cont_names=cont_names,
            engine=engine,
            shuffle=shuffle,
            seed_fn=seed_fn,
            buffer_size=buffer_size,
            device=device,
            parts_per_chunk=parts_per_chunk,
            reader_kwargs=reader_kwargs,
            global_size=global_size,
            global_rank=global_rank,
            drop_last=drop_last,
            sparse_names=sparse_names,
            sparse_max=sparse_max,
            sparse_as_dense=sparse_as_dense,
        )

    @annotate("make_tensors", color="darkgreen", domain="nvt_python")
    def make_tensors(self, gdf, use_nnz=False):
        split_idx = self._get_segment_lengths(len(gdf))
        # map from big chunk to framework-specific tensors
        chunks = self._create_tensors(gdf)

        # if we have any offsets, calculate nnzs up front
        if len(chunks) == 5:
            offsets = chunks[-1]
            if use_nnz:
                nnzs = offsets[1:] - offsets[:-1]
            chunks = chunks[:-1]

        # split them into batches and map to the framework-specific output format
        batches = [[] for _ in range(len(split_idx))]
        offset_idx = 0
        for i, chunk in enumerate(chunks):
            lists = None
            if isinstance(chunk, tuple):
                chunk, lists = chunk

            if len(split_idx) > 1 and chunk is not None:
                if is_series_or_dataframe_object(chunk):
                    chunk = df_split(chunk, split_idx)
                else:
                    chunk = self._split_fn(chunk, split_idx)
            else:
                chunk = [chunk for _ in split_idx]

            if lists is not None:
                num_list_columns = len(lists)

                # grab the set of offsets and nnzs corresponding to
                # the list columns from this chunk
                chunk_offsets = offsets[:, offset_idx : offset_idx + num_list_columns]
                if use_nnz:
                    chunk_nnzs = nnzs[:, offset_idx : offset_idx + num_list_columns]
                offset_idx += num_list_columns

                # split them into batches, including an extra 1 on the offsets
                # so we know how long the very last element is
                batch_offsets = self._split_fn(chunk_offsets, split_idx + [1])
                if use_nnz and len(split_idx) > 1:
                    batch_nnzs = self._split_fn(chunk_nnzs, split_idx)
                elif use_nnz:
                    batch_nnzs = [chunk_nnzs]
                else:
                    batch_nnzs = [None] * (len(batch_offsets) - 1)

                # group all these indices together and iterate through
                # them in batches to grab the proper elements from each
                # values tensor
                chunk = zip(chunk, batch_offsets[:-1], batch_offsets[1:], batch_nnzs)

            for n, c in enumerate(chunk):
                if isinstance(c, tuple):
                    c, off0s, off1s, _nnzs = c
                    offsets_split_idx = [1 for _ in range(num_list_columns)]
                    off0s = self._split_fn(off0s, offsets_split_idx, axis=1)
                    off1s = self._split_fn(off1s, offsets_split_idx, axis=1)
                    if use_nnz:
                        _nnzs = self._split_fn(_nnzs, offsets_split_idx, axis=1)

                    # TODO: does this need to be ordereddict?
                    batch_lists = {}
                    for k, (column_name, values) in enumerate(lists.items()):
                        off0, off1 = off0s[k], off1s[k]
                        if use_nnz:
                            nnz = _nnzs[k]

                        # need to grab scalars for TF case
                        if len(off0.shape) == 1:
                            start, stop = off0[0], off1[0]
                        elif len(off0.shape) == 2:
                            start, stop = off0[0, 0], off1[0, 0]
                        else:
                            print(off0, off1)
                            raise ValueError

                        value = values[start:stop]
                        index = off0 - start if not use_nnz else nnz
                        batch_lists[column_name] = (value, index)
                    c = (c, batch_lists)

                batches[n].append(c)
        return [self._handle_tensors(*batch) for batch in batches]

    @annotate("_create_tensors", color="darkgreen", domain="nvt_python")
    def _create_tensors(self, gdf):
        """
        Breaks a dataframe down into the relevant
        categorical, continuous, and label tensors.
        Can be overrideen
        """

        if self.side_cols is not None:
            gdf_i = gdf[self.side_cols]
            gdf.drop(columns=self.side_cols, inplace=True)
        else:
            gdf_i = None
        tensors = [gdf_i]
        tensors.extend(super()._create_tensors(gdf))
        return tensors

    def _handle_tensors(self, side_df, cats, conts, labels):
        to_return = super()._handle_tensors(cats, conts, labels)

        def list_col_as_sparse(k, v):
            if isinstance(v, tuple) or isinstance(v, list):
                values = v[0][:, 0]
                row_lengths = v[1][:, 0]
                return tf.RaggedTensor.from_row_lengths(values, row_lengths, name=f"{k}_ragged").to_sparse()
            return v

        # to_return = {k: list_col_as_sparse(k, v) for k, v in to_return[0].items()}, to_return[1]

        return (*to_return, side_df) if side_df is not None else to_return
