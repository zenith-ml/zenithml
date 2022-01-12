import numpy as np
import pytest

import tensorflow as tf
from zenithml.tf import layers as pa_layers


def _to_tensor(_list, is_list=False):
    return tf.convert_to_tensor([i if is_list else [i] for i in _list])


@pytest.mark.parametrize("inputs", [[1, 2], [1.0, 2.0], [True, False]])
@pytest.mark.parametrize("dimension", [1, 2])
def test_numerical(inputs, dimension):
    layer = pa_layers.NumericalLayer(name="test", dimension=dimension)
    if dimension > 1:
        inputs = [inputs] * dimension

    output = layer(_to_tensor(inputs, is_list=dimension > 1))
    assert output.shape == (2, dimension)
    assert output.dtype == tf.float32


@pytest.mark.parametrize("inputs", [([1, 2], [0.0, 1.0]), ([1.5, 0], [0.5, -1.0])])
def test_minmax_normalizer(inputs):
    inputs, truth = _to_tensor(inputs[0]), np.reshape(np.asarray(inputs[1]), (2, 1))
    layer = pa_layers.MinMaxNormalizeLayer(name="test", max_val=2.0, min_val=1.0)
    layer.build(inputs.shape)
    output = layer(inputs)
    assert output.shape == (2, 1)
    assert output.dtype == tf.float32

    assert np.array_equal(output.numpy(), truth)


@pytest.mark.parametrize(
    "inputs",
    [
        ([1, 2], [(np.log10(1 + 1.0005)), (np.log10(2 + 1.0005))]),
        ([1.5, 0], [(np.log10(1.5 + 1.0005)), (np.log10(0.0 + 1.0005))]),
    ],
)
def test_log_normalizer(inputs):
    inputs, truth = _to_tensor(inputs[0]), np.reshape(np.asarray(inputs[1]), (2, 1))
    layer = pa_layers.LogNormalizeLayer(name="test", log_threshold=1.0005)
    layer.build(inputs.shape)
    output = layer(inputs)
    assert output.shape == (2, 1)
    assert output.dtype == tf.float32

    assert np.isclose(output.numpy(), truth, atol=0.001).all()


# @pytest.mark.skip()
# def test_bucketize(self, test_data, truth):
#     layer = get_layer(pp.BucketizedLayer(input_col="f_buk", bins=2))
#     output = np.squeeze(layer(next(test_data)).numpy())
#     print(output)
#     for i in range(len(truth)):
#         self.assertSequenceAlmostEqual(output[i], truth[i])


@pytest.mark.parametrize(
    "inputs",
    [
        ([1, 2], [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        ([1, 0], [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),
    ],
)
def test_nhotencoder(inputs):
    inputs, truth = _to_tensor(inputs[0]), np.asarray(inputs[1])
    layer = pa_layers.NHotEncodingLayer(name="test", num_buckets=3)
    layer.build(inputs.shape)
    output = layer(inputs)
    assert output.shape == (2, 3)
    assert output.dtype == tf.float32

    assert np.array_equal(output.numpy(), truth)
