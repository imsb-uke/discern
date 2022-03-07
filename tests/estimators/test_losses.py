"""Tests for loss functions."""
import itertools

import numpy as np
from tensorflow import keras, test
from tensorflow.python.framework import test_util as tf_test_util  # pylint: disable=no-name-in-module

from discern.estimators import losses as discern_losses
# pylint: disable=missing-function-docstring


@tf_test_util.run_all_in_graph_and_eager_modes
class LnormTest(test.TestCase):
    """ Tests Lnorm """
    def test_lnorm_output_shape(self):
        tests = itertools.product((1, 2, 3, 10), (0, 1), (True, False))
        shape = (6, 7)
        for pnorm, axis, use_root in tests:
            y_a = keras.backend.variable(np.random.random_sample(shape))
            y_b = keras.backend.variable(np.random.random_sample(shape))
            obj = discern_losses.Lnorm(p=pnorm, axis=axis, use_root=use_root)
            objective_output = obj.call(y_a, y_b)
            self.assertListEqual(objective_output.shape.as_list(),
                                 [shape[int(1 - axis)]])

    def test_lnorm_loss(self):
        y_a_np = np.array([[2, 4, 7], [5, 3, 8]])
        y_a = keras.backend.variable(y_a_np)
        y_b_np = np.array([[2, 2, 3], [8, 5, 2]])
        y_b = keras.backend.variable(y_b_np)
        tests = [
            (1, 0, False),
            (1, 1, False),
            (1, 0, True),
            (1, 1, True),
            (2, 0, False),
            (2, 1, False),
            (2, 0, True),
            (2, 1, True),
            (3, 0, False),
            (3, 1, False),
            (3, 0, True),
            (3, 1, True),
        ]
        for pnorm, axis, use_root in tests:
            obj = discern_losses.Lnorm(p=pnorm,
                                     axis=axis,
                                     use_root=use_root,
                                     reduction=keras.losses.Reduction.SUM)
            objective_output = keras.backend.eval(obj.call(y_a, y_b))
            expected = np.linalg.norm(y_a_np - y_b_np, axis=axis, ord=pnorm)
            if not use_root:
                expected = np.power(expected, pnorm)
            self.assertAllClose(objective_output, expected)
            objective_output = keras.backend.eval(obj(y_a, y_b))
            expected = expected.sum()
            self.assertAllClose(objective_output, expected)

    def test_lnorm_serialization(self):
        feature_shape = 10
        y_a = keras.backend.variable(
            np.random.random_sample((20, feature_shape)))
        y_b = keras.backend.variable(
            np.random.random_sample((20, feature_shape)))
        obj = discern_losses.Lnorm(p=2, axis=0, use_root=True)
        config = obj.get_config()
        recovered = discern_losses.Lnorm.from_config(config)
        expected = keras.backend.eval(obj.call(y_a, y_b))
        got = keras.backend.eval(recovered.call(y_a, y_b))
        self.assertAllClose(expected, got)
        obj = discern_losses.Lnorm(p=2, axis=0, use_root=True)
        config = obj.get_config()
        recovered = discern_losses.Lnorm.from_config(config)
        expected = keras.backend.eval(obj.call(y_a, y_b))
        got = keras.backend.eval(recovered.call(y_a, y_b))
        self.assertAllClose(expected, got)


@tf_test_util.run_all_in_graph_and_eager_modes
class DummyLossTest(test.TestCase):
    """ Tests DummyLoss """
    def test_dummyloss_output_shape(self):
        shape = (6, 7)
        y_a = keras.backend.variable(np.random.random_sample(shape))
        y_b = keras.backend.variable(np.random.random_sample(shape))
        obj = discern_losses.DummyLoss()
        objective_output = obj.call(y_a, y_b)
        self.assertListEqual(objective_output.shape.as_list(),
                             y_b.shape.as_list())
        y_b = keras.backend.variable(np.random.rand(8))
        obj = discern_losses.DummyLoss()
        objective_output = obj.call(y_a, y_b)
        self.assertListEqual(objective_output.shape.as_list(),
                             y_b.shape.as_list())

    def test_dummyloss_loss(self):
        tests = [(3), (1, 2), (3, 4, 5)]
        for shape in tests:
            y_a_np = np.random.random_sample(shape)
            y_a = keras.backend.variable(y_a_np)
            y_b_np = np.random.random_sample(shape)
            y_b = keras.backend.variable(y_b_np)
            obj = discern_losses.DummyLoss(reduction=keras.losses.Reduction.SUM)
            objective_output = keras.backend.eval(obj.call(y_a, y_b))
            self.assertAllClose(objective_output, y_b_np)
            objective_output = keras.backend.eval(obj(y_a, y_b))
            expected = y_b_np.sum()
            self.assertAllClose(objective_output, expected)

    def test_dummyloss_serialization(self):
        feature_shape = 10
        y_a = keras.backend.variable(
            np.random.random_sample((20, feature_shape)))
        y_b = keras.backend.variable(
            np.random.random_sample((20, feature_shape)))
        obj = discern_losses.DummyLoss()
        config = obj.get_config()
        recovered = discern_losses.DummyLoss.from_config(config)
        expected = keras.backend.eval(obj.call(y_a, y_b))
        got = keras.backend.eval(recovered.call(y_a, y_b))
        self.assertAllClose(expected, got)


@tf_test_util.run_all_in_graph_and_eager_modes
class HuberLossTest(test.TestCase):
    """ Tests HuberLoss """
    def test_huber_loss_output_shape(self):
        tests = (0.1, 0.5, 1., 10.)
        shape = [6, 7]
        for delta in tests:
            y_a = keras.backend.variable(np.random.random_sample(shape))
            y_b = keras.backend.variable(np.random.random_sample(shape))
            obj = discern_losses.HuberLoss(delta=delta)
            objective_output = obj.call(y_a, y_b)
            self.assertListEqual(objective_output.shape.as_list(), shape[:1])

    def test_huber_loss_loss(self):
        y_a_np = np.array([[2, 4, 7], [5, 3, 8]], dtype=np.float32)
        y_a = keras.backend.variable(y_a_np)
        y_b_np = np.array([[2, 2, 3], [8, 5, 2]], dtype=np.float32)
        y_b = keras.backend.variable(y_b_np)
        tests = (0.0, 0.1, 0.25, 0.5, 1., 3., 5.)
        for delta in tests:
            expected_obj = keras.losses.Huber(delta=delta)
            obj = discern_losses.HuberLoss(delta=delta)
            expected = keras.backend.eval(expected_obj.call(y_a, y_b))
            expected = expected.mean(axis=1)
            got = keras.backend.eval(obj.call(y_a, y_b))
            self.assertAllClose(got, expected)


@tf_test_util.run_all_in_graph_and_eager_modes
class MaskedCrossEntropy(test.TestCase):
    """ Tests MaskedCrossEntropy."""
    def test_output_shape(self):
        tests = itertools.product((0.0, [0, 0, 0, 0], 3, 10), (0, 1e-6, 1e-8))
        shape = (6, 4)
        for zeros, eps in tests:
            y_a = keras.backend.variable(np.random.normal(size=shape))
            y_b = keras.backend.variable(np.random.rand(*shape))
            obj = discern_losses.MaskedCrossEntropy(zeros=zeros, zeros_eps=eps)
            objective_output = obj.call(y_a, y_b)
            self.assertListEqual(objective_output.shape.as_list(), [shape[0]])

    def test_masked_loss(self):
        y_a_np = np.random.randint(0, 2, size=[2, 3]).astype(np.float32)
        y_a = keras.backend.variable(y_a_np)
        y_b_np = np.random.rand(2, 3).astype(np.float32)
        y_b = keras.backend.variable(y_b_np)
        tests = itertools.product(([0, 0, 0], [0, 1, 0], [1, 1, 1], 0, 1),
                                  (0, 1e-6, 1e-8, 1e4))
        for zeros, eps in tests:
            obj = discern_losses.MaskedCrossEntropy(
                zeros=zeros,
                zeros_eps=eps,
                reduction=keras.losses.Reduction.SUM)
            mask = (1 - np.isclose(zeros, y_a_np, atol=eps)).astype(np.float32)
            loss = keras.losses.BinaryCrossentropy().call(mask, y_b)
            expected = keras.backend.eval(loss)
            objective_output = keras.backend.eval(obj.call(y_a, y_b))
            self.assertAllClose(objective_output, expected)
            objective_output = keras.backend.eval(obj(y_a, y_b))
            expected = expected.sum()
            self.assertAllClose(objective_output, expected)

    def test_masked_loss_lower_labelsmoothing(self):
        y_a_np = np.random.randint(0, 2, size=[2, 3]).astype(np.float32)
        y_a = keras.backend.variable(y_a_np)
        y_b_np = np.random.rand(2, 3).astype(np.float32)
        y_b = keras.backend.variable(y_b_np)
        tests = [0.0, 0.05, 0.1, 0.2]
        for labelsmoothing in tests:
            obj = discern_losses.MaskedCrossEntropy(
                zeros=0.0,
                zeros_eps=1e-6,
                lower_label_smoothing=labelsmoothing,
                reduction=keras.losses.Reduction.SUM)
            mask = (1 - np.isclose(0.0, y_a_np, atol=1e-6)).astype(np.float32)
            mask = np.minimum(mask + labelsmoothing, 1.0)
            loss = keras.losses.BinaryCrossentropy().call(mask, y_b)
            expected = keras.backend.eval(loss)
            objective_output = keras.backend.eval(obj.call(y_a, y_b))
            self.assertAllClose(objective_output, expected)
            objective_output = keras.backend.eval(obj(y_a, y_b))
            expected = expected.sum()
            self.assertAllClose(objective_output, expected)

    def test_serialization(self):
        feature_shape = 3
        y_a = keras.backend.variable(
            np.random.random_sample((20, feature_shape)))
        y_b = keras.backend.variable(
            np.random.random_sample((20, feature_shape)))
        obj = discern_losses.MaskedCrossEntropy(zeros=0.0)
        config = obj.get_config()
        recovered = discern_losses.MaskedCrossEntropy.from_config(config)
        self.assertAllClose(obj._zeros, recovered._zeros)  # pylint: disable=protected-access
        self.assertAllClose(obj._zeros_eps, recovered._zeros_eps)  # pylint: disable=protected-access
        expected = keras.backend.eval(obj.call(y_a, y_b))
        got = keras.backend.eval(recovered.call(y_a, y_b))
        self.assertAllClose(expected, got)
