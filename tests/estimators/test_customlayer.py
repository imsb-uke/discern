"""Tests for custom layers."""
import numpy as np
import tensorflow as tf
from tensorflow import keras
# pylint: disable=no-name-in-module
from tensorflow.python.framework import test_util as tf_test_util
from tensorflow.python.keras import keras_parameterized, testing_utils

# pylint: enable=no-name-in-module
from discern.estimators import customlayers


class SigmaRegularizationTest(keras_parameterized.TestCase):
    """Tests for SigmaRegularization."""

    #
    @keras_parameterized.run_all_keras_modes
    def test_basic_sigma_regularization(self):  # pylint: disable=no-self-use
        """Test layer creation."""
        testing_utils.layer_test(customlayers.SigmaRegularization,
                                 kwargs={},
                                 input_shape=(3, 4))

    @tf_test_util.run_in_graph_and_eager_modes
    def test_sigma_regularization_weights(self):
        """Test weight creation."""
        layer = customlayers.SigmaRegularization()
        layer.build((None, 3))
        self.assertEqual(len(layer.trainable_weights), 0)
        self.assertEqual(len(layer.weights), 0)

    @keras_parameterized.run_all_keras_modes
    def test_sigma_regularization(self):
        """Test output."""
        with self.session(use_gpu=True):
            model = keras.models.Sequential()
            reg = customlayers.SigmaRegularization()
            model.add(reg)
            should = testing_utils.should_run_tf_function()  # pylint: disable=no-member
            model.compile(loss=lambda x, y: tf.reduce_mean(x - y),
                          optimizer='adam',
                          run_eagerly=testing_utils.should_run_eagerly(),
                          experimental_run_tf_function=should)

            # centered on 5.0, variance 10.0
            xval = np.random.normal(loc=5.0, scale=10.0, size=(1000, 3))
            yval = np.random.rand(1000)
            out = model.predict(xval)
            model.fit(xval, yval, verbose=0, epochs=4, batch_size=100)
            expected = 3 * np.abs(xval).mean(axis=1)
            self.assertEqual(out.shape, expected.shape)
            self.assertAllClose(out, expected, rtol=2e-0)

    @keras_parameterized.run_all_keras_modes
    def test_sigma_regularization_gradients(self):
        """Test gradients."""
        xval = np.random.normal(loc=5.0, scale=10.0,
                                size=(1000, 3)).astype(dtype=np.float32)
        add_layer = customlayers.SigmaRegularization()
        add_layer.build((3, ))
        theoretical, numerical = tf.test.compute_gradient(add_layer, [xval])
        self.assertTrue(np.all(np.isfinite(theoretical)))
        self.assertTrue(np.all(np.isfinite(numerical)))


class GaussianReparametrizationTest(keras_parameterized.TestCase):
    """Tests for GaussianReparametrization."""

    #
    @tf_test_util.run_in_graph_and_eager_modes
    def test_gaussian_reparametrization_weights(self):
        """Test weight creation."""
        layer = customlayers.GaussianReparametrization()
        layer.build([(None, 3), (None, 3)])
        self.assertEqual(len(layer.trainable_weights), 0)
        self.assertEqual(len(layer.weights), 0)

    @keras_parameterized.run_all_keras_modes
    def test_gaussian_reparametrization(self):
        """Test output."""
        input1 = keras.layers.Input(shape=(4))
        input2 = keras.layers.Input(shape=(4))
        add_layer = customlayers.GaussianReparametrization()
        output = add_layer([input1, input2])
        self.assertListEqual(output.shape.as_list(), [None, 4])
        model = keras.models.Model([input1, input2], output)
        model.run_eagerly = testing_utils.should_run_eagerly()
        should = testing_utils.should_run_tf_function()  # pylint: disable=no-member
        model._experimental_run_tf_function = should  # pylint: disable=protected-access
        mean = np.random.rand(2, 4)
        sigma = np.random.rand(2, 4)
        out = model.predict([mean, sigma])
        self.assertEqual(out.shape, (2, 4))
        self.assertAllClose(out, mean, rtol=1e-15)
        model.compile(loss="mse", optimizer="adam")
        history = model.fit([mean, sigma], mean, epochs=4, batch_size=2)
        loss = sum(history.history["loss"])
        self.assertTrue(loss > 0.2)
        self.assertEqual(
            add_layer.compute_mask([input1, input2], [None, None]), None)
        with self.assertRaisesRegex(
                TypeError,
                'does not support masking, but was passed an input_mask'):
            add_layer.compute_mask([input1, input2], [mean, sigma])

    @keras_parameterized.run_all_keras_modes
    def test_gaussian_reparametrization_serialization(self):
        """Test serialization."""
        input1 = keras.layers.Input(shape=(4, ))
        input2 = keras.layers.Input(shape=(4, ))
        add_layer = customlayers.GaussianReparametrization()
        output = add_layer([input1, input2])
        model = keras.models.Model([input1, input2], output)
        model_config = model.get_config()
        recovered_model = keras.models.Model.from_config(model_config)
        mean = np.random.rand(2, 4)
        sigma = np.random.rand(2, 4)
        actual_output = model.predict([mean, sigma])
        output = recovered_model.predict([mean, sigma])
        self.assertAllClose(output, actual_output, rtol=1e-3, atol=1e-6)

    @keras_parameterized.run_all_keras_modes
    def test_gaussian_reparametrization_gradients(self):
        """Test gradients."""
        mean = np.random.rand(2, 4).astype(dtype=np.float32)
        sigma = np.random.rand(2, 4).astype(dtype=np.float32)
        add_layer = customlayers.GaussianReparametrization()
        add_layer.build(((4, ), (4, )))
        func = lambda x, y: add_layer((x, y), training=True)
        for theoretical, numerical in tf.test.compute_gradient(
                func, [mean, sigma]):
            self.assertTrue(np.all(np.isfinite(theoretical)))
            self.assertTrue(np.all(np.isfinite(numerical)))


class MMDPPTest(keras_parameterized.TestCase):
    """Tests for MMPPP."""

    #
    @tf_test_util.run_in_graph_and_eager_modes
    def test_mmdpp_weights(self):
        """Test weights creation."""
        layer = customlayers.MMDPP(scale=100)
        layer.build([(None, 3), (None, 3)])
        self.assertEqual(len(layer.trainable_weights), 0)
        self.assertEqual(len(layer.weights), 0)

    @keras_parameterized.run_all_keras_modes
    def test_mmdpp(self):
        """Test full run."""
        input1 = keras.layers.Input(shape=(4, ))
        input2 = keras.layers.Input(shape=(4, ))
        add_layer = customlayers.MMDPP(scale=100)
        output = add_layer([input1, input2])
        self.assertListEqual(output.shape.as_list(), [None])
        model = keras.models.Model([input1, input2], output)
        model.run_eagerly = testing_utils.should_run_eagerly()
        should = testing_utils.should_run_tf_function()  # pylint: disable=no-member
        model._experimental_run_tf_function = should  # pylint: disable=protected-access
        mean = np.random.rand(2, 4)
        sigma = np.random.rand(2, 4)
        out = model.predict([mean, sigma])
        self.assertEqual(out.shape, (2, ))
        self.assertEqual(
            add_layer.compute_mask([input1, input2], [None, None]), None)
        with self.assertRaisesRegex(
                TypeError,
                'does not support masking, but was passed an input_mask'):
            add_layer.compute_mask([input1, input2], [mean, sigma])
        # Testing of produced values in test_mmdpp!!

    @keras_parameterized.run_all_keras_modes
    def test_mmdpp_serialization(self):
        """Test serialization."""
        input1 = keras.layers.Input(shape=(4, ))
        input2 = keras.layers.Input(shape=(4, ))
        add_layer = customlayers.MMDPP(scale=100)
        output = add_layer([input1, input2])
        model = keras.models.Model([input1, input2], output)
        model_config = model.get_config()
        recovered_model = keras.models.Model.from_config(model_config)
        mean = np.random.rand(2, 4)
        sigma = np.random.rand(2, 4)
        actual_output = model.predict([mean, sigma])
        output = recovered_model.predict([mean, sigma])
        self.assertAllClose(output, actual_output, rtol=1e-3, atol=1e-6)

    @keras_parameterized.run_all_keras_modes
    def test_mmdpp_gradients(self):
        """Test gradients."""
        mean = np.random.rand(2, 4).astype(dtype=np.float32)
        sigma = np.random.rand(2, 4).astype(dtype=np.float32)
        add_layer = customlayers.MMDPP(scale=100)
        add_layer.build(((4, ), (4, )))
        func = lambda x, y: add_layer((x, y))
        for theoretical, numerical in tf.test.compute_gradient(
                func, [mean, sigma]):
            self.assertTrue(np.all(np.isfinite(theoretical)))
            self.assertTrue(np.all(np.isfinite(numerical)))
