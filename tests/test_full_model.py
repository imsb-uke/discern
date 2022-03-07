"""Tests for the full model."""
import numpy as np
import tensorflow as tf
import pytest


def _compute_loss(waemodel, predicted, trueoutput):
    targets = {
        "decoder_counts": trueoutput,
        "decoder_dropouts": trueoutput,
        "sigma_regularization": waemodel._targets[2],  # pylint: disable=protected-access
        "mmdpp": waemodel._targets[3],  # pylint: disable=protected-access
    }
    losses = {
        k: waemodel.loss[k](target, predicted[k])
        for k, target in targets.items()
    }

    weighted_losses = {
        k: waemodel.loss_weights[k] * loss
        for k, loss in losses.items()
    }
    return sum(weighted_losses.values())


@pytest.mark.forked
def test_model_gradients(default_model):
    """Test gradients of the model."""
    model = default_model
    n_genes = 2000
    n_labels = 2
    model.build_model(n_genes=n_genes, n_labels=n_labels, scale=13276)
    train_data = np.full((5, n_genes), fill_value=100, dtype=np.float32)
    labels = np.zeros((5, n_labels), dtype=np.float32)
    labels[:, 1] = 1
    with tf.GradientTape() as tape:
        predicted = model.wae_model((labels, labels, train_data),
                                    training=True)
        loss = _compute_loss(model.wae_model, predicted, train_data)
        weights = model.wae_model.trainable_weights
        grads = tape.gradient(loss, weights)
    for variable, grad in zip(model.wae_model.trainable_weights, grads):
        assert np.all(np.isfinite(grad))
        if variable.name.startswith("conditioning_layer"):
            np.testing.assert_array_equal(grad[0], 0.)


@pytest.mark.slow
@pytest.mark.forked
def test_model_overfit(default_model, anndata_file):
    """Test overfitting model, if it is able to reproduce cells."""
    model = default_model

    train_data = anndata_file(3)
    model.build_model(n_genes=train_data.var_names.size,
                      n_labels=2,
                      scale=train_data.shape[0])
    train_data.X = np.ones_like(train_data.X)
    labels = tf.one_hot(train_data.obs.batch.cat.codes.values.astype(int),
                        2,
                        dtype=tf.float32)
    model.wae_model.fit(x=(labels, labels, train_data.X),
                        y=(train_data.X, train_data.X),
                        epochs=200,
                        verbose=0)

    got = model.wae_model.predict({
        "batch_input_enc": labels,
        "batch_input_dec": labels,
        "input_data": train_data.X,
    })[0]

    got[got < 0.1] = 0
    n_close = np.count_nonzero(np.isclose(got, train_data.X, atol=0.2), axis=1)
    freq = n_close / train_data.shape[1]
    np.testing.assert_array_compare(np.greater, freq, 0.98)
