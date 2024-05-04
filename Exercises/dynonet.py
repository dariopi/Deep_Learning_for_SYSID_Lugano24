"""
Keras implementation of the dynonet
"""
import os

import pandas as pd
import numpy as np

import keras
from keras import layers, ops
from keras import KerasTensor as Tensor

import jax
from jax import lax

import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error, r2_score


class F16DsSeq():
  """
  Dataset Constructor
  """

  def __init__(self, pd_file, l, ar_order, use_context=False):
    output_names = [f"Acceleration{idx + 1}" for idx in range(3)]
    self.pd_file = pd_file
    self.seq_len = l

    if use_context:
      n = int(np.floor((len(pd_file) - ar_order) / l))
      adv_idx = np.arange(n * l).reshape([n, l]) + ar_order
    else:
      n = int(np.floor(len(pd_file) / l))
      adv_idx = np.arange(n * l).reshape([n, l])

    x = np.stack([pd_file["Force"].to_numpy()[adv_idx],
                  pd_file["Voltage"].to_numpy()[adv_idx]], axis=-1)

    y = np.stack([pd_file[variable].to_numpy()[adv_idx]
                  for variable in output_names], axis=-1)

    if use_context:
      x_0 = np.stack([pd_file[variable].to_numpy()[
          adv_idx[:, :ar_order] - ar_order]
          for variable in output_names], axis=-1)
      self.u = [jax.numpy.asarray(x), jax.numpy.asarray(x_0)]
    else:
      self.u = [jax.numpy.asarray(x), jax.numpy.zeros_like(y)[:, :ar_order, :]]
    self.y = jax.numpy.asarray(y)


@keras.saving.register_keras_serializable(package="dynonet")
class MimoLinearDynamicalOperator(layers.Layer):
  """
  Apply MIMO filter
  """

  def __init__(self, out_channels: int = 1,
               n_b: int = 0, n_a: int = 0,
               **kwargs):
    super(MimoLinearDynamicalOperator, self).__init__(**kwargs)

    self.out_channels = out_channels
    self.n_b = n_b
    self.n_a = n_a

  def build(self, input_shape):
    self.a_coeff = self.add_weight(
        [self.n_a, self.out_channels],
        initializer="zeros")
    self.b_coeff = self.add_weight(
        [self.n_b + 1, input_shape[0][-1], self.out_channels])

  def _a_ss(self) -> Tensor:
    return ops.stack([
        ops.pad(self.a_coeff[idx:, :], [[idx, 0], [0, 0]])
        for idx in range(self.n_a)
    ], axis=0)

  def call(self, inputs: Tensor):
    u, x0 = inputs

    # Calculate effect of input over output
    b_u = ops.conv(
        ops.pad(u, [[0, 0], [self.n_b, 0], [0, 0]]),
        self.b_coeff, padding="valid", data_format="channels_last")

    # Early return if non-ar
    if self.n_a == 0:
      return b_u

    # Apply effect of first iteration
    b_u = ops.concatenate([
        (b_u[:, 0, :] + ops.einsum("to,bto->bo",
                                   ops.flip(self.a_coeff, axis=0),
                                   x0))[:, None, :],
        b_u[:, 1:, :]
    ], axis=1)

    # Turn time-shifts into states + context
    b_u = ops.stack([
        ops.concatenate([
            x0[:, self.n_a - idx:, :],
            b_u[:, idx:, :]
        ], axis=1)
        for idx in range(self.n_a)
    ], axis=1)

    # Get time-shifted ar coefficients
    a_ss = self._a_ss()

    def a_apply(x0, b_u):
      return ops.einsum("sno,bnto->bsto", a_ss, x0) + b_u

    y = lax.associative_scan(a_apply, b_u, axis=2)

    return y[:, 0, ...]


@keras.saving.register_keras_serializable(package="dynonet")
class DynoNet(keras.Model):
  """
  Slides' DynoNet implemented as keras-jax model
  """

  def __init__(self, hidden_size, encoding_depth, output_size,
               n_a, n_b,
               **kwargs):
    super(DynoNet, self).__init__(**kwargs)
    self.hidden_size = hidden_size
    self.encoding_depth = encoding_depth
    self.output_size = output_size
    self.n_a = n_a
    self.n_b = n_b

    self.g1 = MimoLinearDynamicalOperator(encoding_depth, n_b, n_a)
    self.f = keras.Sequential([
        layers.Dense(hidden_size, activation="sigmoid"),
        layers.Dense(encoding_depth)
    ])
    self.g2 = MimoLinearDynamicalOperator(output_size, n_b, n_a)
    self.g_lin = MimoLinearDynamicalOperator(output_size, n_b, n_a)

  def get_config(self):
    config = super().get_config()
    config.update({
        "hidden_size": self.hidden_size,
        "encoding_depth": self.encoding_depth,
        "output_size": self.output_size,
        "n_a": self.n_a,
        "n_b": self.n_b
    })
    return config

  def build(self, input_shape):
    x = keras.Input([None, input_shape[0][-1]])
    x0_hidden = keras.Input([self.n_a, self.encoding_depth])
    x0 = keras.Input([self.n_a, self.output_size])

    x1 = self.g1([x, x0_hidden])
    xf = self.f(x1)
    self.g2([xf, x0])
    self.g_lin([x, x0])

  def call(self, inputs):
    x, x0 = inputs

    x0_shape = list(x0.shape)
    x0_shape[-2] = self.n_a
    x0_shape[-1] = self.encoding_depth
    x0_aux = ops.zeros(x0_shape)

    x = self.g1([x, x0_aux])

    x = self.f(x)
    x = self.g2([x, x0])

    y = self.g_lin(inputs)

    return x + y


if __name__ == "__main__":
  na = 50
  nb = 100
  test_model = DynoNet(32, 8, 3, na, nb)

  folder = os.path.join("..", "Datasets", "F16")
  f_train_ds = os.path.join(folder, "F16Data_SineSw_Level3.csv")
  f_test_ds = os.path.join(folder, "F16Data_SineSw_Level4_Validation.csv")

  # create dictionary with training and test dataset
  dict_ds = {"train": [], "test": [], }
  dict_ds["train"] = pd.read_csv(f_train_ds)
  dict_ds["test"] = pd.read_csv(f_test_ds)

  # data normalization
  ds_mean = dict_ds["train"].mean()
  ds_std = dict_ds["train"].std()
  dict_ds["train"] = (dict_ds["train"] - ds_mean) / ds_std
  dict_ds["test"] = (dict_ds["test"] - ds_mean) / ds_std

  seq_len = 1000  # Length of the training sub-sequences
  batch_size = 8  # batch size for the data loader

  folder = os.path.join("..", "Datasets", "F16")
  f_train_ds = os.path.join(folder, "F16Data_SineSw_Level3.csv")
  f_test_ds = os.path.join(folder, "F16Data_SineSw_Level4_Validation.csv")

  dict_ds = {"train": [], "test": [], }
  dict_ds["train"] = pd.read_csv(f_train_ds)
  dict_ds["test"] = pd.read_csv(f_test_ds)

  ds_mean = dict_ds["train"].mean()
  ds_std = dict_ds["train"].std()
  dict_ds["train"] = (dict_ds["train"] - ds_mean) / ds_std
  dict_ds["test"] = (dict_ds["test"] - ds_mean) / ds_std

  # Create instance of the class F16DS_seq and plot shape of inputs and outputs
  F16DS_train = F16DsSeq(pd_file=dict_ds["train"],
                         l=seq_len, ar_order=na,
                         use_context=True)

  test_model.compile("adam", "mse")
  test_model.fit(F16DS_train.u,
                 F16DS_train.y,
                 batch_size=batch_size, epochs=32)

  print(f"Linear AR weights:\n{
      np.mean(np.abs(np.array(test_model.g_lin.a_coeff)), axis=-1)}")
  print(f"Non-Linear AR weights 1:\n{
      np.mean(np.abs(np.array(test_model.g1.a_coeff)), axis=-1)}")
  print(f"Non-Linear AR weights 2:\n{
      np.mean(np.abs(np.array(test_model.g2.a_coeff)), axis=-1)}")

  test_model.save("./dynonet.keras")
  test_model = keras.saving.load_model("./dynonet.keras")

  F16DS_test = F16DsSeq(pd_file=dict_ds["test"],
                        l=seq_len, ar_order=na,
                        use_context=True)
  y_true = np.array(F16DS_test.y)
  plot_idx = np.argmax(
      np.mean(y_true - np.mean(y_true, 1, keepdims=True), axis=(1, 2)))

  y_pred = test_model.predict(F16DS_test.u, batch_size=batch_size)

  fig, ax = plt.subplots(3, 1, sharex=True, figsize=(10, 5))

  for idx in range(3):
    ax[idx].plot(y_true[plot_idx, :, idx].reshape(-1),
                 label="Reference")
    ax[idx].plot(y_pred[plot_idx, :, idx].reshape(-1),
                 label="Prediction")
    ax[idx].legend()
    ax[idx].set_title(f"Acceleration {idx + 1}")

  print(f"RMSE: {root_mean_squared_error(y_true.reshape([-1, 3]),
                                         y_pred.reshape([-1, 3]))}")
  print(f"R2: {r2_score(y_true.reshape([-1, 3]),
                        y_pred.reshape([-1, 3]))}")
  fig.savefig("./keras_result.png")
