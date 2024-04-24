"""
Keras implementation of the dynonet
"""
import os

import pandas as pd
import numpy as np

import keras
import jax

import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error, r2_score


class F16DsSeq():
  """
  Dataset Constructor
  """

  def __init__(self, pd_file, l):
    output_names = [f"Acceleration{idx + 1}" for idx in range(3)]
    self.pd_file = pd_file
    self.seq_len = l

    n = int(np.floor(len(pd_file) / l))

    adv_idx = np.arange(n * l).reshape([n, l])

    u = np.stack([pd_file["Force"].to_numpy()[adv_idx],
                  pd_file["Voltage"].to_numpy()[adv_idx]], axis=-1)

    y = np.stack([pd_file[variable].to_numpy()[adv_idx]
                  for variable in output_names], axis=-1)

    self.u = jax.numpy.asarray(u)
    self.y = jax.numpy.asarray(y)


class MimoLinearDynamicalOperator(keras.layers.Layer):
  """
  Apply MIMO filter
  """

  def __init__(self, out_channels: int = 1, n_b: int = 1, n_a: int = 0):
    super(MimoLinearDynamicalOperator, self).__init__()

    self.out_channels = out_channels
    self.n_b = n_b
    self.n_a = n_a

  def build(self, input_shape):
    self.a_coeff = self.add_weight(
        [self.n_a, self.out_channels],
        initializer="zeros")
    self.b_coeff = self.add_weight(
        [self.n_b, input_shape[-1], self.out_channels])

  def _ar_pply(self, x0, ut):
    xt = keras.ops.einsum("to,tbo->bo",
                          keras.ops.flip(self.a_coeff, axis=0),
                          x0) + ut
    return keras.ops.concatenate([x0[1:, ...], xt[None, ...]]), xt

  def _mimo_apply(self, inputs):
    filtered_input = keras.ops.conv(
        keras.ops.pad(inputs, [[0, 0], [self.n_b - 1, 0], [0, 0]]),
        self.b_coeff,
        padding="valid",
        data_format="channels_last")

    padded_shape = [self.n_a,
                    inputs.shape[0],
                    self.out_channels]

    _, filtered_output = jax.lax.scan(self._ar_pply,
                                      keras.ops.zeros(padded_shape),
                                      keras.ops.transpose(filtered_input,
                                                          [1, 0, 2]))

    return keras.ops.transpose(filtered_output, [1, 0, 2])

  def call(self, inputs: keras.KerasTensor):
    return self._mimo_apply(inputs)


class DynoNet(keras.Model):
  """
  Slides' DynoNet implemented as keras-jax model
  """

  def __init__(self, hidden_size, output_size, n_a, n_b):
    super(DynoNet, self).__init__()
    self.hidden_size = hidden_size
    self.output_size = output_size
    self.n_a = n_a
    self.n_b = n_b

    self.g1 = MimoLinearDynamicalOperator(4, n_b, n_a)
    self.f = keras.Sequential([
        keras.layers.Dense(hidden_size, activation="sigmoid"),
        keras.layers.Dense(3)
    ])
    self.g2 = MimoLinearDynamicalOperator(output_size, n_b, n_a)
    self.g_lin = MimoLinearDynamicalOperator(output_size, n_b, n_a)

  def build(self, input_shape):
    self.g1.build(input_shape)
    self.g2.build(list(input_shape)[:-1] + [3])
    self.g_lin.build(input_shape)

  def call(self, inputs):
    x = self.g1(inputs)
    x = self.f(x)
    x = self.g2(x)

    y = self.g_lin(inputs)

    return x + y


if __name__ == "__main__":
  na = 3
  nb = 25
  test_model = DynoNet(16, 3, na, nb)

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

  seq_len = 500  # Length of the training sub-sequences
  batch_size = 4  # batch size for the data loader

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
  F16DS_train = F16DsSeq(pd_file=dict_ds["train"], l=seq_len)

  print(F16DS_train.u.shape)

  test_model.compile("adam", "mse")
  test_model.fit(F16DS_train.u, F16DS_train.y,
                 batch_size=batch_size, epochs=100)

  F16DS_test = F16DsSeq(pd_file=dict_ds["test"], l=seq_len)
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
