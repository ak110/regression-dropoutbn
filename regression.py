#!/usr/bin/env python3
"""CNNで回帰をするコード。"""
import argparse
import functools
import pathlib
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

batch_size = 16
steps_per_epoch = 16
epochs = 100


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--drop", action="store_true")
    parser.add_argument("--bn", action="store_true")
    args = parser.parse_args()

    # 試しに画像出力
    if args.check:
        for i, (img, r) in enumerate(create_dataset().unbatch().take(10)):
            save_path = pathlib.Path(f"check-img{i:02d}.png")
            cv2.imwrite(str(save_path), img.numpy())
            print(f"{save_path}: {r.numpy()=:.0f}")

    # CNN
    conv2d = functools.partial(
        tf.keras.layers.Conv2D,
        kernel_size=3,
        padding="same",
        use_bias=not args.bn,
        kernel_initializer="he_uniform",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
    )
    if args.bn:
        bn = functools.partial(
            tf.keras.layers.BatchNormalization,
            gamma_regularizer=tf.keras.regularizers.l2(1e-4),
        )
    else:
        bn = functools.partial(tf.keras.layers.Activation, None)
    act = functools.partial(tf.keras.layers.Activation, "relu")
    inputs = [tf.keras.layers.Input((256, 256), dtype="uint8")]
    x = inputs[0]
    x = tf.reshape(x, (-1, 256, 256, 1))
    x = tf.cast(x, tf.float32)
    x = tf.keras.applications.imagenet_utils.preprocess_input(x, mode="tf")
    x = act()(bn()(conv2d(32)(x)))
    x = tf.keras.layers.MaxPooling2D()(x)  # 128
    x = act()(bn()(conv2d(64)(x)))
    x = tf.keras.layers.MaxPooling2D()(x)  # 64
    x = act()(bn()(conv2d(128)(x)))
    x = tf.keras.layers.MaxPooling2D()(x)  # 32
    x = act()(bn()(conv2d(256)(x)))
    x = tf.keras.layers.MaxPooling2D()(x)  # 16
    x = act()(bn()(conv2d(512)(x)))
    x = tf.keras.layers.MaxPooling2D()(x)  # 8
    x = act()(bn()(conv2d(512)(x)))
    if args.drop:
        x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(1)(x)
    x = tf.squeeze(x, axis=-1)
    x = x * 14 + 50  # シフト・スケーリング
    model = tf.keras.models.Model(inputs, x)
    optimizer = tf.keras.optimizers.Adam(
        tf.keras.experimental.CosineDecay(1e-3, decay_steps=steps_per_epoch * epochs)
    )
    model.compile(
        loss="mse",
        optimizer=optimizer,
        metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse")],
    )
    model.summary()

    # 学習
    history = model.fit(
        create_dataset(), epochs=epochs, validation_data=create_dataset()
    )

    # plot
    fig = plt.figure(figsize=(4.0, 3.0))
    ax = fig.add_subplot()
    ax.plot(history.history["rmse"])
    ax.plot(history.history["val_rmse"])
    ax.set_ylim([0.0, 25.0])  # スケールは固定
    ax.set_title("Model RMSE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("RMSE")
    ax.legend(["Train", "Test"], loc="upper right")
    ax.get_figure().savefig(f"history_drop={args.drop}_bn={args.bn}.png")


def create_dataset():
    def gen(i):
        del i  # とりあえず使わない
        # タスクは少し簡単にする
        img = np.zeros((256, 256), dtype=np.uint8)
        cx = random.randrange(96, 256 - 96)
        cy = random.randrange(96, 256 - 96)
        r = random.randrange(25, 75 + 1)
        cv2.circle(img, center=(cx, cy), radius=r, color=(255, 255, 255), thickness=-1)
        return img, np.float32(r)

    ds = tf.data.Dataset.from_tensor_slices(np.arange(steps_per_epoch * batch_size))
    ds = ds.map(
        lambda i: tf.numpy_function(gen, [i], Tout=(tf.uint8, tf.float32)),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False,
    )
    ds = ds.batch(batch_size)
    return ds


if __name__ == "__main__":
    _main()
