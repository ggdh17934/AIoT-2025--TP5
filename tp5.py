"""TP5: train the two architectures requested in the Fashion-MNIST TP."""
# %% 1. Imports and settings
import json
import os
from pathlib import Path
from time import perf_counter

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

ROOT = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
OUT = ROOT / "results"
OUT.mkdir(parents=True, exist_ok=True)
keras.utils.set_random_seed(42)
EPOCHS = 5
BATCH_SIZE = 64

# %% 2. Load and normalize the complete Fashion-MNIST dataset
(train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()
x_train_mlp = train_images.astype("float32") / 255.0
x_test_mlp = test_images.astype("float32") / 255.0
x_train_cnn = x_train_mlp[..., np.newaxis]
x_test_cnn = x_test_mlp[..., np.newaxis]
print("MLP:", x_train_mlp.shape, x_test_mlp.shape)
print("CNN:", x_train_cnn.shape, x_test_cnn.shape)

# %% 3. Build the exact MLP and CNN architectures from the assignment
mlp_model = keras.Sequential([
    layers.Input(shape=(28, 28)),
    layers.Flatten(),
    layers.Dense(256, activation="relu"),
    layers.Dense(128, activation="relu"),
    layers.Dense(10, activation="softmax"),
], name="mlp_model")

cnn_model = keras.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(16, 3, activation="relu"),
    layers.MaxPooling2D(2),
    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPooling2D(2),
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax"),
], name="cnn_model")

for model in [mlp_model, cnn_model]:
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()

# %% 4. Train five epochs and evaluate on the independent test set
models = {"MLP": mlp_model, "CNN": cnn_model}
histories = {}
rows = []
for name, model in models.items():
    train_x = x_train_mlp if name == "MLP" else x_train_cnn
    test_x = x_test_mlp if name == "MLP" else x_test_cnn
    start = perf_counter()
    history = model.fit(
        train_x, train_labels, epochs=EPOCHS, batch_size=BATCH_SIZE,
        validation_split=0.1, verbose=2,
    )
    training_s = perf_counter() - start
    loss, accuracy = model.evaluate(test_x, test_labels, verbose=0)
    histories[name] = history.history
    model.save(OUT / f"{name.lower()}_model.keras")
    model.save(OUT / f"{name.lower()}_model.h5", include_optimizer=False)

    # Analytical estimates: count a multiply plus an addition as two FLOPs.
    # Ignore activation, pooling, bias, and softmax operations in this estimate.
    macs = 0
    activations = 0
    for layer in model.layers:
        shape = tuple(int(d) for d in layer.output.shape[1:])
        activations += int(np.prod(shape))
        if isinstance(layer, layers.Dense):
            macs += int(np.prod(layer.kernel.shape))
        elif isinstance(layer, layers.Conv2D):
            macs += int(np.prod(shape)) * int(np.prod(layer.kernel.shape[:3]))
    parameters = int(sum(np.prod(weight.shape) for weight in model.trainable_weights))
    inference_flops = 2 * macs
    # Approximately one forward and two backward passes per batch.
    training_step_flops = 3 * inference_flops * BATCH_SIZE
    # Float32 parameters + gradients + Adam's two moment arrays = 16 bytes/P.
    core_bytes = 16 * parameters
    # Rough activation/gradient storage; excludes allocator and runtime overhead.
    estimated_training_bytes = core_bytes + 8 * BATCH_SIZE * activations
    rows.append({
        "Model": name, "Test_accuracy": accuracy, "Test_loss": loss,
        "Trainable_parameters": parameters,
        "H5_size_MB": (OUT / f"{name.lower()}_model.h5").stat().st_size / 1024**2,
        "Inference_FLOPs_estimate": inference_flops,
        "Training_step_FLOPs_estimate_batch64": training_step_flops,
        "Parameters_gradients_Adam_MB": core_bytes / 1024**2,
        "Training_memory_estimate_MB": estimated_training_bytes / 1024**2,
        "Training_s": training_s,
    })
results = pd.DataFrame(rows)
results.to_csv(OUT / "results.csv", index=False)
(OUT / "history.json").write_text(json.dumps(histories, indent=2), encoding="utf-8")
print(results.to_string(index=False))

# %% 5. Plot learning curves and write the comparison report
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for name, history in histories.items():
    epochs = range(1, EPOCHS + 1)
    axes[0].plot(epochs, history["val_accuracy"], marker="o", label=name)
    axes[1].plot(epochs, history["val_loss"], marker="o", label=name)
axes[0].set(title="Validation accuracy", xlabel="Epoch", ylabel="Accuracy")
axes[1].set(title="Validation loss", xlabel="Epoch", ylabel="Loss")
for axis in axes:
    axis.legend()
    axis.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "learning_curves.png", dpi=140)
plt.close(fig)

table = "| Model | Test accuracy | Loss | Parameters | H5 MB | Inference FLOPs* | Training step FLOPs* (batch 64) | Training memory MB* |\n|---|---|---|---|---|---|---|---|\n"
for row in rows:
    table += f"| {row['Model']} | {row['Test_accuracy']:.4f} | {row['Test_loss']:.4f} | {row['Trainable_parameters']} | {row['H5_size_MB']:.3f} | {row['Inference_FLOPs_estimate']} | {row['Training_step_FLOPs_estimate_batch64']} | {row['Training_memory_estimate_MB']:.3f} |\n"
winner = max(rows, key=lambda row: row["Test_accuracy"])["Model"]
smaller = min(rows, key=lambda row: row["Trainable_parameters"])["Model"]
report = f"""# TP5 report

Both assignment architectures were trained for five epochs, batch size 64,
Adam, sparse categorical cross-entropy. Fashion-MNIST provides 60,000 training
and 10,000 test images. A 10% validation partition is taken from the training
set; the test set is evaluated only after training. Pixels are divided by 255.

{table}
*FLOPs and training memory are analytical estimates, not hardware measurements.
One multiply-add counts as two FLOPs. Forward FLOPs include Dense and Conv2D
kernels; activations, bias, pooling, and softmax are excluded. A training step
means **one batch of 64**, approximated as three forward-pass costs per sample.
Training memory estimates parameters, gradients, two Adam moments (16 bytes
per parameter), plus approximate batch activation/gradient storage. TensorFlow
runtime, data storage, workspaces, and allocator overhead are excluded, so this
is not a measured peak. Model-file MB means bytes / 1024².

**{winner}** achieved higher test accuracy in this run; **{smaller}** has fewer
parameters. CNN filters share weights across image locations and preserve
spatial structure, helping learn local patterns. The MLP is simple and uses
fewer arithmetic operations here, but its first dense layer connects every
pixel to every hidden neuron. Fewer parameters do not necessarily mean fewer
FLOPs: convolutions reuse kernels at many locations.

![Validation learning curves](learning_curves.png)

Saved .keras files are the reusable models for TP6; .h5 files satisfy the size
comparison. Measurements are from this computer; Colab runs may differ.

Source: https://github.com/ahmed3991/AIoT-2025/blob/main/TP5/TP5.md
"""
(OUT / "REPORT.md").write_text(report, encoding="utf-8")
print("Saved trained models, curves, metrics, and REPORT.md in", OUT)
