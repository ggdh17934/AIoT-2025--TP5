# AIoT-2025--TP5 — Fashion-MNIST MLP versus CNN

Simple, documented solution to [the course TP5](https://github.com/ahmed3991/AIoT-2025/tree/main/TP5).

[Open notebook in Colab](https://colab.research.google.com/github/ggdh17934/AIoT-2025--TP5/blob/main/TP5-Ghanabzia.ipynb)

This repository is **private**. Sign in with your GitHub account when Colab asks
for access. You can also download the notebook and use Colab's **Upload** option.

## Run locally

Use Python 3.11 or newer (TensorFlow builds depend on the supported Python version).
From this project folder:

```sh
python -m pip install -r requirements.txt
python tp5.py
```

## Run the notebook

Run all. Fashion-MNIST downloads automatically. Training uses five epochs; Colab GPU is optional. Keep the generated mlp_model.keras and cnn_model.keras for TP6.

The first notebook cell installs missing dependencies. TP2/TP4 also ensure the
versions used for their pickle models. All notebooks contain the actual solution
code; they do not require downloading the Python script just to execute cells.
Saved outputs come from the local run; rerunning generates your own measurements.

## Files

- `TP5-Ghanabzia.ipynb`: notebook with code, outputs, and report display.
- `tp5.py`: Python solution.
- `EXPLICATION_AR.md`: Arabic explanation of the code, concepts and limitations.
- `results/REPORT.md`: measured results and conclusions.
- `results/VALIDATION.md`: what was actually checked.
- `requirements.txt`: runtime dependencies.


## Sources and attribution

Assignment and original wiring: Ahmed Ghenabzia,
[ahmed3991/AIoT-2025](https://github.com/ahmed3991/AIoT-2025).
Solution Python and firmware are implemented for this project; diagrams are
adapted from the course starter files. Smoke data source:
[Kaggle Smoke Detection](https://www.kaggle.com/datasets/deepcontractor/smoke-detection-dataset).
Fashion-MNIST is loaded through the official Keras dataset loader; source:
[Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist).
