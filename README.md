# Pneumonia Detection in Chest X-Rays with CNN

Medical image classification project using convolutional neural networks to identify pneumonia in chest X-rays.

## Description

The goal of this project is to develop a medical image classification model capable of identifying the presence of pneumonia in chest radiographs. Three approaches are compared: a custom CNN trained from scratch, a hybrid MobileNetV2 + Random Forest pipeline, and a fine-tuned MobileNetV2 with a dense classification head. The project uses TensorFlow, Keras and scikit-learn.

**Dataset**: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/data).

## Requirements and recommendations

> ⚠️ Python 3.12 or lower is required to run the project
>
> ⚠️ Running the project on Linux is strongly recommended

To work in an isolated environment, create a virtual environment (optional but recommended):

```
python -m venv venv
source venv/bin/activate  # Linux/macOS
.\venv\Scripts\activate   # Windows
```

Then install the required dependencies from `requirements.txt`:

```
pip install -r requirements.txt
```

## Running the Jupyter notebook

The file `deteccion_neumonia.ipynb` is a Jupyter notebook organised in sections that explain each step of the code. Open it with **VSCode** or launch it in a browser via **Jupyter Lab**:

1. Start a Jupyter server:

   ```
   jupyter lab
   ```

   The browser should open automatically. If not, continue with the steps below.

2. List running servers in another terminal:

   ```
   jupyter server list
   ```

3. You will see a localhost URL. `Ctrl + click` to open the notebook in the browser, where you can run it section by section or all at once.

## Running the training script

If you prefer not to use the notebook, run `train_and_test_model.py` directly:

```
python3 train_and_test_model.py
```

This will train all three models (or load them if already saved) and export comparative metrics to `model_metrics.json`.

## Launching the web interface

```
streamlit run app.py
```
