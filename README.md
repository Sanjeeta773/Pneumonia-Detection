# Pneumonia Detection from Chest X-rays

This project is a simple deep learning application for detecting **pneumonia from chest X-ray images** using a convolutional neural network (CNN).

The goal of the project is to build and evaluate a **binary image classification model** that predicts whether an X-ray image shows **NORMAL** lungs or **PNEUMONIA**.

---

## Dataset

The dataset consists of chest X-ray images organized into the following folders:

- `train/`
- `val/`
- `test/`

Each folder contains two classes:
- `NORMAL`
- `PNEUMONIA`

Images are converted to grayscale and resized before being fed into the model.

---

## Model

A simple CNN classifier (`SimpleClassifier`) is used:
- Convolutional layers with ReLU activation
- Batch normalization
- Max pooling
- Global average pooling
- A single output neuron for binary classification

The model outputs a **logit**, and `BCEWithLogitsLoss` is used during training.

---

## Training

The model is trained using:
- Adam optimizer
- Binary Cross Entropy loss=
- GPU if available, otherwise CPU

The best model weights are saved to:
output/best_model.pth

---

## Evaluation

The model is evaluated on the validation set using the following metrics:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

Example evaluation results:
accuracy: 0.75
precision: 0.70
recall: 0.88
f1: 0.78
auc: 0.91

High recall and AUC are important for medical screening tasks.

---

## How to Run
1. Install dependencies
pip install -r requirements.txt
2. Train the model
python3 main.py
3. Evaluate the model
python3 evaluate.py