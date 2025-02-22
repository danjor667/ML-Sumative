# ML Sumative
# Medical Image Analysis with CNN - Pneumonia Detection

This project leverages Chest X-ray images to classify medical conditions as **NORMAL** or **PNEUMONIA** using a **Convolutional Neural Network (CNN)**. The project involves preprocessing the dataset, building a CNN model, training it, and analyzing its performance.

---

# About the dataset:
the dataset use for this project is the Chest X-Ray Images (Pneumonia) found on kaggle with over 5000 images of  chest x-ray for both sick(Pneumonia) and healthy patients.
the Link to the dataset is provided below.

## Table of Contents
- [Features](#features)
- [Prerequisites and Requirements](#prerequisites-and-requirements)
- [Dataset](#dataset)
- [Project Overview](#project-overview)
  - [Data Preprocessing](#data-preprocessing)
  - [Visualization](#visualization)
  - [Model Building](#model-building)
  - [Model Evaluation](#model-evaluation)
  - [Saving and Using the Model](#saving-and-using-the-model)
- [Results](#results)
- [Setup Instructions](#setup-instructions)
- [References](#references)

---

## Features

1. **End-to-End Image Classification**:
   - Diagnosis of chest X-ray images into "NORMAL" or "PNEUMONIA".

2. **Preprocessing Pipeline**:
   - Includes image resizing, normalization, and dataset preparation.

3. **Custom CNN Model**:
   - A simple yet effective CNN architecture built using TensorFlow/Keras.

4. **Analysis and Visualization**:
   - Metrics including accuracy, precision, recall, F1-score, and a confusion matrix.
   - Visual inspection of classified images.

---

## Prerequisites and Requirements

To run this project, ensure you have the following installed:

- Python 3.8 or higher
- Required libraries: `numpy`, `pandas`, `matplotlib`, `tensorflow`, `scikit-learn`, `seaborn`, `opencv-python`

Install the dependencies using:
```shell script
pip install numpy pandas matplotlib tensorflow scikit-learn seaborn opencv-python
```

---

## Dataset

The dataset used in this project contains X-ray images of human lungs. It is organized into training and testing folders with separate subfolders for "NORMAL" and "PNEUMONIA" cases.

### Dataset Structure
```
chest_xray/
    train/
        NORMAL/
        PNEUMONIA/
    test/
        NORMAL/
        PNEUMONIA/
    val/
        NORMAL/
        PNEUMONIA/
```

> **Note:** Due to memory constraints, only a portion of the dataset has been processed for training and testing. But you can use all the whole dataset to train your model if you  have enough compute or even  an Nvidea GPU

---

## Project Overview

This project follows a systematic approach involving the following steps:

### Data Preprocessing

1. **Loading and Normalizing Images**:
   - Images are resized to `(255x255)` dimensions.
   - Pixel values are normalized to `[0, 1]`.

   **Labels**:
   - `NORMAL` images are labeled as `0`.
   - `PNEUMONIA` images are labeled as `1`.

   **Dataset Split**:
   - Training (for model learning)
   - Testing (for evaluation)
   - Validation (for validation)

2. **Memory Constraints**:
   - Due to limited RAM, only a subset of the dataset is processed during training and testing.
   - feel free to use all the dataset to train your model if enoyugh compute power i avalable

---

### Visualization

1. **Train Data Visualization: Normal**:
   Example samples of training data labeled as `NORMAL` (0) and `PNEUMONIA` (1) cases.

   ![Train Data: Normal](images/Screenshot%20from%202025-02-22%2018-34-54.png) <!-- Add a link to your pre-generated train visualization image -->

2. **Train Data Visualization: Pneumonia**:
   Visualization of test data samples.

   ![Train Data: Pneumonia](/images/Screenshot%20from%202025-02-22%2018-36-00.png) <!-- Add a link to your pre-generated test visualization image -->

---

### Model Building

The CNN architecture used in this project is designed as:

- **Convolution & Pooling Layers**:
  - Feature extraction from input images.
- **Dense Layers**:
  - Fully connected layers for classification.
- **Output Layer**:
  - Two output neurons (representing `NORMAL` and `PNEUMONIA` classes) with softmax activation.

**Model Summary**:
```
Layer (type)                Output Shape              Param #
================================================================
conv2d_1 (Conv2D)           (None, 255, 255, 32)      896
max_pooling2d_1 (MaxPooling2D)(None, 127, 127, 32)    0
conv2d_2 (Conv2D)           (None, 127, 127, 16)      4624
max_pooling2d_2 (MaxPooling2D)(None, 63, 63, 16)      0
...
```

The model is compiled with:
- Optimizer: `adam`
- Loss function: `categorical_crossentropy`
- Metric: `accuracy`

The training process involves:
- **Epochs**: 10
- **Validation Data**: Test data is used for validation after each epoch.

---

### Model Evaluation

The model is evaluated on the test set using the following performance metrics:

1. **Accuracy**:
   The percentage of correct predictions.

2. **Precision**:
   The proportion of correctly identified positive predictions.

3. **Recall**:
   Measures the ability to identify true positives.

4. **F1-Score**:
   A balance between precision and recall.

5. **Confusion Matrix**:
   A matrix showing the performance based on predicted and actual values.

   ![Confusion Matrix](/images/Screenshot%20from%202025-02-22%2018-49-19.png) <!-- Add your confusion matrix image -->

---

### Saving and Using the Model

The trained model is saved as a `.h5` file using:
```python
model.save('CNN_model.h5')
```

You can later reload the model for predictions on unseen data.

---

## Results

1. **Performance**:
   - **Accuracy**: `0.73`
   - **Precision**: `0.7`
   - **Recall**: `1.0`
   - **F1 Score**: `0.82`



The CNN model demonstrates a strong effectiveness in distinguishing between normal and pneumonia-affected lungs.

---

## Setup Instructions

1. Clone this repository:
```shell script
git clone https://github.com/danjor667/ML-Sumative.git
   cd ML-Sumative
```

2. Install dependencies:
```shell script
pip install -r requirements.txt
```

3. Place your dataset in the `chest_xray/` directory, following the structure outlined above.

4. Run the Jupyter Notebook:
```shell script
jupyter notebook
```

5. Execute the cells step-by-step to train and evaluate the model.

---

## References

- **Dataset**: [Chest X-ray Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Tools and Libraries**:
  - [TensorFlow](https://www.tensorflow.org/)
  - [OpenCV](https://opencv.org/)

