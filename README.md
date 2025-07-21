# Sign Language MNIST Classification with CNN

This project focuses on classifying American Sign Language (ASL) letters using a deep learning model trained on the Sign Language MNIST dataset. The model achieves high accuracy using a Convolutional Neural Network (CNN), making it suitable for gesture recognition tasks in real-time applications.

---

## Dataset

The dataset used is the [Sign Language MNIST](https://www.kaggle.com/datamunge/sign-language-mnist) dataset from Kaggle

- **Training Set**: 27,455 examples  
- **Test Set**: 7,172 examples  
- Each sample is a grayscale 28x28 image, flattened into 784 pixels + 1 label column.  
- Represents 24 static ASL letters (A-Y excluding J).

<img width="989" height="524" alt="image" src="https://github.com/user-attachments/assets/cf7f8ee6-615b-41c2-9cef-90e2eb08f914" />

---

## Exploratory Data Analysis

- No missing values found  
- Visualized class distributions  
- Mapped labels to ASL letters (`J` is omitted)  
- Plotted pixel intensity histograms  
- Analyzed per-image mean and standard deviation  

---

## Preprocessing Steps

- One-hot encoding applied to labels  
- Normalized pixel values to [0, 1]  
- Reshaped data into `(28, 28, 1)` format for CNN input  
- Visualized sample training images  

---

## Model Architecture

A custom CNN model was implemented using TensorFlow and Keras:

```python
model = Sequential([
    Conv2D(75, (3,3), activation='relu', padding='same', input_shape=(28,28,1)),
    BatchNormalization(),
    MaxPooling2D(),

    Conv2D(50, (3,3), activation='relu', padding='same'),
    Dropout(0.2),
    BatchNormalization(),
    MaxPooling2D(),

    Conv2D(25, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(),

    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.3),
    Dense(24, activation='softmax')
])
```

## Training Parameters
-Loss Function: categorical_crossentropy
-Optimizer: adam
-Learning Rate Schedule: ReduceLROnPlateau
-Batch Size: 64
-Epochs: 30


## Model Performance

| Metric   | Training  | Validation | Test     |
|----------|-----------|------------|----------|
| Accuracy | 100.00%   | 100.00%    | 99.93%   |
| Loss     | ~0.0005   | ~0.0003    | 0.0039   |

<p align="center"><b>Accuracy & Loss over Epochs</b></p>

<img width="1188" height="490" alt="image" src="https://github.com/user-attachments/assets/7c1d5031-04c1-47cc-96f2-16c067d88ac4" />

## 📁 File Structure
Sign-Language-MNIST-Classification  
├── sign-language-mnist-classification-cnn-eda.ipynb            
├── README.md                                                   
├── requirements.txt                
└── training_plot.png             

