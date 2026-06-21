Optimized Spectrum Sensing Algorithm Using Machine Learning
Project Overview

This project focuses on improving spectrum sensing in Cognitive Radio Networks using Machine Learning and Deep Learning algorithms.

Traditional spectrum sensing methods often struggle in noisy environments and low Signal-to-Noise Ratio (SNR) conditions. To address this challenge, multiple machine learning models were evaluated and compared to improve signal detection accuracy and reduce false alarms.

Objective

The main objective of this project is to identify the most effective machine learning algorithm for detecting spectrum occupancy under varying noise conditions.

Algorithms Evaluated
K-Nearest Neighbors (KNN)
Support Vector Machine (SVM)
Logistic Regression (LR)
Random Forest (RF)
Multi-Layer Perceptron (MLP)
Convolutional Neural Network (CNN)
Long Short-Term Memory (LSTM)
Feature Engineering

The following signal-processing features were generated:

Energy Detection
Differential Entropy
Geometric Mean
Log Power

These features help capture signal characteristics required for accurate classification.

Methodology
Data Collection and Preprocessing
Feature Extraction
Model Training
Model Evaluation
Performance Comparison
Technologies Used
Python
NumPy
Pandas
Scikit-Learn
TensorFlow / Keras
Matplotlib
Performance Evaluation

Models were evaluated using:

Probability of Detection (Pd)
ROC Curves
Classification Accuracy
Performance under varying SNR conditions
Key Findings
CNN and LSTM demonstrated strong performance in detecting complex signal patterns.
Random Forest showed robust performance under noisy conditions.
SVM achieved consistent results across multiple SNR levels.
Different models performed better under different operating conditions, highlighting the importance of model selection.
Results
Model Comparison




ROC Curve




Repository Structure

Dataset/
Models/
Source_Code/
Images/

How To Run
Clone the repository

git clone YOUR_REPOSITORY_LINK

Install dependencies

pip install -r requirements.txt

Run the project

python modelling.py

Research Paper

Published Research Paper:

https://www.ijfmr.com/papers/2025/2/40605.pdf

Future Improvements
Real-time spectrum sensing deployment
Larger datasets
Transformer-based architectures
Edge-device optimization

Author
Santhosh Govvala
