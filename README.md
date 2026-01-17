Computer Vision: Skin Cancer Classification (HAM10000)

This repository contains a comprehensive deep learning pipeline for the detection and classification of skin cancer using the HAM10000 ("Human Against Machine with 10,000 training images") dataset. This project leverages state-of-the-art architectures to assist in dermatological diagnosis.

📁 Repository Structure

HAM10000_Skin_Cancer_Classification.ipynb: The primary implementation notebook including data preprocessing, model training, and evaluation.

Research Paper.pdf: Supporting academic documentation detailing the methodology and clinical context.

Skin_Cancer_Classification_Ham10000.ipynb: Alternative experiments and model variations.

📋 Project Overview

Skin cancer is the most common human malignancy. This project implements a multi-class classifier to distinguish between seven types of skin lesions:

Melanocytic nevi (NV)

Melanoma (MEL)

Benign keratosis-like lesions (BKL)

Basal cell carcinoma (BCC)

Actinic keratoses (AKIEC)

Vascular lesions (VASC)

Dermatofibroma (DF)

🛠️ Technical Implementation

1. Data Preprocessing & Augmentation

Due to the significant class imbalance inherent in medical datasets (specifically the prevalence of NV), the following techniques were implemented:

Resizing & Normalization: Standardizing images for neural network input.

Over-sampling: Balancing the dataset to ensure minority classes like Melanoma are adequately represented during training.

Data Augmentation: Applying random rotations, flips, and shifts to improve model generalization.

2. Deep Learning Models

The project explores various architectures, focusing on:

Convolutional Neural Networks (CNNs): Custom architectures and transfer learning approaches.

Evaluation Metrics: Prioritizing Accuracy, Precision, Recall, and F1-Score to ensure safety in clinical predictions.

🚀 Getting Started

Prerequisites

Ensure you have the following libraries installed:

pip install numpy pandas matplotlib seaborn scikit-learn tensorflow keras


Usage

Clone the repository:

git clone [https://github.com/usamasShk/ComputerVision.git](https://github.com/usamasShk/ComputerVision.git)


Dataset:
Download the HAM10000 dataset from Kaggle or the official ISIC archive and place the images in a /data directory.

Execution:
Open the .ipynb notebooks in Jupyter or Google Colab to run the training pipeline.

📊 Results and Evaluation

The models are evaluated using Confusion Matrices to visualize misclassifications, particularly focusing on reducing false negatives in high-risk categories like Melanoma. Detailed classification reports are generated for each experiment.

📜 References

Dataset: HAM10000 Dataset via Harvard Dataverse

Paper: Detailed methodology is available in the Research Paper.pdf included in this repository.

This documentation was generated for the ComputerVision repository by Usama.
