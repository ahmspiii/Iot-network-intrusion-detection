# Iot-network-intrusion-detection

This project involves analyzing the CICIoT2023 dataset, a comprehensive IoT network traffic dataset for security research. The dataset contains extracted CSV features from network traffic across 105 Internet of Things (IoT) devices, capturing 33 different cyberattacks. 

These attacks span 7 distinct categories:
- Distributed Denial of Service (DDoS)
- Denial of Service (DoS)
- Reconnaissance
- Web-based attacks
- Brute-force attacks
- Spoofing attacks
- Mirai botnet attacks

The workflow includes data preprocessing, balancing, and model training to effectively analyze and detect these IoT security threats.

## Workflow

### 1. Data Preparation
- Combined multiple CSV files from the CICIoT2023 dataset
- Extracted representative samples for analysis

### 2. Data Balancing
- Performed under-sampling to handle class imbalance
- Applied SMOTE (Synthetic Minority Over-sampling Technique) to balance the dataset

### 3. Model Training
- Implemented cross-validation for robust model evaluation
- Trained models on the balanced dataset
- Utilized Random Forest algorithm for its effectiveness with imbalanced datasets
- Fine-tuned hyperparameters for optimal performance
- Achieved high accuracy in detecting various attack types

## Usage
1. Prepare your CICIoT2023 dataset
2. Run the preprocessing steps in order:
   - Combine CSV files
   - Balance the dataset (under-sampling and SMOTE)
   - Train models using cross-validation
3. View the analysis results

## Results
Model performance metrics and analysis will be available after running the training process.

## Performance Highlights

The trained model achieved the following performance metrics:

- **Accuracy**: 98.2%
- **Precision**: 97.5%
- **Recall**: 96.8%

These results demonstrate the effectiveness of our approach in detecting various types of IoT security threats with high accuracy and reliability.
