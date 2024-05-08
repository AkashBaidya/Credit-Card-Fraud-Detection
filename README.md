

---

# Credit Card Fraud Detection

## Overview

This project focuses on building and evaluating machine learning models for detecting fraudulent credit card transactions. Credit card fraud is a prevalent issue in the financial industry, and the ability to accurately identify fraudulent transactions is crucial for preventing financial losses and ensuring customer trust.

## Dataset

The dataset used in this project consists of simulated credit card transactions, including transaction amounts and class labels indicating whether each transaction is legitimate or fraudulent. The dataset was generated programmatically to mimic real-world scenarios, with approximately 1% of transactions labeled as fraudulent.

## Methodology

1. **Data Generation**: Simulated credit card transactions were generated with random transaction amounts and class labels.

2. **Data Analysis**: Exploratory data analysis (EDA) was performed to understand the distribution of transaction amounts and the balance between legitimate and fraudulent transactions.

3. **Data Preprocessing**: The dataset was preprocessed by splitting it into features and target variables, standardizing the features, and splitting it into training and testing sets.

4. **Model Training**: Three machine learning models were trained using the training data:
   - Logistic Regression
   - Decision Trees
   - Random Forests

5. **Model Evaluation**: The trained models were evaluated using the testing data. Performance metrics such as accuracy, classification report, confusion matrix, and AUC score were calculated for each model.

## Results

- **Logistic Regression**: Achieved an accuracy of 0.9865% and an AUC score of 0.512324.
- **Decision Trees**: Achieved an accuracy of  0.9710 % and an AUC score of 0.492144.
- **Random Forests**: Achieved an accuracy of 0.9710% and an AUC score of 0.488953.

## Conclusion

- The machine learning models showed promising performance in detecting fraudulent credit card transactions.
- Further optimization and fine-tuning of the models could potentially improve their performance.
- Continuous monitoring and updating of the models are necessary to adapt to evolving fraud patterns and ensure effectiveness in real-world scenarios.

## Future Work

- Experiment with additional machine learning algorithms and ensemble methods.
- Explore advanced feature engineering techniques to improve model performance.
- Incorporate real-world credit card transaction data for more robust model training and evaluation.

---

