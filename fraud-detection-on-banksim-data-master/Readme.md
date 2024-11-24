# Smart Detection of Fraudulent Bank Transactions Using Machine learning

# Introduction
Fraudulent behavior is prevalent across various fields such as e-commerce, healthcare, payment, and banking systems. Each year, fraud costs businesses millions of dollars. This project focuses on automating the detection of fraudulent transactions using machine learning techniques.

# Overview
This repository employs machine learning approaches to classify fraudulent transactions. It provides an in-depth analysis, including Exploratory Data Analysis (EDA), and evaluates the performance of various machine learning models.

# Table of Contents
Introduction

Data Collection and Preprocessing

Exploratory Data Analysis (EDA)

Model Selection and Training

Model Evaluation

Results and Insights

Conclusion

Future Work

References

1. Data Collection and Preprocessing
The dataset used in this project is synthetically generated and consists of payments from various customers made over different time periods with different amounts. The preprocessing steps include handling missing values, encoding categorical variables, and scaling numerical features.

Dataset: Banksim1 Dataset

2. Exploratory Data Analysis (EDA)
EDA is performed to understand the data distribution, identify patterns, and spot anomalies. Key steps include:

Visualizing the distribution of transaction amounts.

Analyzing the frequency of transactions over time.

Investigating the correlation between features.

Checking for class imbalance in the dataset.

3. Feature Extraction
For feature extraction, we utilize TF-IDF Vectorization for machine learning models and Tokenization for deep learning models.

4. Model Selection and Training
Machine Learning Models:
K-Nearest Neighbors (KNN)

Random Forest

XGBoost

Deep Learning Models:
LSTM (Long Short-Term Memory) using BERT embeddings

5. Model Evaluation
Models are evaluated based on their accuracy, precision, recall, F1-score, and ROC-AUC score. The performance of each model is compared using confusion matrices and classification reports. the SMOTE for balancing the dataset. Overall results looks more better just check the file called Fraud Detection on Bank Payments.ipynb from inside the repo.

<br/>Classification Report for K-Nearest Neighbours (1:fraudulent,0:non-fraudulent) :

|class | precision | recall | f1-score | support|
| ---- | --------- | ------ | -------- | -------|        
|  0   |   1.00    |   1.00 |  1.00    | 176233 |
|  1   |   0.83    |   0.61 |  0.70    |  2160  |
           
Confusion Matrix of K-Nearest Neigbours:
<br/> [175962    271]
<br/> [   845   1315] 



<br/>Classification Report for XGBoost : 

class | precision | recall | f1-score | support|
| ---- | --------- | ------ | -------- | -------|        
|  0   |   1.00    |   1.00 |  1.00    | 176233 |
|  1   |   0.89    |   0.76 |  0.82    |  2160  |
           
           
Confusion Matrix of XGBoost: 
<br/> [176029    204] 
<br/> [   529   1631] 




<br/>Classification Report for Random Forest Classifier : 

class | precision | recall | f1-score | support|
| ---- | --------- | ------ | -------- | -------|        
|  0   |   1.00    |   0.96 |  0.98    | 176233 |
|  1   |   0.24    |   0.98 |  0.82    |  2160  |
           
         
 Confusion Matrix of Random Forest Classifier: 
<br/> [169552   6681]
<br/> [    39   2121]



<br/>Classification Report for Ensembled Models(RandomForest+KNN+XGBoost) : 

class | precision | recall | f1-score | support|
| ---- | --------- | ------ | -------- | -------|        
|  0   |   1.00    |   1.00 |  1.00    | 176233 |
|  1   |   0.73    |   0.81 |  0.77    |  2160  |
           

Confusion Matrix of Ensembled Models: 
<br/> [175604    629]
<br/> [   417   1743]


## Dataset
https://www.kaggle.com/ntnu-testimon/banksim1

# Results and Insights
The results section summarizes the performance of different models. The ensembled models combining Random Forest, KNN, and XGBoost demonstrate the highest accuracy and balanced performance across various metrics.

