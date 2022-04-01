# Credit Risk Analysis for Lending club

## Overview of Project

The purpose of this project is to use supervised machine learning in predicting credit risk. Lending Club management believes that using statistical reasoning and machine learning algorithm will provide a quicker and more reliable loan experience. It also believe that machine learning will lead to more accurate identification of good candidates for loans. This project will explore several machine learning models or algorithms to predict credit risk.
This project will also employ different techniques to train and evaluate models with unbalanced classes.The data science analytics team lead has specifically requested that i make use of imbalanced-learn and scikit-learn libraries to build and evaluate models using resampling.

I will be using the credit card credit dataset from LendingClub. I shall oversample the data using the RandomOverSampler and SMOTE algorithms, and undersample the data using the ClusterCentroids algorithm. Also i will use a combinatorial approach of over- and undersampling using the SMOTEENN algorithm. And finally compare two new machine learning models that reduces bias, BalancedRandomForestClassifier and EasyEnsembleClassifier, to predict credit risk.

## Resources and Tools Used

Data Source:LendingClub dataset on credit card risk from LoanStats_2019Q1.csv.
Tools: Matplotlib,Visual studio code,Github,python libries such as sklearn,nunpy,pandas,pathlib,collections and imblearn.

# Results

As mentioned in the overview, i use Machine Learning to resample the dataset using Python libraries: scikit-learn and imbalanced-learn evaluate the results and provide a comparison for our analysis.

The original dataset contained 115,675 loan applications in Q1 of 2019. We used the "loan status" to determine whether the application was considered "low" or "high" risk. Applications that had "current" as the "loan status" were classified as "low risk" and the remaining as "high risk". This reduced the dataset to 68,817 total applications with 99% classified as "low risk".

![loan_status](https://github.com/femiimam001/Credit_Risk_Analysis/blob/main/Resources/loan_status.PNG)

Using the 75/25% method to split the data for training vs. testing, 51,366 "low risk" and 246 "high risk" applications were categorized into the training set.
![training_testing](https://github.com/femiimam001/Credit_Risk_Analysis/blob/main/Resources/training_testing.PNG)

# Deliverable 1: Use Resampling Models to Predict Credit Risk

## Oversampling

RandomOverSampler Model randomly selects from the minority class and adds it to the training set until both classifications are equal. The results classified 51,366 records each as High Risk and Low Risk.

![random_sample](https://github.com/femiimam001/Credit_Risk_Analysis/blob/main/Resources/random_sample.PNG)

Balanced accuracy score: 64%.

![balanced_acc_score](https://github.com/femiimam001/Credit_Risk_Analysis/blob/main/Resources/balanced_acc_score.PNG)

The "High Risk" precision rate was only 1% with the recall at 66% giving this model an F1 score of 2%.
"Low Risk" had a precision rate of 100% and recall at 62%.

SMOTE (Synthetic Minority Oversampling Technique) Model, like RandomOverSampler increases the size of the minority class by creating new values based on the value of the closest neighbors to the minority class instead of random selection.

The balanced accuracy score improved slightly to 65.1%.
![SMOTE](https://github.com/femiimam001/Credit_Risk_Analysis/blob/main/Resources/SMOTE.PNG)

Like RandomOverSampler, the "High Risk" precision rate again was only 1% with the recall degraded to 61% giving this model an F1 score of 2%.
"Low Risk" had a precision rate of 100% and an improved recall at 69%.
![SMOTE_precision](https://github.com/femiimam001/Credit_Risk_Analysis/blob/main/Resources/SMOTE_precision.PNG)

## Undersampling

ClusterCentroids Model, an algorithm that identifies clusters of the majority class to generate synthetic data points that are representative of the clusters. The model classified 246 records each as High Risk and Low Risk.
![clus_cenroid_model](https://github.com/femiimam001/Credit_Risk_Analysis/blob/main/Resources/clus_centroid_model.PNG)

Balanced accuracy score was lower than the oversampling models at 54.5%.

![Clus_centroid_score](https://github.com/femiimam001/Credit_Risk_Analysis/blob/main/Resources/clus_centroid_score.PNG)
The "High Risk" precision rate again was only at 1% with the recall at 69% giving this model an F1 score of 1%.
"Low Risk" had a precision rate of 100% and with a lower recall at 40% compared to the oversampling models.

![clus_centroid_precision](https://github.com/femiimam001/Credit_Risk_Analysis/blob/main/Resources/clus_cenroid_precision.PNG)

# Deliverable 2: Use the SMOTEENN algorithm to Predict Credit Risk

## Combination Sampling

SMOTEENN (Synthetic Minority Oversampling Technique + Edited NearestNeighbors) Model combines aspects of both oversampling and undersampling. The model classified 68,460 records as High Risk and 62,011 as Low Risk.

![SMOTEENN_model](https://github.com/femiimam001/Credit_Risk_Analysis/blob/main/Resources/SMOTEENN_model.PNG)

The balanced accuracy score improved to 64.5% when using a combined sampling model.

![SMOTEENN_score](https://github.com/femiimam001/Credit_Risk_Analysis/blob/main/Resources/SMOTEENN_score.PNG)

The "High Risk" precision rate did not improve was only 1%, however the recall increased to 72% giving this model an F1 score of 2%.
"Low Risk" still showed a precision rate of 100% with the recall at 57%.

![SMOTEENN_precision](https://github.com/femiimam001/Credit_Risk_Analysis/blob/main/Resources/SMOTEENN_precision.PNG)

# Deliverable 3: Use Ensemble Classifiers to Predict Credit Risk

Compare two new Machine Learning models that reduce bias to predict credit risk. The models classified 51,366 as High Risk and 246 as Low Risk.

![ensemble_model](https://github.com/femiimam001/Credit_Risk_Analysis/blob/main/Resources/ensemble_model.PNG)

BalancedRandomForestClassifier Model, two trees of the same size and equal size to the minority class are constructed to represent one for the majority class and one for the minority class.

![ensemble_score](https://github.com/femiimam001/Credit_Risk_Analysis/blob/main/Resources/ensemble_score.PNG)

The balanced accuracy score increased to 78.9% for this model.

!The "High Risk precision rate increased to 3% with the recall at 70% giving this model an F1 score of 6%.
"Low Risk" still had a precision rate of 100% with the recall at 87%.
The top feature by importance was "total_rec_prncp" at 7.9% of the total.

![ensemble_precision](https://github.com/femiimam001/Credit_Risk_Analysis/blob/main/Resources/ensemble_precision.PNG)

![ensemble_features]()
