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
![con_classification]()
