# Credit Risk Analysis for Lending club

## Overview of Project

The purpose of this project is to use supervised machine learning in predicting credit risk. Lending Club management believes that using statistical reasoning and machine learning algorithm will provide a quicker and more reliable loan experience. It also believe that machine leraning will lead to more accurate identification of good candidates for loans. This project will explore several machine learning models or algorithms to predict credit risk.
This project will also employ different techniques to train and evaluate models with unbalaced classes.The data science analytics team lead has specifically requested that i make use of imbalanced-learn and scikit-learn libraries to build and evaluate models using resampling.

I will be using the credit card credit dataset from LendingClub. I shall oversample the data using the RandomOverSampler and SMOTE algorithms, and undersample the data using the ClusterCentroids algorithm. Also we will use a combinatorial approach of over- and undersampling using the SMOTEENN algorithm. And finally compare two new machine learning models that reduces bias, BalancedRandomForestClassifier and EasyEnsembleClassifier, to predict credit risk.

## Resources and Tools Used

Data Source:LendingClub dataset on credit card risk from LoanStats_2019Q1.csv.
Tools: Matplotlib,Visual studio code,Github,python libries such as sklearn,nunpy,pandas,pathlib,collections and immblearn.

# Results

As mentioned in the overview, we use Machine Learning to resample the dataset using Python libraries: scikit-learn and imbalanced-learn evaluate the results and provide a comparison for our analysis.

The original dataset contained 115,675 loan applications in Q1 of 2019. We used the "loan status" to determine whether the application was considered "low" or "high" risk. Applications that had "current" as the "loan status" were classified as "low risk" and the remaining as "high risk". This reduced the dataset to 68,817 total applications with 99% classified as "low risk".
![loan_status]()
