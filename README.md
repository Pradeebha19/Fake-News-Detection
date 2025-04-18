## 📰 Fake News Detection Using Machine Learning and Power BI

🔍 Overview
This project is focused on detecting fake news articles using Natural Language Processing (NLP) and Machine Learning techniques. Given the rise of misinformation across digital platforms, this system aims to automatically classify news as real or fake based on textual content.

It includes:

A cleaned and preprocessed dataset

Multiple ML models with hyperparameter tuning

Exported prediction results

A Power BI dashboard to visualize the model’s performance

## 📌 Problem Statement
Automatically identify whether a news article is real or fake based on the content of its text using machine learning.

## 🧠 Technologies & Tools
Python (Pandas, NumPy, scikit-learn, NLTK, etc.)

Jupyter Notebook / Colab

Power BI (for visualizations)

Git & GitHub (for version control)

NLP techniques (text cleaning, TF-IDF)

## ⚙️ Project Workflow
1. Data Collection
Two datasets were used:

Fake.csv – Contains fake news articles

True.csv – Contains real news articles

Both were merged and labeled accordingly.

## 2. Data Cleaning & Preprocessing
Lowercasing text

Removing special characters and stopwords

Tokenization (optional)

Text normalization

## 3. Feature Engineering
Used TF-IDF Vectorization to convert text data into numerical features.

## 4. Model Training & Hyperparameter Tuning
Trained and evaluated multiple ML models:

✅ Logistic Regression

✅ Multinomial Naive Bayes

✅ Support Vector Machine (SVM)

✅ Random Forest (with hyperparameter tuning)

Used GridSearchCV for optimal parameter selection.

## 5. Evaluation Metrics
Accuracy

Classification Report

Confusion Matrix

Predicted labels were saved into model_results.csv.

## 6. Power BI Dashboard
The model_results.csv file was used to create an interactive Power BI dashboard showcasing:

Actual vs. Predicted label distribution

Model accuracy

Misclassified examples

Confusion Matrix

## 📈 Sample Visuals in Power BI

The Power BI dashboard was created to visually analyze the performance of a machine learning model trained to classify news articles as Real (1) or Fake (0). This visualization helps stakeholders, non-technical users, or evaluators understand:

How accurate the model is

The distribution of predictions

Where the model performs well or struggles

Overall trustworthiness of the fake news detection system

## 📌 Key Dashboard Elements and Their Purpose
1. 🧾 Prediction Summary Table (Top Left)
Displays individual predictions (Text, Actual Label, Predicted Label)

Shows how the model classified specific news articles

Currently shows vectorized format — could be improved to show original text for clarity

2. ✅ Accuracy Card (Top Right)
Clearly shows model accuracy: 0.99 (99%)

This KPI card gives a quick snapshot of how reliable the model is overall

3. ❓ Q&A Section (Top Center)
Allows users to ask natural-language questions like:

“show accuracy”

“total actual label”

“average predicted label”

Powered by Power BI's built-in AI Q&A engine

4. 📘 Pie Chart – Count by Actual & Predicted Labels (Bottom Left)
Visualizes the distribution of predictions by category

Helps you see how balanced the dataset is between Real and Fake news

Useful for identifying class imbalance

5. 🔵 Donut Chart – Actual vs Predicted Summary (Bottom Center)
Compares the total number of actual vs. predicted labels

Confirms if the model is biased towards a certain class

Easy to spot underfitting or overfitting visually

6. 📊 Bar Chart – Confusion Matrix View (Bottom Right)
Shows a breakdown of how many predictions are:

True Positives (TP)

True Negatives (TN)

False Positives (FP)

False Negatives (FN)

Axis: Actual Label vs. Predicted Label

Lets you visually assess where the model is making mistakes




![image](https://github.com/user-attachments/assets/65f175b8-23fa-411a-ba0d-37decd3f04a3)

