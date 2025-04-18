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
📊 Confusion Matrix (Actual vs. Predicted)

📌 Accuracy Card

🧾 Misclassified Text Table

🧠 Text-based filters and exploration

![image](https://github.com/user-attachments/assets/65f175b8-23fa-411a-ba0d-37decd3f04a3)

