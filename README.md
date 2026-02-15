# Iris-Data-Classification

📌 Project Overview

This project predicts the species of an Iris flower based on four input features: sepal length, sepal width, petal length, and petal width. Machine Learning algorithms such as K-Nearest Neighbors (KNN) and Naive Bayes are used to perform classification.

A Flask-based web application allows users to enter flower measurements through a simple interface and receive instant predictions.

🎯 Problem Statement

Manual classification of Iris flower species based on physical measurements can be time-consuming and prone to errors. This project aims to automate the classification process using machine learning techniques to improve accuracy and efficiency.

🤖 Machine Learning Models Used
✅ K-Nearest Neighbors (KNN)

K-Nearest Neighbors (KNN) is a supervised machine learning algorithm used for classification and regression tasks. It classifies a new data point based on the majority class among its nearest neighbors in the training dataset.

How KNN Works

Calculates the distance between input data and training samples

Selects the K nearest neighbors

Determines the majority class among neighbors

Assigns the predicted class

Parameter Used

Number of neighbors (K): Selected based on best accuracy

✅ Naive Bayes Classification

Naive Bayes is a supervised classification algorithm based on probability theory and Bayes’ theorem. It predicts the class by calculating the probability of each class given the input features and selecting the class with the highest probability.

How Naive Bayes Works

Calculates probability of each class

Computes probability of input features for each class

Compares probabilities

Selects the class with highest probability

🛠 Technologies Used

Python

NumPy

Pandas

Scikit-learn

Flask

HTML, CSS

Pickle (Model Serialization)

📊 Dataset Information

Dataset: Iris Dataset

Features

Sepal Length

Sepal Width

Petal Length

Petal Width

Target Classes

Setosa

Versicolor

Virginica

⚙️ System Architecture / Workflow

The workflow begins with loading the Iris dataset followed by data preprocessing to prepare the data for model training. The dataset is divided into training and testing sets to evaluate model performance. Machine learning algorithms such as KNN and Naive Bayes are trained to learn relationships between input features and flower species. After achieving satisfactory accuracy, the trained model is saved using Pickle to avoid retraining during execution. The saved model is integrated with a Flask backend, which connects the machine learning model with the web interface. Users enter input values through the web application, the Flask server processes the inputs using the trained model, and the predicted Iris species is displayed as the final result.

📁 Project Structure
Project/
│
├── app.py
├── model.pkl
├── templates/
│   └── index.html
├── static/
│   └── style.css
└── README.md

▶️ How to Run the Project

Clone the repository

git clone <repository-url>


Install required dependencies

pip install -r requirements.txt


Run the Flask application

python app.py


Open the browser and navigate to

http://127.0.0.1:5000/

✅ Result

The system successfully predicts Iris flower species based on user input values with good classification accuracy through a simple and interactive web interface.
