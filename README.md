🔐 Online Payments Fraud Detection with Machine Learning
Accurately classify fraudulent transactions using Python, Decision Trees, and real-world payment data.

📌 Overview
This project demonstrates the development of a Machine Learning model to detect fraudulent online payment transactions. Using a real-world dataset of financial transactions, we train a Decision Tree Classifier to distinguish between legitimate and fraudulent activity. This has vital applications in the finance and cybersecurity sectors.

📊 Dataset Summary
The dataset contains over 6 million transaction records, with the following key features:

Column Name	Description
step	Time unit where 1 step = 1 hour
type	Transaction type (TRANSFER, CASH_OUT, PAYMENT, etc.)
amount	Transaction amount
oldbalanceOrg	Sender’s balance before transaction
newbalanceOrig	Sender’s balance after transaction
oldbalanceDest	Receiver’s balance before transaction
newbalanceDest	Receiver’s balance after transaction
isFraud	Target label: 1 for fraud, 0 for non-fraud

📁 Source: PaySim Fraud Detection Dataset – Kaggle

🧠 Objective
To:

Build a predictive model using a Decision Tree Classifier.

Detect fraudulent transactions in real-time using transaction metadata.

Visualize transaction distributions and understand the risk factors.

🚀 Technologies Used
Python

Pandas, NumPy for data manipulation

Scikit-learn for model building

Plotly for interactive data visualization

Google Colab / Jupyter Notebook

🧪 How It Works (Step-by-Step)
✅ 1. Upload the Dataset
python
Copy
Edit
from google.colab import files
uploaded = files.upload()  # Upload 'credit_card.csv'
📦 2. Install Required Libraries
python
Copy
Edit
!pip install -q pandas numpy scikit-learn plotly
📂 3. Load and Clean Data
python
Copy
Edit
import pandas as pd
import numpy as np

df = pd.read_csv("credit_card.csv")
assert not df.isnull().values.any(), "Dataset contains null values!"
📊 4. Preprocess the Data
python
Copy
Edit
# Convert transaction types to numeric
df["type"] = df["type"].map({
    "CASH_OUT": 1,
    "PAYMENT": 2,
    "CASH_IN": 3,
    "TRANSFER": 4,
    "DEBIT": 5
})

# Map output labels
df["isFraud"] = df["isFraud"].map({0: "No Fraud", 1: "Fraud"})
📈 5. Visualize Transaction Type Distribution
python
Copy
Edit
import plotly.express as px

type_counts = df["type"].value_counts()
fig = px.pie(values=type_counts.values, names=type_counts.index,
             hole=0.5, title="Transaction Type Distribution")
fig.show()
🧠 6. Train the Model
python
Copy
Edit
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

X = df[["type", "amount", "oldbalanceOrg", "newbalanceOrig"]].values
y = df["isFraud"].values

xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.10, random_state=42)

model = DecisionTreeClassifier()
model.fit(xtrain, ytrain)

print("✅ Model Accuracy:", model.score(xtest, ytest))
🔍 7. Predict Fraudulent Transactions
python
Copy
Edit
# Example: TRANSFER of $9000
sample = np.array([[4, 9000.60, 9000.60, 0.0]])
print("🔮 Prediction:", model.predict(sample)[0])
📊 Results
Model Accuracy: ~99.97%

Insight: Most frauds were found in TRANSFER and CASH_OUT transactions.

Impact: Detecting fraudulent behavior early can help mitigate millions in financial losses.

🎯 Key Learnings
How to clean and transform real-world financial datasets.

Importance of feature selection and encoding in classification tasks.

Use of Decision Trees for interpretable ML models.

Creating end-to-end pipelines for fraud detection.

