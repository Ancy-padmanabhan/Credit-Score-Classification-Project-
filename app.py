from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the model, preprocessor, and label encoder from the pickle file
with open('credit_score_model.pkl', 'rb') as file:
    saved_objects = pickle.load(file)
    model = saved_objects['model']
    preprocessor = saved_objects['preprocessor']
    label_encoder = saved_objects['label_encoder']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from form
    form_data = request.form

    # Map form inputs to expected order in the model pipeline
    input_data = {
        'Age': float(form_data['Age']),
        'Occupation': form_data['Occupation'],
        'Annual_Income': float(form_data['Annual_Income']),
        'Num_Bank_Accounts': float(form_data['Num_Bank_Accounts']),
        'Num_Credit_Card': float(form_data['Num_Credit_Card']),
        'Interest_Rate': float(form_data['Interest_Rate']),
        'Num_of_Loan': float(form_data['Num_of_Loan']),
        'Type_of_Loan': form_data['Type_of_Loan'],
        'Delay_from_due_date': float(form_data['Delay_from_due_date']),
        'Num_of_Delayed_Payment': float(form_data['Num_of_Delayed_Payment']),
        'Changed_Credit_Limit': float(form_data['Changed_Credit_Limit']),
        'Num_Credit_Inquiries': float(form_data['Num_Credit_Inquiries']),
        'Outstanding_Debt': float(form_data['Outstanding_Debt']),
        'Credit_Utilization_Ratio': float(form_data['Credit_Utilization_Ratio']),
        'Credit_History_Age': float(form_data['Credit_History_Age']),
        'Payment_of_Min_Amount': form_data['Payment_of_Min_Amount'],
        'Total_EMI_per_month': float(form_data['Total_EMI_per_month']),
        'Amount_invested_monthly': float(form_data['Amount_invested_monthly']),
        'Payment_Behaviour': form_data['Payment_Behaviour'],
        'Monthly_Balance': float(form_data['Monthly_Balance'])
    }

    # Convert the input data into a DataFrame
    input_df = pd.DataFrame([input_data])

    # Apply the preprocessor to the input data
    X_processed = preprocessor.transform(input_df)

    # Predict the credit score using the loaded model
    prediction = model.predict(X_processed)[0]

    # Convert numerical prediction back to label using the label encoder's inverse transform
    prediction_label = label_encoder.inverse_transform([prediction])[0]

    # Render the prediction result in the web page
    return render_template('result.html',prediction_label=prediction_label)

if __name__ == "__main__":
    app.run(debug=True)
