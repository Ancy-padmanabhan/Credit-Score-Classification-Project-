#import the libraries 
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import category_encoders as ce
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,accuracy_score

#Load the dataset to dataframe 'data'
data=pd.read_csv('credit_preprocessed.csv')

#Encode the Target variable Credit_Score using the Label Encoder
lb_encoder=LabelEncoder()
data['Credit_Score']=lb_encoder.fit_transform(data['Credit_Score'])

#Separate features and target
X = data.drop('Credit_Score',axis=1)  # Features
y = data['Credit_Score']               # Target variable

# Define the column transformer for binary encoding, target encoding
preprocessor = ColumnTransformer(transformers=[
    # Apply binary encoding to these categorical columns
    ('binary_encoder', ce.BinaryEncoder(cols=['Occupation', 'Payment_of_Min_Amount', 'Payment_Behaviour']), 
     ['Occupation', 'Payment_of_Min_Amount', 'Payment_Behaviour']),
    
    # Apply target encoding to the 'Type_of_Loan' column
    ('target_encoder', ce.TargetEncoder(cols=['Type_of_Loan']), ['Type_of_Loan']),], remainder='passthrough')

# Build the full pipeline including SMOTE
preprocessing_pipeline = Pipeline(steps=[
    # Step 1: Apply the preprocessing steps (encoding, log transformation)
    ('preprocessor', preprocessor),
    
    # Step 2: Apply scaling to all features
    ('scaler', StandardScaler()),
])

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=32)

# Further split the training set into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=32)

# Check the sizes of the splits
print(f'X_train: {X_train.shape}, y_train: {y_train.shape}')
print(f'X_val: {X_val.shape}, y_val: {y_val.shape}')
print(f'X_test: {X_test.shape}, y_test: {y_test.shape}')

# Apply the preprocessing pipeline to the training data (including y for target encoding)
X_train_preprocessed = preprocessing_pipeline.fit_transform(X_train,y_train)
# Apply the preprocessing pipeline to the test data 
X_val_preprocessed=preprocessing_pipeline.transform(X_val)
X_test_preprocessed = preprocessing_pipeline.transform(X_test)

# Using SMOTE for over_sampling on train data
from imblearn.over_sampling import SMOTE
# Apply SMOTE
smote = SMOTE(random_state=42)
# Apply SMOTE to the preprocessed training data
X_balanced, y_balanced = smote.fit_resample(X_train_preprocessed, y_train)

#Fit the model using the resampled training data
model = RandomForestClassifier(n_estimators= 400,min_samples_split= 2,min_samples_leaf=1,max_features= 'sqrt',max_depth= 100,bootstrap= False,random_state=42)
model.fit(X_balanced, y_balanced)

# Validate the model on the validation set
y_pred = model.predict(X_test_preprocessed)

# Output the validation predictions and evaluate the model
print("Results:")
print(classification_report(y_test, y_pred))
print(f"Model Accuracy: {accuracy_score(y_test, y_pred)}")

import pickle

# Save the model, preprocessor, and label encoder to a pickle file
with open('credit_score_model.pkl', 'wb') as file:
    pickle.dump({
        'model': model,
        'preprocessor': preprocessing_pipeline,
        'label_encoder': lb_encoder
    }, file)

print("Model and pipeline saved successfully.")


