# -*- coding: utf-8 -*-

# Import Packages
import pandas as pd
import numpy as np
import sklearn.preprocessing as preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from lightgbm import LGBMClassifier
import re

def load_data(filepath):
    # Load dataset
    data = pd.read_csv(filepath, skipinitialspace=True)
    # Remove any 'Unnamed:' columns
    data = data.loc[:, ~data.columns.str.startswith('Unnamed: ')]
    # Rename columns
    data.columns = [re.sub('[^A-Za-z_%#\(\)\+\-\.\?\!\<\>\=가-힣ㄱ-ㅎㅏ-ㅣ0-9]', '_', col.strip()) for col in data.columns]
    # Drop uninterested columns
    data = data.drop(['주용도(동)', '대표용도(동)', '주구조(동)', '기타구조(동)'], axis=1)

    return data

def preprocess_data(data, target_col):
    # Split X and y
    X = data.drop(target_col, axis=1)
    y = data[target_col].values

    # Encode categorical target
    label_encoder = preprocessing.LabelEncoder()
    y = label_encoder.fit_transform(y.astype(str).ravel())

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Classify column types
    cat_columns = X_train.select_dtypes(include='object').columns

    # Encode categorical columns
    encoder = preprocessing.OrdinalEncoder(
        handle_unknown='use_encoded_value', unknown_value=-1)
    train_cat_values = encoder.fit_transform(X_train[cat_columns])
    test_cat_values = encoder.transform(X_test[cat_columns])

    # Rebuild train and test dataset
    X_train = train_cat_values
    X_test = test_cat_values

    return X_train, X_test, y_train, y_test, label_encoder

def train_model(X_train, y_train):
    # Build a model with best possible parameters
    model = LGBMClassifier(**{
        "n_estimators": 185,
        "num_leaves": 21,
        "min_child_samples": 5,
        "learning_rate": 0.2831716718674683,
        "log_max_bin": 10,
        "colsample_bytree": 0.6013779332617044,
        "reg_alpha": 0.0009765625,
        "reg_lambda": 0.0009765625
    })

    # And train (fit) it
    model.fit(X_train, y_train)

    return model

def evaluate_model(model, X_test, y_test, label_encoder):
    # Predict using the test dataset
    y_test = label_encoder.inverse_transform(y_test)
    y_pred = label_encoder.inverse_transform(model.predict(X_test))
    y_yhat = pd.DataFrame({'Real': y_test, 'Pred': y_pred})
    print(y_yhat.reset_index(drop=True), '\n')

    # Calculate performance metrics
    print("Classification Report")
    print(classification_report(y_test, y_pred), '\n')

def main():
    # Load and explore data
    data = load_data("modified_APP_inventor.csv")
    print('Dataset Shape:', data.shape, '\n')
    print(data.head(), '\n')
    data.info()
    print()
    print(data.describe(), '\n')

    # Preprocess data
    X_train, X_test, y_train, y_test, label_encoder = \
        preprocess_data(data, '허가_비허가')

    # Train and evaluate model
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test, label_encoder)

if __name__ == "__main__":
    main()
