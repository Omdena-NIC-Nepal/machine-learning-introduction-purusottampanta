#Load necessary libraries and packages

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def getDataframe():
    # Loading Dataset
    data_path = '../data/boston_housing.csv'
    # Read CSV file and convert na values in the data to np.nan
    return pd.read_csv(data_path,  na_values='NA') 

def describeDataProperties(df):
    #View Overview of Data
    print("Data Structure:")
    print(df.info())

    print("\nSummary Statistics:")
    print(df.describe())

    #Check if there are any missing values 
    missing_values = df.isnull().sum()
    print("\nMissing Values per Column:")
    print(missing_values)

def handleMissingValues(df):
    #Handling missing values
    imputer = SimpleImputer(strategy='median')
    return pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

def normalizeData(df):
    # Standardize Numerical Features to make mean 0
    scaler = StandardScaler()
    numerical_features = df.drop(columns=['MEDV']).columns  # Exclude target variable
    df[numerical_features] = scaler.fit_transform(df[numerical_features])

    return df

def splitData(df):
    X = df.drop(columns=['MEDV'])  # Features
    y = df['MEDV']  # Target


    return train_test_split(X, y, test_size=0.2, random_state=42)

def preprocessAndSplit():
    df = getDataframe()
    df = handleMissingValues(df)
    df = normalizeData(df)
    return splitData(df)
    