# dataset.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer

def load_dataset(file_path):
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def preprocess_data(df):
    if df is None:
        return None

    columns_to_drop = [
        'CLIENTNUM',
        df.columns[-1],  # Last column
        df.columns[-2]   # Second-to-last column
    ]
    df = df.drop(columns=columns_to_drop)

    df['Attrition_Flag'] = df['Attrition_Flag'].map({
        'Attrited Customer': 1,
        'Existing Customer': 0
    })

    df = pd.get_dummies(df, columns=['Gender', 'Marital_Status', 'Income_Category', 'Card_Category', 'Education_Level'], dtype=int)

    return df

def split_data(df, test_size, random_state, preprocessing_method):
    if df is None:
        return None, None, None, None

    X = df.drop('Attrition_Flag', axis=1)
    y = df['Attrition_Flag']

    scalers = {
        'standardization': StandardScaler(),
        'normalization': Normalizer(),
        'min-max-scaling': MinMaxScaler(),
        'robust-scaling': RobustScaler()
    }

    if preprocessing_method not in scalers:
        raise ValueError("Invalid preprocessing method")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    scaler = scalers[preprocessing_method]
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test