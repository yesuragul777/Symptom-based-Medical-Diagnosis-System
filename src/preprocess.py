import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, chi2

def preprocess_data(filepath, target_column='Disease', k_features=8):
    df = pd.read_csv(filepath)
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Feature selection
    selector = SelectKBest(score_func=chi2, k=k_features)
    X_new = selector.fit_transform(X, y)

    # Normalization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_new)

    return X_scaled, y
