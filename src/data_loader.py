import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits

def load_and_split_data(test_size=0.2, random_state=42):
    """
    Carga el dataset Iris y lo divide en train/test.
    Retorna: X_train, X_test, y_train, y_test
    """
    print("📊 Cargando datos...")
    data = load_digits()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target

    X = df.drop('target', axis=1)
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state
    )
    
    print(f"✅ Datos cargados. Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test