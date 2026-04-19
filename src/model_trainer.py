import joblib
import json
import os
from sklearn.metrics import accuracy_score
# Importa los modelos que quieras probar
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

def get_model(model_name, params):
    """
    Fábrica de modelos: Devuelve la instancia del modelo según el nombre.
    """
    if model_name == "RandomForest":
        return RandomForestClassifier(
            n_estimators=params.get('n_estimators', 100),
            max_depth=params.get('max_depth', 5),
            random_state=params.get('random_state', 42)
        )
    elif model_name == "SVM":
        # SVM usa C y kernel, no n_estimators
        return SVC(
            C=params.get('C', 1.0),
            kernel=params.get('kernel', 'rbf'),
            random_state=params.get('random_state', 42)
        )
    elif model_name == "NaiveBayes":
        # Naive Bayes no tiene hiperparámetros complejos usualmente
        return GaussianNB()
    
    elif model_name == "LogisticRegression":
        return LogisticRegression(
            max_iter=params.get('max_iter', 100),
            random_state=params.get('random_state', 42)
        )
    else:
        raise ValueError(f"Modelo '{model_name}' no reconocido.")

def train_and_save_model(X_train, X_test, y_train, y_test, config, output_path="models/model.pkl"):
    """
    Entrena el modelo especificado en el config, evalúa y guarda.
    """
    model_name = config['model'].get('name', 'RandomForest')
    print(f"🤖 Entrenando modelo: {model_name}...")
    
    # 1. Obtener la instancia del modelo
    model = get_model(model_name, config['model'])
    
    # 2. Entrenar (todas las clases de sklearn tienen .fit)
    model.fit(X_train, y_train)
    
    # 3. Evaluar (todas tienen .predict)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"✅ Accuracy ({model_name}): {acc:.4f}")

    # 4. Guardar
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(model, output_path)
    print(f"💾 Modelo guardado en: {output_path}")

    # 5. Guardar Métricas
    metrics = {
        "accuracy": acc, 
        "model_name": model_name,
        "params": config['model']
    }
    
    # Guardamos en un archivo único o separado, aquí lo simplificamos
    with open("metrics.json", 'w') as f:
        json.dump(metrics, f, indent=4)
        
    return acc