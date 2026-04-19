import joblib
import json
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_and_save_model(X_train, X_test, y_train, y_test, config, output_path="models/model_v1.pkl"):
    """
    Entrena un modelo RandomForest, lo evalúa y lo guarda.
    """
    print("🤖 Entrenando modelo...")
    
    # Inicializar modelo con params del config
    model = RandomForestClassifier(
        n_estimators=config['model']['n_estimators'],
        max_depth=config['model']['max_depth'],
        random_state=config['model']['random_state']
    )
    
    model.fit(X_train, y_train)
    
    # Evaluar
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"✅ Accuracy: {acc:.4f}")

    # Guardar Modelo
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(model, output_path)
    print(f"💾 Modelo guardado en: {output_path}")

    # Guardar Métricas
    metrics = {"accuracy": acc, "params": config['model']}
    with open("metrics.json", 'w') as f:
        json.dump(metrics, f, indent=4)
    print("📈 Métricas guardadas en: metrics.json")
    
    return acc