import pandas as pd
import yaml
import joblib
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def load_config(config_path="config/params.yaml"):
    """Carga la configuración desde el archivo YAML"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    # 1. Cargar configuración
    config = load_config()
    
    # Extraer parámetros para usarlos fácilmente
    data_cfg = config['data']
    model_cfg = config['model']
    output_cfg = config['output']

    print(f"🚀 Iniciando entrenamiento con config: {model_cfg['name']}")
    print(f"📊 Usando {model_cfg['n_estimators']} estimadores...")

    # 2. Cargar datos (usamos Iris como ejemplo seguro)
    from sklearn.datasets import load_iris
    data = load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target

    # 3. Preparar datos usando los params del YAML
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Usamos test_size y random_state del config
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=data_cfg['test_size'], 
        random_state=data_cfg['random_state']
    )

    # 4. Entrenar modelo usando los params del YAML
    model = RandomForestClassifier(
        n_estimators=model_cfg['n_estimators'],
        max_depth=model_cfg['max_depth'],
        random_state=model_cfg['random_state']
    )
#hago un comentario
    
    model.fit(X_train, y_train)

    # 5. Evaluar
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"✅ Accuracy: {acc:.4f}")

    # 6. Guardar modelo y métricas
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, output_cfg['model_path'])
    
    # Guardar métricas en JSON (buena práctica MLOps)
    metrics = {"accuracy": acc, "model_params": model_cfg}
    with open(output_cfg['metrics_path'], 'w') as f:
        json.dump(metrics, f, indent=4)
        
    print(f"💾 Modelo guardado en: {output_cfg['model_path']}")
    print(f"📈 Métricas guardadas en: {output_cfg['metrics_path']}")

if __name__ == "__main__":
    main()