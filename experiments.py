import itertools
import json
import os
from datetime import datetime
from src.data_loader import load_and_split_data
from src.model_trainer import train_and_save_model

# 1. Define los rangos de parámetros que quieres probar
PARAM_GRID = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 10, None],  # None significa profundidad ilimitada
    "test_size": [0.2, 0.3]
}

def generate_experiments(param_grid):
    """
    Genera una lista de diccionarios con todas las combinaciones posibles
    de los parámetros definidos en PARAM_GRID.
    """
    keys = param_grid.keys()
    values = param_grid.values()
    
    experiments = []
    # itertools.product crea todas las combinaciones: (50, 3, 0.2), (50, 3, 0.3)...
    for combination in itertools.product(*values):
        # Creamos un diccionario para esta combinación específica
        params = dict(zip(keys, combination))
        
        # Generamos un nombre único para el experimento
        name = (f"rf_n{params['n_estimators']}"
                f"_d{params['max_depth']}"
                f"_ts{params['test_size']}")
        
        # Estructura completa del config (igual que antes)
        exp_config = {
            "name": name,
            "params": {
                "data": {"test_size": params['test_size'], "random_state": 42},
                "model": {
                    "n_estimators": params['n_estimators'],
                    "max_depth": params['max_depth'],
                    "random_state": 42
                },
                "output": {"model_path": f"models/{name}.pkl"}
            }
        }
        experiments.append(exp_config)
        
    return experiments

def run_experiment(exp_config):
    """Ejecuta un solo experimento"""
    name = exp_config['name']
    params = exp_config['params']
    
    print(f"\n🚀 Ejecutando: {name}")
    
    try:
        # 1. Datos
        X_train, X_test, y_train, y_test = load_and_split_data(
            test_size=params['data']['test_size'],
            random_state=params['data']['random_state']
        )

        # 2. Entrenar
        acc = train_and_save_model(
            X_train, X_test, y_train, y_test,
            config=params,
            output_path=params['output']['model_path']
        )

        # 3. Log
        save_log(name, params['model'], acc)
        print(f"   ✅ Accuracy: {acc:.4f}")

    except Exception as e:
        print(f"   ❌ Error: {e}")

def save_log(name, model_params, accuracy):
    """Guarda resultados en experiments_log.json"""
    log_file = "experiments_log.json"
    logs = []
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            logs = json.load(f)

    logs.append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "name": name,
        "params": model_params,
        "accuracy": accuracy
    })

    with open(log_file, 'w') as f:
        json.dump(logs, f, indent=4)

if __name__ == "__main__":
    print("⚙️  Generando combinaciones de parámetros...")
    all_experiments = generate_experiments(PARAM_GRID)
    print(f"📋 Se ejecutarán {len(all_experiments)} experimentos.\n")
    
    for exp in all_experiments:
        run_experiment(exp)
        
    print("\n🏁 ¡Terminado! Revisa 'experiments_log.json' para ver el mejor modelo.")