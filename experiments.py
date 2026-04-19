import itertools
import json
import os
from datetime import datetime
from src.data_loader import load_and_split_data
from src.model_trainer import train_and_save_model

# 1. Define los parámetros y MODELOS a probar
# Nota: Algunos parámetros son específicos de ciertos modelos. 
# La función get_model ignorará los que no le sirvan o usará defaults.
PARAM_GRID = {
    "model_name": ["RandomForest", "SVM", "NaiveBayes"], # <-- NUEVO
    "n_estimators": [50, 100],       # Solo para RF
    "max_depth": [5, None],          # Solo para RF
    "C": [0.1, 1.0],                 # Solo para SVM
    "kernel": ['rbf', 'linear'],     # Solo para SVM
    "test_size": [0.2]
}

def generate_experiments(param_grid):
    keys = param_grid.keys()
    values = param_grid.values()
    
    experiments = []
    for combination in itertools.product(*values):
        params = dict(zip(keys, combination))
        
        model_name = params['model_name']
        
        # Generar nombre limpio para el archivo
        name = f"{model_name}"
        if model_name == "RandomForest":
            name += f"_n{params['n_estimators']}_d{params['max_depth']}"
        elif model_name == "SVM":
            name += f"_C{params['C']}_k{params['kernel']}"
            
        name += f"_ts{params['test_size']}"

        # Construir el config completo
        # Aquí está la clave: pasamos TODOS los params al config['model']
        # get_model() se encargará de usar solo los que necesita.
        exp_config = {
            "name": name,
            "params": {
                "data": {"test_size": params['test_size'], "random_state": 42},
                "model": {
                    "name": model_name,
                    "n_estimators": params.get('n_estimators'),
                    "max_depth": params.get('max_depth'),
                    "C": params.get('C'),
                    "kernel": params.get('kernel'),
                    "random_state": 42
                },
                "output": {"model_path": f"models/{name}.pkl"}
            }
        }
        experiments.append(exp_config)
        
    return experiments

# ... (El resto de run_experiment y save_log sigue igual que antes) ...

def run_experiment(exp_config):
    name = exp_config['name']
    params = exp_config['params']
    print(f"\n🚀 Ejecutando: {name}")
    try:
        X_train, X_test, y_train, y_test = load_and_split_data(
            test_size=params['data']['test_size'],
            random_state=params['data']['random_state']
        )
        acc = train_and_save_model(
            X_train, X_test, y_train, y_test,
            config=params,
            output_path=params['output']['model_path']
        )
        save_log(name, params['model'], acc)
        print(f"   ✅ Accuracy: {acc:.4f}")
    except Exception as e:
        print(f"   ❌ Error: {e}")

def save_log(name, model_params, accuracy):
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
    print("⚙️  Generando experimentos multi-modelo...")
    all_experiments = generate_experiments(PARAM_GRID)
    print(f"📋 Se ejecutarán {len(all_experiments)} experimentos.\n")
    for exp in all_experiments:
        run_experiment(exp)
    print("\n🏁 ¡Terminado!")