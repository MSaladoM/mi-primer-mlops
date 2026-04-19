import yaml
from src.data_loader import load_and_split_data
from src.model_trainer import train_and_save_model

def load_config(config_path="config/params.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    # 1. Cargar Configuración
    print("⚙️  Cargando configuración...")
    config = load_config()
    
    # 2. Cargar Datos (usando módulo externo)
    X_train, X_test, y_train, y_test = load_and_split_data(
        test_size=config['data']['test_size'],
        random_state=config['data']['random_state']
    )

    # 3. Entrenar (usando módulo externo)
    train_and_save_model(
        X_train, X_test, y_train, y_test,
        config=config,
        output_path=config['output']['model_path']
    )
    
    print("🎉 Pipeline completado exitosamente.")

if __name__ == "__main__":
    main()