import joblib
import numpy as np
import os

def load_model(model_path="models/model_v1.pkl"):
    """
    Carga el modelo desde un archivo .pkl.
    Lanza error si no existe.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Modelo no encontrado en: {model_path}. ¿Entrenaste primero?")
    
    print(f"📦 Cargando modelo desde: {model_path}")
    model = joblib.load(model_path)
    return model

def predict(model, data):
    """
    Realiza predicciones sobre datos nuevos.
    data: array-like o lista de listas.
    """
    # Aseguramos que sea un array de numpy para sklearn
    data_array = np.array(data).reshape(1, -1) if len(np.array(data).shape) == 1 else np.array(data)
    
    predictions = model.predict(data_array)
    return predictions

if __name__ == "__main__":
    # --- EJEMPLO DE USO LOCAL ---
    
    # 1. Cargar el mejor modelo (puedes cambiar la ruta manualmente o leerla de un config)
    # Para este ejemplo, usamos uno genérico, pero en producción usarías el ganador del ranking
    model_path = "models/model_v1.pkl" 
    
    try:
        model = load_model(model_path)
        
        # 2. Datos de ejemplo (4 características de Iris: sepal length, width, petal length, width)
        new_data = [5.1, 3.5, 1.4, 0.2] # Debería ser clase 0 (Setosa)
        
        print(f"🔮 Prediciendo para: {new_data}")
        result = predict(model, new_data)
        
        print(f"✅ Clase predicha: {result[0]}")
        
    except Exception as e:
        print(f"Error: {e}")