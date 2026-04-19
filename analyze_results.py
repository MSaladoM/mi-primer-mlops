import pandas as pd
import json
import os

def analyze_experiments(log_file="experiments_log.json"):
    # 1. Verificar si existe el archivo
    if not os.path.exists(log_file):
        print(f"❌ No se encontró el archivo {log_file}. ¿Has ejecutado 'experiments.py'?")
        return

    # 2. Cargar datos
    with open(log_file, 'r') as f:
        data = json.load(f)

    if not data:
        print("📭 El log de experimentos está vacío.")
        return

    # 3. Normalizar los datos para Pandas
    # Los parámetros están anidados en un diccionario, vamos a "aplanarlos"
    rows = []
    for exp in data:
        row = {
            "timestamp": exp['timestamp'],
            "experiment_name": exp['name'],
            "accuracy": exp['accuracy'],
            "model_name": exp['params'].get('name', 'Unknown')
        }
        # Añadir hiperparámetros clave si existen
        row['n_estimators'] = exp['params'].get('n_estimators', '-')
        row['max_depth'] = exp['params'].get('max_depth', '-')
        row['C_SVM'] = exp['params'].get('C', '-')
        row['kernel'] = exp['params'].get('kernel', '-')
        row['test_size'] = exp['params'].get('test_size', '-') # Ojo: esto suele estar en data, ajustalo si tu estructura varia
        
        rows.append(row)

    # 4. Crear DataFrame
    df = pd.DataFrame(rows)

    # 5. Ordenar por Accuracy (de mayor a menor)
    df_sorted = df.sort_values(by="accuracy", ascending=False).reset_index(drop=True)

    # 6. Mostrar Resultados
    print("\n" + "="*60)
    print("🏆 RANKING DE EXPERIMENTOS (Top 10)")
    print("="*60)
    
    # Seleccionamos columnas relevantes para mostrar
    cols_to_show = ["rank", "model_name", "accuracy", "n_estimators", "max_depth", "C_SVM", "kernel"]
    
    # Añadimos columna de rango visual
    df_sorted['rank'] = range(1, len(df_sorted) + 1)
    
    # Imprimir tabla bonita
    print(df_sorted[cols_to_show].head(10).to_string(index=False))
    
    print("\n" + "="*60)
    print(f"📈 Mejor Modelo: {df_sorted.iloc[0]['experiment_name']}")
    print(f"🎯 Accuracy Máxima: {df_sorted.iloc[0]['accuracy']:.4f}")
    print("="*60)

    # 7. (Opcional) Guardar resumen en CSV
    df_sorted.to_csv("results_summary.csv", index=False)
    print("\n💾 Resumen completo guardado en 'results_summary.csv'")

if __name__ == "__main__":
    analyze_experiments()