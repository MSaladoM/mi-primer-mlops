# 🚀 Mi Primer Proyecto MLOps

Este repositorio demuestra una estructura profesional para proyectos de Machine Learning, enfocada en la **reproducibilidad**, **modularidad** y **experimentación automatizada**.

El objetivo es entrenar modelos de clasificación (usando el dataset Iris) comparando diferentes algoritmos (Random Forest, SVM, Naive Bayes) y hiperparámetros, sin depender de herramientas complejas como Docker o MLflow en esta etapa inicial.

## 📂 Estructura del Proyecto

```text
mi-primer-mlops/
├── config/
│   └── params.yaml          # Configuración centralizada de hiperparámetros
├── src/
│   ├── __init__.py
│   ├── data_loader.py       # Módulo de carga y preprocesamiento de datos
│   ├── model_trainer.py     # Lógica de entrenamiento, evaluación y guardado
│   ├── predict.py           # Script para inferencia con el mejor modelo
│   └── main.py              # Punto de entrada para entrenamiento único
├── models/                  # Carpeta donde se guardan los modelos .pkl
├── experiments.py           # Script para ejecutar Grid Search automático
├── analyze_results.py       # Script para analizar y ranker experimentos
├── requirements.txt         # Dependencias del proyecto
├── .gitignore               # Archivos ignorados por Git
└── README.md                # Este archivo
```

## 🛠️ Instalación y Configuración

### 1. Clonar el repositorio
```bash
git clone https://github.com/MSaladoM/mi-primer-mlops.git
cd mi-primer-mlops
```

### 2. Crear entorno virtual
```powershell
# En Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Instalar dependencias
```powershell
pip install -r requirements.txt
```

## 🧪 Cómo Usar

### A. Entrenamiento Básico
Para entrenar un solo modelo con la configuración por defecto (`config/params.yaml`):

```powershell
python -m src.main
```

### B. Experimentación Automatizada (Grid Search)
Para probar múltiples combinaciones de modelos y parámetros definidos en `experiments.py`:

```powershell
python experiments.py
```
*Esto generará varios archivos `.pkl` en la carpeta `models/` y un registro en `experiments_log.json`.*

### C. Análisis de Resultados
Para ver un ranking de los mejores modelos basados en accuracy:

```powershell
python analyze_results.py
```
*Este comando también actualiza el archivo `best_model.txt` con la ruta del modelo ganador.*

### D. Predicción (Inferencia)
Para probar el mejor modelo encontrado con datos de ejemplo:

```powershell
python src/predict.py
```

## ⚙️ Configuración

Puedes modificar los hiperparámetros editando el archivo `config/params.yaml` o ajustando el diccionario `PARAM_GRID` en `experiments.py`.

Ejemplo de `params.yaml`:
```yaml
data:
  test_size: 0.2
  random_state: 42

model:
  name: "RandomForest"
  n_estimators: 100
  max_depth: 5
```

## 📊 Tecnologías Utilizadas

*   **Python 3.10+**
*   **Scikit-Learn**: Para modelos ML y preprocessing.
*   **Pandas & NumPy**: Para manejo de datos.
*   **PyYAML**: Para lectura de configuraciones.
*   **Joblib**: Para serialización eficiente de modelos.
*   **Git**: Para control de versiones.

## 🎯 Conceptos MLOps Aplicados

1.  **Separación de Responsabilidades**: Código de datos, entrenamiento e inferencia en módulos distintos.
2.  **Configuración Externa**: Hiperparámetros fuera del código fuente.
3.  **Reproducibilidad**: Seeds fijos y versionado de experimentos via JSON.
4.  **Automatización**: Scripts para correr baterías de pruebas sin intervención manual.

## 👤 Autor

*   **Mariana Salado** - [MSaladoM](https://github.com/MSaladoM)


¿Te gusta este formato? Es limpio, directo y muestra que sabes organizar el código, no solo entrenar modelos. 🚀
