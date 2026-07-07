# Comparación QSVM vs SVM

Replicación y extensión del paper **"Quantum support vector machines for classification and regression on a trapped-ion quantum computer"** (Suzuki et al., 2023 — [arXiv:2307.02091](https://arxiv.org/abs/2307.02091)).

## Descripción

Este proyecto compara Support Vector Machines clásicas (SVM) contra Quantum Support Vector Machines (QSVM) en tareas de clasificación y regresión, usando los siguientes datasets:

- **Credit Card Fraud Detection** — clasificación binaria
- **Fashion-MNIST** — clasificación multiclase (subconjunto)
- **MNIST** — clasificación multiclase (subconjunto)
- **Superconductividad** — regresión

## Estructura

```
.
├── experimentación_svm_qsvm.ipynb   # Notebook principal con todos los experimentos
├── _run_hw.py                       # Script para ejecutar en hardware real de IBM Quantum
└── resultados_experimentos/
    ├── figuras/                     # Gráficas comparativas (métricas, kernels, matrices de confusión)
    ├── resultados_completos.xlsx    # Tabla resumen de todos los resultados
    ├── clasificacion_nuestros.csv   # Resultados de clasificación (replicación propia)
    ├── clasificacion_paper.csv      # Resultados reportados en el paper original
    ├── regresion_nuestros.csv       # Resultados de regresión (replicación propia)
    └── regresion_paper.csv          # Resultados de regresión del paper original
```

## Datasets

Los datasets **no están incluidos** en el repositorio por su tamaño. Descárgalos desde:

- **Credit Card Fraud**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Fashion-MNIST**: [zalandoresearch/fashion-mnist](https://github.com/zalandoresearch/fashion-mnist)
- **MNIST**: disponible via `torchvision.datasets.MNIST` o `sklearn.datasets`
- **Superconductividad**: [UCI - Superconductivty Data](https://archive.ics.uci.edu/dataset/464/superconductivty+data)

## Dependencias

```
qiskit
qiskit-machine-learning
scikit-learn
numpy
pandas
matplotlib
seaborn
openpyxl
```

Instalar con:
```bash
pip install qiskit qiskit-machine-learning scikit-learn numpy pandas matplotlib seaborn openpyxl
```

## IBM Quantum (hardware real)

Para ejecutar `_run_hw.py` en hardware cuántico real se necesita una API key de [IBM Quantum](https://quantum.ibm.com/). Configurarla en la celda correspondiente del notebook antes de correr el script.

## Referencia

Suzuki, Y. et al. (2023). *Quantum support vector machines for classification and regression on a trapped-ion quantum computer*. arXiv:2307.02091.
