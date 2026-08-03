# Comparación QSVM vs SVM — Replicación Suzuki et al. 2023

Replicación del paper **"Quantum support vector machines for classification and regression on a trapped-ion quantum computer"** (Suzuki et al., 2023 — [arXiv:2307.02091](https://arxiv.org/abs/2307.02091)) usando **IBM Quantum** en lugar de IonQ Harmony.

## Descripción

Este proyecto replica los experimentos de clasificación del paper comparando:

- **SVM clásica** con kernel RBF
- **QSVM (simulación noiseless)** — kernel cuántico de fidelidad simulado exactamente
- **QSVM en hardware real** — ejecutado en procesadores cuánticos IBM (`ibm_fez`, `ibm_marrakesh`)

### Datasets

| Dataset | Tarea | Clases |
|---------|-------|--------|
| Credit Card Fraud Detection | Clasificación binaria | Normal (0) vs Fraude (1) |
| MNIST | Clasificación binaria | Dígito 0 vs Dígito 1 |
| Fashion-MNIST | Clasificación binaria | T-shirt (0) vs Trouser (1) |

### Feature map (Ec. 1 del paper)

```
H → Rz(λxq) → Ry(λxq) → CNOT(cadena) → Rz(λxq)
```

Kernel cuántico de fidelidad: `K(x,x') = |⟨φ(x)|φ(x')⟩|²`

Kernel alignment (Ec. 7): `A(K,K') = ⟨K,K'⟩_F / (‖K‖_F · ‖K'‖_F)`

---

## Resultados

### Simulación noiseless (local)

| Dataset | SVM Clásico | QSVM Noiseless |
|---------|:-----------:|:--------------:|
| Credit Card | 70% | 70% |
| MNIST | 100% | 70% |
| Fashion-MNIST | 100% | 90% |

### Hardware IBM Quantum (parámetros del paper: N=20, shots=500)

| Dataset | IBM Accuracy | Paper IonQ | IBM Alignment | Paper IonQ |
|---------|:-----------:|:----------:|:-------------:|:----------:|
| Credit Card | **100%** | 70% | 0.9987 | 0.986 |
| MNIST | **80%** | 100% | 0.9985 | 0.984 |
| Fashion-MNIST | **100%** | 100% | 0.9982 | 0.993 |

> Alignment > 0.998 en los 3 datasets — el hardware IBM actual introduce menos ruido que IonQ Harmony (2023).

---

## Estructura del repositorio

```
.
├── experimentación_svm_qsvm.ipynb     # Notebook principal (simulación noiseless)
├── _run_hw.py                         # Script para hardware real IBM Quantum
├── resultados_experimentos/           # Resultados de la simulación local
│   ├── clasificacion_nuestros.csv
│   ├── clasificacion_paper.csv
│   ├── resultados_completos.xlsx
│   └── figuras/                       # Gráficas comparativas
└── resultados_hw_ibm/                 # Resultados del hardware real
    ├── credit_card_ibm.csv
    ├── mnist_ibm.csv
    ├── fashion-mnist_ibm.csv
    └── resumen_ibm.csv
```

---

## Configuración

### Datasets (no incluidos por tamaño)

Descargar y colocar en la raíz del proyecto:

- **Credit Card Fraud** (`creditcard.csv`): [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Fashion-MNIST** (carpeta `fashion-mnist-master/data/fashion/` con los 4 archivos `.gz`): [zalandoresearch/fashion-mnist](https://github.com/zalandoresearch/fashion-mnist)
- **MNIST**: se descarga automáticamente via `sklearn.datasets.fetch_openml`

### Dependencias

```bash
pip install qiskit qiskit-ibm-runtime qiskit-machine-learning \
            scikit-learn numpy pandas matplotlib seaborn openpyxl nbconvert
```

Versiones usadas:

| Paquete | Versión |
|---------|---------|
| qiskit | 2.4.0 |
| qiskit-ibm-runtime | 0.47.0 |
| qiskit-machine-learning | 0.9.0 |
| scikit-learn | 1.9.0 |

---

## Ejecución

### 1. Simulación noiseless (sin cuenta IBM)

Ejecutar el notebook `experimentación_svm_qsvm.ipynb` celda a celda, o:

```bash
python -m nbconvert --to notebook --execute --inplace "experimentación_svm_qsvm.ipynb"
```

### 2. Hardware real IBM Quantum

Crear el archivo `apikey.json` en la raíz del proyecto (no se sube al repo):

```json
{
  "apikey": "TU_API_KEY_IBM_QUANTUM"
}
```

Obtener la API key en [quantum.ibm.com](https://quantum.ibm.com) (plan Open: 10 min/mes de QPU gratis).

Ejecutar:

```bash
python _run_hw.py
```

El script selecciona automáticamente el backend real con menos cola, ejecuta los 3 experimentos y guarda los resultados en `resultados_hw_ibm/`.

#### Parámetros configurables en `_run_hw.py`

| Parámetro | Valor paper | Descripción |
|-----------|:-----------:|-------------|
| `N_QUBITS` | 4 | Dimensión reducida (PCA) |
| `LAM` | 1.0 | Escala del feature map |
| `N_TRAIN` | 20 | Muestras de entrenamiento |
| `N_TEST` | 10 | Muestras de prueba |
| `N_SHOTS` | 500 | Shots por evaluación de kernel |

> Para plan gratuito (10 min QPU) se puede reducir a `N_TRAIN=10, N_TEST=5, N_SHOTS=100`.

---

## Diferencias respecto al paper

| Aspecto | Paper (Suzuki et al.) | Este proyecto |
|---------|----------------------|---------------|
| Hardware | IonQ Harmony (11 qubits) | IBM Quantum (`ibm_fez`, `ibm_marrakesh`, 156 qubits) |
| Modo ejecución | Circuitos individuales | Job mode (plan Open) |
| Experimentos | Clasificación + Regresión | Solo clasificación (3 datasets) |
| Framework | No especificado | Qiskit 2.4 + qiskit-machine-learning 0.9 |

---

## Referencia

Suzuki, Y., Kawase, Y., Masumura, Y., Hiraga, Y., Nakadai, M., Chen, J., ... & Yamamoto, N. (2023). *Quantum support vector machines for classification and regression on a trapped-ion quantum computer*. [arXiv:2307.02091](https://arxiv.org/abs/2307.02091).
