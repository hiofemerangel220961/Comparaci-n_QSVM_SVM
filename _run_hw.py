import warnings
warnings.filterwarnings('ignore')
import json, re, sys, time, os
import numpy as np
import pandas as pd
from scipy.special import expit

# Leer API key del notebook (celda 27)
nb  = json.load(open('quantum_credit_comparison.ipynb', encoding='utf-8'))
src = ''.join(nb['cells'][27]['source'])
m   = re.search(r'IBM_QUANTUM_API_KEY\s*=\s*["\']([^"\']+)["\']', src)
if not m or m.group(1) == 'TU_API_KEY_AQUI':
    print("API key no encontrada en celda 27. Asegurate de haber guardado el notebook.")
    sys.exit(1)
IBM_QUANTUM_API_KEY  = m.group(1)
IBM_QUANTUM_INSTANCE = "crn:v1:bluemix:public:quantum-computing:us-east:a/240c2b3cf17d4fd7b24544362df6e26d:238d2c0a-d6b9-4d7e-a27d-250173ec09a9::"
print("API key OK: " + IBM_QUANTUM_API_KEY[:6] + "****")

# CONFIG
N_FEATURES   = 4
N_TRAIN_HW   = 30
N_TEST_HW    = 14
RANDOM_STATE = 42
OUTPUT_DIR   = 'resultados_experimentales'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Imports
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, balanced_accuracy_score, matthews_corrcoef,
    confusion_matrix, roc_curve, precision_recall_curve, classification_report)
from sklearn.decomposition import PCA
try:
    from imblearn.under_sampling import RandomUnderSampler
    IMBLEARN = True
except ImportError:
    IMBLEARN = False

from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.state_fidelities import ComputeUncompute
from qiskit_machine_learning.algorithms import QSVC

# Datos
print("\n[1] Preprocesando datos...")
df = pd.read_csv('creditcard.csv').drop_duplicates()
X  = df.drop('Class', axis=1)
y  = df['Class'].values
sc = StandardScaler()
Xs = X.copy()
Xs[['Time','Amount']] = sc.fit_transform(X[['Time','Amount']])
Xa = Xs.values

if IMBLEARN:
    X_bal, y_bal = RandomUnderSampler(sampling_strategy=0.5, random_state=RANDOM_STATE).fit_resample(Xa, y)
else:
    rng  = np.random.RandomState(RANDOM_STATE)
    i0   = rng.choice(np.where(y==0)[0], size=np.sum(y==1)*2, replace=False)
    idx  = np.concatenate([i0, np.where(y==1)[0]]); rng.shuffle(idx)
    X_bal, y_bal = Xa[idx], y[idx]

X_tr, X_te, y_tr, y_te = train_test_split(X_bal, y_bal, test_size=0.2,
                                            random_state=RANDOM_STATE, stratify=y_bal)

def bsample(X, y, n, seed):
    rng = np.random.RandomState(seed); n2 = n // 2
    i0  = rng.choice(np.where(y==0)[0], min(n2, np.sum(y==0)), replace=False)
    i1  = rng.choice(np.where(y==1)[0], min(n2, np.sum(y==1)), replace=False)
    idx = np.concatenate([i0,i1]); rng.shuffle(idx)
    return X[idx], y[idx]

X_tr_hw, y_tr_hw = bsample(X_tr, y_tr, N_TRAIN_HW, RANDOM_STATE+1)
X_te_hw, y_te_hw = bsample(X_te, y_te, N_TEST_HW,  RANDOM_STATE+1)

pca = PCA(n_components=N_FEATURES, random_state=RANDOM_STATE).fit(X_tr)
sq  = MinMaxScaler(feature_range=(0, float(np.pi))).fit(pca.transform(X_tr))
X_tr_hw_q = sq.transform(pca.transform(X_tr_hw))
X_te_hw_q = sq.transform(pca.transform(X_te_hw))
print("Train:", X_tr_hw_q.shape, "| Test:", X_te_hw_q.shape)

# IBM Quantum
print("\n[2] Conectando a IBM Quantum...")
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit import transpile

QiskitRuntimeService.save_account(channel="ibm_quantum_platform",
    token=IBM_QUANTUM_API_KEY, instance=IBM_QUANTUM_INSTANCE, overwrite=True)
service = QiskitRuntimeService(channel="ibm_quantum_platform",
    token=IBM_QUANTUM_API_KEY, instance=IBM_QUANTUM_INSTANCE)

def _nq(b):
    v = getattr(b,"num_qubits",0)
    return len(v) if isinstance(v,(list,tuple)) else int(v)
def _pj(b):
    try: return int(getattr(b.status(),"pending_jobs",0))
    except: return 0

avail = [b for b in service.backends() if not getattr(b,"simulator",True) and _nq(b) >= N_FEATURES]
bk    = min(avail, key=_pj)
print("Backend:", bk.name, "| Cola:", _pj(bk), "jobs")

fm = ZZFeatureMap(feature_dimension=N_FEATURES, reps=2, entanglement='full')
pm = generate_preset_pass_manager(backend=bk, optimization_level=1)

dummy = fm.assign_parameters({p: 0.5 for p in fm.parameters})
tsp   = transpile(dummy, backend=bk, optimization_level=1)
print("Depth orig:", dummy.decompose().depth(), "| transpilado:", tsp.depth())

# QSVC Hardware Real
print("\n[3] Ejecutando QSVC en hardware real...")
print("    Train:", len(y_tr_hw), "| Test:", len(y_te_hw))
print("    Evaluaciones kernel estimadas:", len(y_tr_hw)**2 + len(y_tr_hw)*len(y_te_hw))

sampler  = Sampler(mode=bk)
fidelity = ComputeUncompute(sampler=sampler, pass_manager=pm)
kernel   = FidelityQuantumKernel(feature_map=fm, fidelity=fidelity)
qsvc     = QSVC(quantum_kernel=kernel)

t0      = time.perf_counter()
qsvc.fit(X_tr_hw_q, y_tr_hw)
t_train = time.perf_counter() - t0
print("    Entrenamiento:", round(t_train,2), "s")

t0    = time.perf_counter()
ypred = qsvc.predict(X_te_hw_q)
t_inf = time.perf_counter() - t0

yprob = expit(qsvc.decision_function(X_te_hw_q))
tn,fp,fn,tp_v = confusion_matrix(y_te_hw, ypred).ravel()

results = {
    'Accuracy'         : accuracy_score(y_te_hw, ypred),
    'Precision'        : precision_score(y_te_hw, ypred, zero_division=0),
    'Recall'           : recall_score(y_te_hw, ypred, zero_division=0),
    'F1-Score'         : f1_score(y_te_hw, ypred, zero_division=0),
    'ROC-AUC'          : roc_auc_score(y_te_hw, yprob) if len(np.unique(y_te_hw))>1 else 0,
    'Balanced Acc'     : balanced_accuracy_score(y_te_hw, ypred),
    'MCC'              : matthews_corrcoef(y_te_hw, ypred),
    'Specificity'      : tn/(tn+fp) if (tn+fp)>0 else 0,
    'Sensitivity'      : tp_v/(tp_v+fn) if (tp_v+fn)>0 else 0,
    'T Entrenamiento s': t_train,
    'T Inferencia s'   : t_inf,
    'T Total s'        : t_train+t_inf,
    'Backend'          : bk.name,
    'Train samples'    : len(y_tr_hw),
    'Test samples'     : len(y_te_hw),
    'Qubits'           : N_FEATURES,
    'Depth orig'       : dummy.decompose().depth(),
    'Depth transpiled' : tsp.depth(),
}

print("\n[4] RESULTADOS QSVC HARDWARE REAL:")
for k,v in results.items():
    print(f"  {k:22s}: {round(v,6) if isinstance(v,float) else v}")

print("\n" + classification_report(y_te_hw, ypred, target_names=['Normal','Fraude']))
print("Matriz de confusion:\n", confusion_matrix(y_te_hw, ypred))

pd.DataFrame([results]).to_csv(f"{OUTPUT_DIR}/resultados_hw_real.csv", index=False)
print("\nResultados guardados en", OUTPUT_DIR + "/resultados_hw_real.csv")
