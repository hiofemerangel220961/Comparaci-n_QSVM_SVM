import warnings
warnings.filterwarnings('ignore')
import json, sys, time, os, gzip, struct

# Garantiza que las rutas relativas (creditcard.csv, fashion-mnist-master/) funcionen
# sin importar desde que directorio se invoque el script.
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
    f1_score, balanced_accuracy_score, confusion_matrix, classification_report)
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector

# ── CREDENCIALES ──────────────────────────────────────────────────────────────
IBM_QUANTUM_API_KEY  = ''
IBM_QUANTUM_INSTANCE = os.environ.get('IBM_QUANTUM_INSTANCE', '')

_json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'apikey.json')
if os.path.isfile(_json_path):
    _data = json.load(open(_json_path, encoding='utf-8'))
    IBM_QUANTUM_API_KEY  = _data.get('apikey', '')
    IBM_QUANTUM_INSTANCE = _data.get('instance', IBM_QUANTUM_INSTANCE)

if not IBM_QUANTUM_API_KEY:
    IBM_QUANTUM_API_KEY = os.environ.get('IBM_QUANTUM_API_KEY', '')

if not IBM_QUANTUM_API_KEY:
    print("API key no encontrada. Crea apikey.json con {'apikey': 'tu_key'}")
    sys.exit(1)

print("API key OK: " + IBM_QUANTUM_API_KEY[:6] + "****")

# ── CONFIG (Suzuki et al. 2023, Tabla 3) ─────────────────────────────────────
# Parametros originales del paper para replicacion fiel.
N_QUBITS     = 4     # dimension reducida = numero de qubits
LAM          = 1.0   # parametro de escala del feature map
N_TRAIN      = 20    # muestras de entrenamiento (igual al paper)
N_TEST       = 10    # muestras de prueba (igual al paper)
N_SHOTS      = 500   # shots por entrada del kernel (igual al paper)
RANDOM_STATE = 42
OUTPUT_DIR   = 'resultados_hw_ibm'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Hiperparametros C del paper para hardware real (IonQ Harmony, Tabla 1)
# Usados como punto de partida en IBM Quantum
C_PAPER_HW = {
    'Credit Card':   4.2,
    'MNIST':         1.0,
    'Fashion-MNIST': 1.0,
}

# ── FEATURE MAP DEL PAPER (Ec. 1, Suzuki et al. 2023) ────────────────────────
def make_paper_feature_map(n_qubits, lam=1.0):
    """
    |phi(x)> = Rz(lam*xq) @ U_ent @ Ry(lam*xq) @ Rz(lam*xq) @ H |0^n>
    Circuito superficial con n-1 CNOTs en cadena (nearest-neighbor).
    """
    x  = ParameterVector('x', n_qubits)
    qc = QuantumCircuit(n_qubits)
    for q in range(n_qubits):
        qc.h(q)
        qc.rz(lam * x[q], q)
        qc.ry(lam * x[q], q)
    for q in range(n_qubits - 1):
        qc.cx(q, q + 1)
    for q in range(n_qubits):
        qc.rz(lam * x[q], q)
    return qc

# ── KERNEL NOISELESS (simulacion, para calcular alignment) ────────────────────
def compute_noiseless_kernel(X_a, X_b, feature_map):
    def statevec(x):
        return Statevector(feature_map.assign_parameters(x)).data
    svs_a = [statevec(x) for x in X_a]
    svs_b = [statevec(x) for x in X_b]
    K = np.zeros((len(svs_a), len(svs_b)))
    for i, sva in enumerate(svs_a):
        for j, svb in enumerate(svs_b):
            K[i, j] = abs(np.dot(sva.conj(), svb)) ** 2
    return K

def kernel_alignment(K1, K2):
    """Alineamiento de Frobenius A(K1,K2) — Ec. 7 del paper."""
    frob  = lambda A, B: np.sum(A * B)
    denom = np.sqrt(frob(K1, K1) * frob(K2, K2))
    return frob(K1, K2) / denom if denom > 0 else 0.0

# ── PREPARACION DE DATOS (igual al notebook) ──────────────────────────────────
def prepare_data(X, y, class_a, class_b):
    mask = (y == class_a) | (y == class_b)
    X_f, y_f = X[mask].copy(), y[mask].copy()
    y_f = (y_f == class_b).astype(int)

    n_per = min((y_f == 0).sum(), (y_f == 1).sum())
    idx0  = np.where(y_f == 0)[0][:n_per]
    idx1  = np.where(y_f == 1)[0][:n_per]
    idx   = np.concatenate([idx0, idx1])
    np.random.RandomState(RANDOM_STATE).shuffle(idx)
    X_b, y_b = X_f[idx], y_f[idx]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_b, y_b, test_size=N_TEST, train_size=N_TRAIN,
        stratify=y_b, random_state=RANDOM_STATE)

    pca    = PCA(n_components=N_QUBITS, random_state=RANDOM_STATE)
    X_tr_p = pca.fit_transform(X_tr)
    X_te_p = pca.transform(X_te)

    scaler  = MinMaxScaler(feature_range=(-np.pi * LAM, np.pi * LAM))
    X_tr_sc = scaler.fit_transform(X_tr_p)
    X_te_sc = scaler.transform(X_te_p)
    return X_tr_sc, X_te_sc, y_tr, y_te

# ── CARGA DE DATASETS ─────────────────────────────────────────────────────────
def load_credit_card():
    df = pd.read_csv('creditcard.csv').drop_duplicates()
    X  = df.drop(columns=['Class', 'Time']).values.astype(np.float64)
    y  = df['Class'].values
    return X, y

def load_mnist():
    from sklearn.datasets import fetch_openml
    print("  Descargando MNIST (puede tardar la primera vez)...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    return mnist.data.astype(np.float64) / 255.0, mnist.target.astype(int)

def load_fashion_mnist():
    base = 'fashion-mnist-master/data/fashion'
    def read_gz(path, is_images):
        with gzip.open(path, 'rb') as f:
            if is_images:
                _, n, r, c = struct.unpack('>IIII', f.read(16))
                return np.frombuffer(f.read(), dtype=np.uint8).reshape(n, r * c).astype(np.float64) / 255.0
            else:
                _, n = struct.unpack('>II', f.read(8))
                return np.frombuffer(f.read(), dtype=np.uint8)
    X = np.vstack([read_gz(f'{base}/train-images-idx3-ubyte.gz', True),
                   read_gz(f'{base}/t10k-images-idx3-ubyte.gz',  True)])
    y = np.concatenate([read_gz(f'{base}/train-labels-idx1-ubyte.gz', False),
                        read_gz(f'{base}/t10k-labels-idx1-ubyte.gz',  False)])
    return X, y

# ── CONEXION IBM QUANTUM ──────────────────────────────────────────────────────
print("\n[1] Conectando a IBM Quantum...")
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.state_fidelities import ComputeUncompute
from qiskit_machine_learning.algorithms import QSVC

_save_kw = dict(channel="ibm_quantum_platform", token=IBM_QUANTUM_API_KEY, overwrite=True)
_conn_kw = dict(channel="ibm_quantum_platform", token=IBM_QUANTUM_API_KEY)
if IBM_QUANTUM_INSTANCE:
    _save_kw['instance'] = IBM_QUANTUM_INSTANCE
    _conn_kw['instance'] = IBM_QUANTUM_INSTANCE

QiskitRuntimeService.save_account(**_save_kw)
service = QiskitRuntimeService(**_conn_kw)

def _nq(b):
    v = getattr(b, "num_qubits", 0)
    return len(v) if isinstance(v, (list, tuple)) else int(v)
def _pj(b):
    try: return int(getattr(b.status(), "pending_jobs", 0))
    except: return 0

all_backends = service.backends()
avail = [b for b in all_backends
         if not getattr(b, "simulator", True) and _nq(b) >= N_QUBITS]

if not avail:
    print(f"[ERROR] No hay backends reales con >= {N_QUBITS} qubits disponibles.")
    print("Backends encontrados:", [b.name for b in all_backends])
    sys.exit(1)

bk = min(avail, key=_pj)
print(f"Backend: {bk.name} | Qubits: {_nq(bk)} | Cola: {_pj(bk)} jobs")

fm  = make_paper_feature_map(N_QUBITS, lam=LAM)
pm  = generate_preset_pass_manager(backend=bk, optimization_level=1)
tsp = transpile(fm.assign_parameters({p: 0.5 for p in fm.parameters}),
                backend=bk, optimization_level=1)
print(f"Circuito del paper: depth={fm.decompose().depth()} | transpilado: {tsp.depth()}")
n_eval = N_TRAIN*(N_TRAIN+1)//2 + N_TRAIN*N_TEST
print(f"Shots por kernel entry: {N_SHOTS}  (paper=500)")
print(f"Evaluaciones por dataset: {n_eval} entradas x {N_SHOTS} shots = {n_eval*N_SHOTS:,} shots")
print(f"Total 3 datasets: {n_eval*N_SHOTS*3:,} shots  [N_TRAIN={N_TRAIN} paper=20, N_TEST={N_TEST} paper=10]")

# ── EXPERIMENTO POR DATASET ───────────────────────────────────────────────────
def run_experiment(name, X, y, class_a, class_b, kernel):
    """kernel es el FidelityQuantumKernel compartido de la Session."""
    print(f"\n{'='*60}")
    print(f"EXPERIMENTO: {name}  (clases {class_a} vs {class_b})")
    print(f"{'='*60}")

    X_tr, X_te, y_tr, y_te = prepare_data(X, y, class_a, class_b)
    print(f"Train: {X_tr.shape} | Test: {X_te.shape}")

    # Kernel noiseless (simulacion exacta, para alignment — local, sin QPU)
    print("  [sim] Calculando kernel noiseless...")
    K_nl_tr = compute_noiseless_kernel(X_tr, X_tr, fm)

    # Evaluar matrices de kernel en QPU UNA SOLA VEZ
    # (evita la doble llamada que ocurre si se usa QSVC.fit + kernel.evaluate por separado)
    print(f"  [hw]  Kernel train ({N_TRAIN}x{N_TRAIN}) en {bk.name}...")
    t0      = time.perf_counter()
    K_hw_tr = kernel.evaluate(X_tr)           # shape (N_TRAIN, N_TRAIN)
    t_k_tr  = time.perf_counter() - t0

    print(f"  [hw]  Kernel test  ({N_TEST}x{N_TRAIN}) en {bk.name}...")
    t0      = time.perf_counter()
    K_hw_te = kernel.evaluate(X_te, X_tr)     # shape (N_TEST, N_TRAIN)
    t_k_te  = time.perf_counter() - t0

    # Fit con kernel precomputado (SVC clasico, sin re-evaluar en QPU)
    t0    = time.perf_counter()
    svc_qk = SVC(kernel='precomputed', C=C_PAPER_HW[name])
    svc_qk.fit(K_hw_tr, y_tr)
    t_train = t_k_tr + (time.perf_counter() - t0)
    print(f"  Entrenamiento total (kernel+fit): {t_train:.1f}s")

    t0     = time.perf_counter()
    y_pred = svc_qk.predict(K_hw_te)
    t_inf  = t_k_te + (time.perf_counter() - t0)

    # Alignment: kernel HW vs noiseless (sin QPU adicional)
    alignment = kernel_alignment(K_hw_tr, K_nl_tr)

    results = {
        'Dataset':          name,
        'Backend':          bk.name,
        'N_Qubits':         N_QUBITS,
        'Lambda':           LAM,
        'N_Train':          N_TRAIN,
        'N_Test':           N_TEST,
        'Shots':            N_SHOTS,
        'C':                C_PAPER_HW[name],
        'Accuracy_test':    accuracy_score(y_te, y_pred),
        'Accuracy_train':   accuracy_score(y_tr, svc_qk.predict(K_hw_tr)),
        'Precision':        precision_score(y_te, y_pred, zero_division=0),
        'Recall':           recall_score(y_te, y_pred, zero_division=0),
        'F1':               f1_score(y_te, y_pred, zero_division=0),
        'Balanced_Acc':     balanced_accuracy_score(y_te, y_pred),
        'Alignment_hw_nl':  alignment,
        'T_Train_s':        t_train,
        'T_Inf_s':          t_inf,
        'Depth_orig':       fm.decompose().depth(),
        'Depth_transpiled': tsp.depth(),
    }

    paper_acc   = {'Credit Card': '70%', 'MNIST': '100%', 'Fashion-MNIST': '100%'}
    paper_align = {'Credit Card': 0.986, 'MNIST': 0.984, 'Fashion-MNIST': 0.993}
    print(f"\n  Accuracy  (test):  {results['Accuracy_test']:.2%}  "
          f"[paper IonQ N=20: {paper_acc[name]}]")
    print(f"  Accuracy  (train): {results['Accuracy_train']:.2%}")
    print(f"  Alignment hw/sim:  {alignment:.4f}  "
          f"[paper IonQ N=20: {paper_align[name]:.3f}]")
    print(f"  (Nota: paper usa N_TRAIN=20/N_TEST=10/N_SHOTS=500; este run usa {N_TRAIN}/{N_TEST}/{N_SHOTS})")
    print(f"  Tiempo total:      {t_train + t_inf:.1f}s")
    print()
    print(classification_report(y_te, y_pred, target_names=['Clase 0', 'Clase 1']))
    print("Matriz de confusion:\n", confusion_matrix(y_te, y_pred))

    out = f"{OUTPUT_DIR}/{name.lower().replace(' ', '_')}_ibm.csv"
    pd.DataFrame([results]).to_csv(out, index=False)
    print(f"  Guardado: {out}")
    return results

# ── CARGAR DATOS ANTES DE ABRIR LA SESSION ────────────────────────────────────
# Se cargan primero para no desperdiciar tiempo de Session en I/O
datasets = [
    ('Credit Card',   0, 1),
    ('MNIST',         0, 1),
    ('Fashion-MNIST', 0, 1),
]
loaders = {
    'Credit Card':   load_credit_card,
    'MNIST':         load_mnist,
    'Fashion-MNIST': load_fashion_mnist,
}

loaded = []
for name, ca, cb in datasets:
    try:
        print(f"Cargando {name}...")
        X, y = loaders[name]()
        loaded.append((name, X, y, ca, cb))
    except Exception as e:
        print(f"[WARNING] {name} no disponible: {e}")

# ── CORRER LOS 3 EXPERIMENTOS (plan Open: modo job directo) ───────────────────
# El plan Open de IBM Quantum no soporta Sessions (requiere plan Premium).
# Se usa Sampler en modo job: cada kernel.evaluate() envía un job al backend.
# Para N_TRAIN=10, N_TEST=5 son 6 jobs en total (2 por dataset × 3 datasets).
all_results = []

print(f"\n[3] Ejecutando en {bk.name} (plan Open — modo job)...")
print(f"    Cada evaluacion de kernel = 1 job. Total estimado: 6 jobs.")

sampler  = Sampler(mode=bk)
sampler.options.default_shots = N_SHOTS
fidelity = ComputeUncompute(sampler=sampler, pass_manager=pm)
kernel   = FidelityQuantumKernel(feature_map=fm, fidelity=fidelity)

for name, X, y, ca, cb in loaded:
    try:
        all_results.append(run_experiment(name, X, y, ca, cb, kernel))
    except Exception as e:
        print(f"[ERROR] {name}: {e}")

# ── RESUMEN COMPARATIVO ───────────────────────────────────────────────────────
if all_results:
    df_all = pd.DataFrame(all_results)
    df_all.to_csv(f"{OUTPUT_DIR}/resumen_ibm.csv", index=False)

    paper_ionq_acc   = {'Credit Card': 0.70, 'MNIST': 1.00, 'Fashion-MNIST': 1.00}
    paper_ionq_align = {'Credit Card': 0.986, 'MNIST': 0.984, 'Fashion-MNIST': 0.993}

    print(f"\n{'='*70}")
    print("RESUMEN — REPLICACION SUZUKI ET AL. 2023 EN IBM QUANTUM")
    print(f"{'='*70}")
    print(f"{'Dataset':<15} {'IBM Acc':>8} {'IonQ Acc':>9} {'IBM Align':>10} {'IonQ Align':>11}")
    print("-" * 58)
    for r in all_results:
        n = r['Dataset']
        print(f"{n:<15} {r['Accuracy_test']:>7.0%} {paper_ionq_acc[n]:>8.0%} "
              f"{r['Alignment_hw_nl']:>10.4f} {paper_ionq_align[n]:>11.3f}")

    print(f"\nResultados guardados en {OUTPUT_DIR}/")
    print("\nNota: C fijo segun valores IonQ del paper. Si el alignment IBM < 0.98,")
    print("considera ajustar C para compensar las diferencias de ruido entre plataformas.")
