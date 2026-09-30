# BirdCLEF+ 2026 experiment export
#
# Clean text export of the original Jupyter/Colab experiment.
# Cell boundaries are preserved with VS Code/Jupytext-style markers.
# Notebook outputs are intentionally omitted; verified metrics are documented in docs/RESULTS.md.

# %% [markdown]
#
# # BirdCLEF+ 2026 - V36 Local Overnight Validation Experiments
#
# Ce notebook est fait pour execution **locale**, pas pour soumission Kaggle. Il sert a tester plusieurs pistes pendant plusieurs heures avec une validation stricte.
#
# Objectif:
# - partir de la meilleure base publique actuelle: V35, score public `0.783`;
# - chercher une piste realiste vers `0.80+`;
# - eviter le leakage et reduire l'overfit de selection.
#
# Protocole anti-leakage:
# - validation uniquement sur `train_soundscapes`;
# - split groupe par `filename` de soundscape;
# - aucun segment du meme fichier soundscape n'est a la fois en train et validation;
# - `train_audio` reste uniquement cote train;
# - selection des configs sur 4 folds de developpement;
# - le dernier fold est garde comme holdout final pour verifier si le gain tient.
#
# Sorties locales:
#
# `G:/Mon Drive/projet ML/birdclef_2026_classical_ml_work/validation/v36_local_overnight_validation`

# %%

from pathlib import Path
import json
import time
import warnings
import gc
from itertools import product

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 120)
pd.set_option('display.width', 240)
try:
    from IPython.display import display
except Exception:
    def display(x):
        if hasattr(x, 'to_string'):
            print(x.to_string())
        else:
            print(x)

LOCAL_PROJECT_DIR = Path(r'G:/Mon Drive/projet ML')
IS_KAGGLE = Path('/kaggle/input').exists()
if IS_KAGGLE:
    print('WARNING: this notebook is configured for local execution. Prefer running it on your local machine.')

ROOT = LOCAL_PROJECT_DIR if LOCAL_PROJECT_DIR.exists() else Path('/kaggle/working')
OUT_DIR = ROOT / 'birdclef_2026_classical_ml_work' / 'validation' / 'v36_local_overnight_validation'
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = OUT_DIR / 'cache_predictions'
CACHE_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR_CANDIDATES = [
    Path('/kaggle/input/competitions/birdclef-2026'),
    Path('/kaggle/input/birdclef-2026'),
    Path(r'G:/Mon Drive/projet ML/birdclef-2026'),
]
DATA_DIR = next((p for p in DATA_DIR_CANDIDATES if (p / 'sample_submission.csv').exists()), DATA_DIR_CANDIDATES[0])
SAMPLE_SUB_PATH = DATA_DIR / 'sample_submission.csv'

RAW_FEATURE_CANDIDATES = [
    Path.cwd().parent / 'feature_caches' / 'train_features_science_full.parquet',
    Path(r'G:/Mon Drive/projet_ML/feature_caches/train_features_science_full.parquet'),
    Path('/kaggle/input/datasets/zineddinemohammedi/birdclef-2026-classical-features/train_features_science_full.parquet'),
    Path('/kaggle/input/birdclef-2026-classical-features/train_features_science_full.parquet'),
    Path('/kaggle/input/birdclef-v7-raw-features/train_features_science_full.parquet'),
    Path(r'G:/Mon Drive/projet ML/birdclef_2026_classical_ml_work/features/train_features_science_full.parquet'),
]

LOGMEL_FEATURE_CANDIDATES = [
    Path.cwd().parent / 'feature_caches' / 'v21_logmel_pcen64_features.parquet',
    Path(r'G:/Mon Drive/projet_ML/feature_caches/v21_logmel_pcen64_features.parquet'),
    Path('/kaggle/input/datasets/zineddinemohammedi/v21-logmel-pcen64-features/v21_logmel_pcen64_features.parquet'),
    Path('/kaggle/input/birdclef-v21-logmel-pcen64-features/v21_logmel_pcen64_features.parquet'),
    Path('/kaggle/input/v21-logmel-pcen64-features/v21_logmel_pcen64_features.parquet'),
    Path(r'G:/Mon Drive/projet ML/birdclef_2026_classical_ml_work/validation/v21_logmel_pcen64_features.parquet'),
]

SUBWIN_FEATURE_CANDIDATES = [
    Path.cwd().parent / 'feature_caches' / 'v31_soundscape_subwindow_features.parquet',
    Path(r'G:/Mon Drive/projet_ML/feature_caches/v31_soundscape_subwindow_features.parquet'),
    Path('/kaggle/input/datasets/zineddinemohammedi/v31-soundscape-subwindow-features/v31_soundscape_subwindow_features.parquet'),
    Path('/kaggle/input/birdclef-v31-soundscape-subwindow-features/v31_soundscape_subwindow_features.parquet'),
    Path('/kaggle/input/v31-soundscape-subwindow-features/v31_soundscape_subwindow_features.parquet'),
    Path(r'G:/Mon Drive/projet ML/birdclef_2026_classical_ml_work/validation/v31_soundscape_subwindow_features.parquet'),
]

MIN_POS = 3
SEED = 42
N_SPLITS = 5
DEV_FOLDS = ['fold0', 'fold1', 'fold2', 'fold3']
HOLDOUT_FOLD = 'fold4'

N_ROUNDS_RAW = 80
N_ROUNDS_LOGMEL = 50
SOURCE_WEIGHT_SS = 1.0

W_RAW = 0.95
W_LOGMEL = 0.03
W_PROTO = 0.02

FLOOR_EPS = 1e-8

LGB_BASE_PARAMS = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'learning_rate': 0.04,
    'num_leaves': 127,
    'feature_fraction': 0.85,
    'bagging_fraction': 0.80,
    'bagging_freq': 5,
    'min_data_in_leaf': 4,
    'lambda_l1': 0.1,
    'lambda_l2': 0.3,
    'verbose': -1,
    'n_jobs': -1,
    'force_col_wise': True,
    'seed': SEED,
    'bagging_seed': SEED + 11,
    'feature_fraction_seed': SEED + 17,
    'data_random_seed': SEED + 23,
}

SUBWIN_MODELS = {
    'sub_l31_r120': {'num_leaves': 31, 'rounds': 120, 'min_data_in_leaf': 3},
    'sub_l63_r80': {'num_leaves': 63, 'rounds': 80, 'min_data_in_leaf': 3},
    'sub_l127_r80': {'num_leaves': 127, 'rounds': 80, 'min_data_in_leaf': 2},
}

print('DATA_DIR:', DATA_DIR)
print('OUT_DIR:', OUT_DIR)
print('CACHE_DIR:', CACHE_DIR)
print('LOCAL MODE:', ROOT == LOCAL_PROJECT_DIR)

# %%

def resolve_parquet(candidates, expected_name):
    for p in candidates:
        if p.exists():
            return p
    input_root = Path('/kaggle/input')
    if input_root.exists():
        matches = list(input_root.rglob(expected_name))
        if matches:
            print('Auto-found:', matches[0])
            return matches[0]
        available = list(input_root.rglob('*.parquet'))[:100]
        raise FileNotFoundError(
            f'{expected_name} not found. Available parquet files:\n' +
            '\n'.join(str(x) for x in available)
        )
    raise FileNotFoundError(expected_name)


def parse_labels(s, label_set):
    try:
        arr = json.loads(s)
        return [str(z) for z in arr if str(z) in label_set]
    except Exception:
        return []


def basename_series(s):
    return s.astype(str).map(lambda x: Path(x).name)


sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
label_cols = sample_sub.columns[1:].tolist()
label_set = set(label_cols)
n_classes = len(label_cols)
floor = 1.0 / n_classes

raw_file = resolve_parquet(RAW_FEATURE_CANDIDATES, 'train_features_science_full.parquet')
logmel_file = resolve_parquet(LOGMEL_FEATURE_CANDIDATES, 'v21_logmel_pcen64_features.parquet')
subwin_file = resolve_parquet(SUBWIN_FEATURE_CANDIDATES, 'v31_soundscape_subwindow_features.parquet')

print('raw_file:', raw_file)
print('logmel_file:', logmel_file)
print('subwin_file:', subwin_file)

raw_df = pd.read_parquet(raw_file)
logmel_df = pd.read_parquet(logmel_file)
subwin_df = pd.read_parquet(subwin_file)

for df in [raw_df, logmel_df]:
    for c in ['row_index', 'split']:
        if c in df.columns:
            df.drop(columns=[c], inplace=True)

META_COLS = ['source', 'filename', 'offset_sec', 'labels', 'primary_label']
for name, df in [('raw', raw_df), ('logmel', logmel_df)]:
    missing = [c for c in META_COLS if c not in df.columns]
    if missing:
        raise RuntimeError(f'Missing columns in {name}: {missing}')
    df.sort_values(META_COLS, inplace=True)
    df.reset_index(drop=True, inplace=True)

SUBWIN_META_COLS = ['filename', 'start', 'end', 'offset_sec', 'primary_label', 'labels_json', 'stable_key']
missing_sub = [c for c in SUBWIN_META_COLS if c not in subwin_df.columns]
if missing_sub:
    raise RuntimeError(f'Missing subwindow columns: {missing_sub}')
subwin_df = subwin_df.sort_values(['filename', 'offset_sec']).reset_index(drop=True)

Y_raw = MultiLabelBinarizer(classes=label_cols).fit_transform(
    [parse_labels(s, label_set) for s in raw_df['labels']]
).astype(np.uint8)
Y_logmel = MultiLabelBinarizer(classes=label_cols).fit_transform(
    [parse_labels(s, label_set) for s in logmel_df['labels']]
).astype(np.uint8)
Y_subwin = MultiLabelBinarizer(classes=label_cols).fit_transform(
    [parse_labels(s, label_set) for s in subwin_df['labels_json']]
).astype(np.uint8)

raw_feature_cols = [c for c in raw_df.columns if c not in META_COLS]
logmel_feature_cols = [c for c in logmel_df.columns if c not in META_COLS]
subwin_feature_cols = [c for c in subwin_df.columns if c not in SUBWIN_META_COLS]

raw_imp = SimpleImputer(strategy='median')
raw_vt = VarianceThreshold(threshold=1e-8)
X_raw = raw_vt.fit_transform(raw_imp.fit_transform(
    raw_df[raw_feature_cols].replace([np.inf, -np.inf], np.nan).astype(np.float32)
)).astype(np.float32)

logmel_imp = SimpleImputer(strategy='median')
logmel_vt = VarianceThreshold(threshold=1e-8)
X_logmel = logmel_vt.fit_transform(logmel_imp.fit_transform(
    logmel_df[logmel_feature_cols].replace([np.inf, -np.inf], np.nan).astype(np.float32)
)).astype(np.float32)

subwin_imp = SimpleImputer(strategy='median')
subwin_vt = VarianceThreshold(threshold=1e-8)
X_subwin = subwin_vt.fit_transform(subwin_imp.fit_transform(
    subwin_df[subwin_feature_cols].replace([np.inf, -np.inf], np.nan).astype(np.float32)
)).astype(np.float32)

raw_df['_base_filename'] = basename_series(raw_df['filename'])
logmel_df['_base_filename'] = basename_series(logmel_df['filename'])
subwin_df['_base_filename'] = subwin_df['filename'].astype(str)

print('raw:', raw_df.shape, X_raw.shape, Y_raw.shape)
print('logmel:', logmel_df.shape, X_logmel.shape, Y_logmel.shape)
print('subwin:', subwin_df.shape, X_subwin.shape, Y_subwin.shape)
print('trainable raw:', int((Y_raw.sum(axis=0) >= MIN_POS).sum()), '/', n_classes)
print('trainable logmel:', int((Y_logmel.sum(axis=0) >= MIN_POS).sum()), '/', n_classes)
print('trainable subwin:', int((Y_subwin.sum(axis=0) >= MIN_POS).sum()), '/', n_classes)

# %%

def scale_pos_weight(n_neg, n_pos):
    ratio = n_neg / max(1, n_pos)
    return float(min(80.0, np.sqrt(ratio)))


def macro_auc(y_true, pred):
    vals = []
    y_true = np.asarray(y_true)
    pred = np.asarray(pred)
    for j in range(y_true.shape[1]):
        yj = y_true[:, j]
        if yj.sum() == 0 or yj.sum() == len(yj):
            continue
        vals.append(roc_auc_score(yj, pred[:, j]))
    return float(np.mean(vals)) if vals else np.nan


def macro_ap(y_true, pred):
    vals = []
    y_true = np.asarray(y_true)
    pred = np.asarray(pred)
    for j in range(y_true.shape[1]):
        yj = y_true[:, j]
        if yj.sum() == 0:
            continue
        vals.append(average_precision_score(yj, pred[:, j]))
    return float(np.mean(vals)) if vals else np.nan


def entropy_flatten_array(pred, q=0.50, gamma=1.15):
    pred = np.clip(np.asarray(pred, dtype=np.float32), 0.0, 1.0)
    maxp = pred.max(axis=1)
    thr = float(np.quantile(maxp, q))
    out = pred.copy()
    mask = maxp <= thr
    if np.any(mask):
        out[mask] = np.power(out[mask], gamma)
    return np.clip(out, 0.0, 1.0)


def train_cooc_matrix(Y, alpha=0.5, diag_zero=True):
    Yf = np.asarray(Y, dtype=np.float32)
    counts = Yf.sum(axis=0)
    C = Yf.T @ Yf
    C = (C + alpha) / (counts[:, None] + alpha * Yf.shape[1] + 1e-6)
    if diag_zero:
        np.fill_diagonal(C, 0.0)
    return C.astype(np.float32)


def apply_cooccurrence_raw_sum(pred, C, blend_w=0.12):
    if blend_w <= 0:
        return np.clip(pred, 0.0, 1.0)
    support = pred @ C
    support = support / np.maximum(C.sum(axis=0, keepdims=True), 1e-6)
    out = (1.0 - blend_w) * pred + blend_w * support
    return np.clip(out, 0.0, 1.0)


def smooth_by_meta(pred, meta, window=17, alpha=0.95):
    if window <= 1 or alpha <= 0:
        return np.clip(pred, 0.0, 1.0).astype(np.float32)
    df = meta[['filename', 'offset_sec']].copy().reset_index(drop=True)
    df['_orig'] = np.arange(len(df))
    out = np.zeros_like(pred, dtype=np.float32)
    half = window // 2
    for _, idxs in df.sort_values(['filename', 'offset_sec']).groupby('filename').groups.items():
        idxs = np.asarray(list(idxs), dtype=int)
        vals = pred[idxs]
        sm = vals.copy()
        for i in range(len(idxs)):
            lo = max(0, i - half)
            hi = min(len(idxs), i + half + 1)
            sm[i] = (1.0 - alpha) * vals[i] + alpha * vals[lo:hi].mean(axis=0)
        out[idxs] = sm
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def rank01(pred):
    pred = np.asarray(pred, dtype=np.float32)
    out = np.zeros_like(pred, dtype=np.float32)
    n = pred.shape[0]
    if n <= 1:
        return pred.copy()
    for j in range(pred.shape[1]):
        out[:, j] = (pd.Series(pred[:, j]).rank(method='average').values.astype(np.float32) - 1.0) / (n - 1.0)
    return out


def train_predict_ovr(X_train, Y_train, X_val, source_values=None, rounds=80, model_name='model', params_override=None):
    trainable = np.where(Y_train.sum(axis=0) >= MIN_POS)[0]
    pred = np.full((len(X_val), n_classes), floor, dtype=np.float32)
    sample_weight = None
    if source_values is not None:
        sample_weight = np.ones(len(source_values), dtype=np.float32)
        if SOURCE_WEIGHT_SS != 1.0:
            sample_weight[np.asarray(source_values).astype(str) == 'train_soundscapes'] = SOURCE_WEIGHT_SS

    params_base = dict(LGB_BASE_PARAMS)
    if params_override:
        params_base.update(params_override)

    t0 = time.perf_counter()
    print(f'Training {model_name}: {len(trainable)}/{n_classes} classes | rounds={rounds}')
    for k, j in enumerate(trainable, 1):
        yj = Y_train[:, j]
        n_pos = int(yj.sum())
        n_neg = len(yj) - n_pos
        params = dict(params_base)
        params['scale_pos_weight'] = scale_pos_weight(n_neg, n_pos)
        ds = lgb.Dataset(X_train, label=yj, weight=sample_weight)
        model = lgb.train(params, ds, num_boost_round=rounds)
        pred[:, j] = model.predict(X_val).astype(np.float32)
        del model, ds
        if k % 40 == 0 or k == len(trainable):
            gc.collect()
            print(f'  {model_name}: {k}/{len(trainable)}')
    print(f'{model_name} elapsed min:', (time.perf_counter() - t0) / 60)
    return pred


def fold_proto_predict(X_train, Y_train, X_val):
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train).astype(np.float32)
    Xva = scaler.transform(X_val).astype(np.float32)
    Xtr = Xtr / np.maximum(np.linalg.norm(Xtr, axis=1, keepdims=True), 1e-12)
    Xva = Xva / np.maximum(np.linalg.norm(Xva, axis=1, keepdims=True), 1e-12)
    proto = np.zeros((n_classes, Xtr.shape[1]), dtype=np.float32)
    counts = Y_train.sum(axis=0)
    for j in range(n_classes):
        idx = np.where(Y_train[:, j] > 0)[0]
        if len(idx):
            proto[j] = Xtr[idx].mean(axis=0)
    proto = proto / np.maximum(np.linalg.norm(proto, axis=1, keepdims=True), 1e-12)
    pred = ((Xva @ proto.T) + 1.0) / 2.0
    pred[:, counts == 0] = floor
    return np.clip(pred, 0.0, 1.0).astype(np.float32)

# %%

# Fold construction and leakage checks.
raw_ss_mask = raw_df['source'].astype(str).eq('train_soundscapes').values
logmel_ss_mask = logmel_df['source'].astype(str).eq('train_soundscapes').values
if not np.array_equal(raw_df[META_COLS].astype(str).values, logmel_df[META_COLS].astype(str).values):
    raise RuntimeError('raw/logmel metadata are not aligned after sorting')

soundscape_files = raw_df.loc[raw_ss_mask, '_base_filename'].astype(str).values
unique_files = np.array(sorted(pd.unique(soundscape_files)))
print('soundscape rows in raw:', raw_ss_mask.sum(), '| unique files:', len(unique_files))
print('subwindow files:', subwin_df['_base_filename'].nunique(), '| subwindow rows:', len(subwin_df))

gkf = GroupKFold(n_splits=N_SPLITS)
dummy_X = np.zeros(len(unique_files))
folds = []
for fold_idx, (_, file_val_idx) in enumerate(gkf.split(dummy_X, groups=unique_files)):
    fold_name = f'fold{fold_idx}'
    val_files = set(unique_files[file_val_idx])

    raw_val_mask = raw_ss_mask & raw_df['_base_filename'].astype(str).isin(val_files).values
    raw_train_mask = ~raw_val_mask

    logmel_val_mask = logmel_ss_mask & logmel_df['_base_filename'].astype(str).isin(val_files).values
    logmel_train_mask = ~logmel_val_mask

    sub_val_mask = subwin_df['_base_filename'].astype(str).isin(val_files).values
    sub_train_mask = ~sub_val_mask

    # Leakage checks.
    train_ss_files_raw = set(raw_df.loc[raw_train_mask & raw_ss_mask, '_base_filename'].astype(str))
    val_ss_files_raw = set(raw_df.loc[raw_val_mask, '_base_filename'].astype(str))
    if train_ss_files_raw & val_ss_files_raw:
        raise RuntimeError(f'Leakage in {fold_name}: same soundscape file in train and validation')
    if set(subwin_df.loc[sub_train_mask, '_base_filename'].astype(str)) & set(subwin_df.loc[sub_val_mask, '_base_filename'].astype(str)):
        raise RuntimeError(f'Subwindow leakage in {fold_name}')

    folds.append({
        'name': fold_name,
        'val_files': sorted(val_files),
        'raw_train_mask': raw_train_mask,
        'raw_val_mask': raw_val_mask,
        'logmel_train_mask': logmel_train_mask,
        'logmel_val_mask': logmel_val_mask,
        'sub_train_mask': sub_train_mask,
        'sub_val_mask': sub_val_mask,
    })
    print(fold_name, '| val files:', len(val_files), '| raw val rows:', int(raw_val_mask.sum()), '| sub val rows:', int(sub_val_mask.sum()))

fold_manifest = pd.DataFrame([
    {'fold': f['name'], 'n_val_files': len(f['val_files']), 'n_raw_val': int(f['raw_val_mask'].sum()), 'n_sub_val': int(f['sub_val_mask'].sum())}
    for f in folds
])
fold_manifest.to_csv(OUT_DIR / 'v36_fold_manifest.csv', index=False)
display(fold_manifest)

# %%

def raw_to_subwin_aligned(pred_raw_val, raw_val_meta, sub_val_meta):
    # raw soundscape features use segment start offset; subwindow uses segment end offset.
    base_map = {
        (Path(str(r.filename)).name, round(float(r.offset_sec) + 5.0, 6)): p
        for r, p in zip(raw_val_meta[['filename', 'offset_sec']].itertuples(index=False), pred_raw_val)
    }
    ordered_pairs = [
        (str(r.filename), round(float(r.offset_sec), 6))
        for r in sub_val_meta[['filename', 'offset_sec']].itertuples(index=False)
    ]
    missing = [p for p in ordered_pairs if p not in base_map]
    if missing:
        raise RuntimeError(f'Missing raw->subwindow alignment rows: {len(missing)} examples={missing[:5]}')
    return np.vstack([base_map[p] for p in ordered_pairs]).astype(np.float32)


fold_payloads = {}
t0_all = time.perf_counter()
fold_status_rows = []

for f in folds:
    name = f['name']
    cache_path = CACHE_DIR / f'{name}_predictions.npz'
    print('\n' + '=' * 80)
    print('FOLD', name, '| cache:', cache_path)

    raw_val_idx = np.where(f['raw_val_mask'])[0]
    sub_val_idx = np.where(f['sub_val_mask'])[0]
    raw_val_meta = raw_df.loc[raw_val_idx, ['filename', 'offset_sec']].reset_index(drop=True)
    sub_val_meta = subwin_df.loc[sub_val_idx, ['filename', 'offset_sec']].reset_index(drop=True)
    Y_val = Y_subwin[sub_val_idx]

    if cache_path.exists():
        z = np.load(cache_path)
        pred_raw = z['pred_raw'].astype(np.float32)
        pred_logmel = z['pred_logmel'].astype(np.float32)
        pred_proto = z['pred_proto'].astype(np.float32)
        sub_preds = {k.replace('pred_', ''): z[k].astype(np.float32) for k in z.files if k.startswith('pred_sub_')}
        print('Loaded cache:', cache_path.name, '| sub models:', sorted(sub_preds))
    else:
        pred_raw = train_predict_ovr(
            X_raw[f['raw_train_mask']], Y_raw[f['raw_train_mask']],
            X_raw[f['raw_val_mask']],
            source_values=raw_df.loc[f['raw_train_mask'], 'source'].values,
            rounds=N_ROUNDS_RAW,
            model_name=f'{name}_raw'
        )
        pred_logmel = train_predict_ovr(
            X_logmel[f['logmel_train_mask']], Y_logmel[f['logmel_train_mask']],
            X_logmel[f['logmel_val_mask']],
            source_values=logmel_df.loc[f['logmel_train_mask'], 'source'].values,
            rounds=N_ROUNDS_LOGMEL,
            model_name=f'{name}_logmel'
        )
        pred_proto = fold_proto_predict(
            X_raw[f['raw_train_mask']], Y_raw[f['raw_train_mask']],
            X_raw[f['raw_val_mask']]
        )

        sub_preds = {}
        for sub_name, cfg in SUBWIN_MODELS.items():
            params_override = {
                'num_leaves': cfg['num_leaves'],
                'min_data_in_leaf': cfg['min_data_in_leaf'],
            }
            sub_preds[sub_name] = train_predict_ovr(
                X_subwin[f['sub_train_mask']], Y_subwin[f['sub_train_mask']],
                X_subwin[f['sub_val_mask']],
                source_values=None,
                rounds=cfg['rounds'],
                model_name=f'{name}_{sub_name}',
                params_override=params_override
            )

        save_dict = {
            'pred_raw': pred_raw,
            'pred_logmel': pred_logmel,
            'pred_proto': pred_proto,
        }
        for sub_name, arr in sub_preds.items():
            save_dict[f'pred_{sub_name}'] = arr
        np.savez_compressed(cache_path, **save_dict)
        print('Saved cache:', cache_path)

    fold_status_rows.append({
        'fold': name,
        'cache_path': str(cache_path),
        'raw_val_rows': len(raw_val_idx),
        'sub_val_rows': len(sub_val_idx),
        'sub_models': ','.join(sorted(sub_preds.keys())),
        'elapsed_hours_total': (time.perf_counter() - t0_all) / 3600,
    })
    pd.DataFrame(fold_status_rows).to_csv(OUT_DIR / 'v36_fold_training_progress.csv', index=False)
    print('Progress saved:', OUT_DIR / 'v36_fold_training_progress.csv')

    fold_payloads[name] = {
        'raw_train_mask': f['raw_train_mask'],
        'sub_train_mask': f['sub_train_mask'],
        'raw_val_meta': raw_val_meta,
        'sub_val_meta': sub_val_meta,
        'Y_val': Y_val,
        'Y_raw_train': Y_raw[f['raw_train_mask']],
        'Y_sub_train': Y_subwin[f['sub_train_mask']],
        'pred_raw': pred_raw,
        'pred_logmel': pred_logmel,
        'pred_proto': pred_proto,
        'sub_preds': sub_preds,
    }

print('All fold prediction stage elapsed hours:', (time.perf_counter() - t0_all) / 3600)

# %%

def base_prediction_for_fold(payload, base_smooth=(17, 0.95), base_entropy=True, base_cooc_w=0.12):
    pred = W_RAW * payload['pred_raw'] + W_LOGMEL * payload['pred_logmel'] + W_PROTO * payload['pred_proto']
    pred = np.clip(pred, 0.0, 1.0)
    if base_entropy:
        pred = entropy_flatten_array(pred, q=0.50, gamma=1.15)
    if base_cooc_w > 0:
        C = train_cooc_matrix(payload['Y_raw_train'], alpha=0.5, diag_zero=True)
        pred = apply_cooccurrence_raw_sum(pred, C, blend_w=base_cooc_w)
    pred = smooth_by_meta(pred, payload['raw_val_meta'], window=base_smooth[0], alpha=base_smooth[1])
    return raw_to_subwin_aligned(pred, payload['raw_val_meta'], payload['sub_val_meta'])


def sub_prediction_for_fold(payload, sub_model_name, sub_smooth=(17, 0.95), sub_entropy=True, sub_cooc_w=0.12):
    pred = payload['sub_preds'][sub_model_name].copy()
    if sub_entropy:
        pred = entropy_flatten_array(pred, q=0.50, gamma=1.15)
    if sub_cooc_w > 0:
        C = train_cooc_matrix(payload['Y_sub_train'], alpha=0.5, diag_zero=True)
        pred = apply_cooccurrence_raw_sum(pred, C, blend_w=sub_cooc_w)
    pred = smooth_by_meta(pred, payload['sub_val_meta'], window=sub_smooth[0], alpha=sub_smooth[1])
    return pred


def class_weight_vector(mode, w, Y_sub_train):
    counts = Y_sub_train.sum(axis=0).astype(np.float32)
    if mode == 'global':
        return np.full(n_classes, w, dtype=np.float32)
    if mode == 'sqrt_count10':
        return (w * np.sqrt(counts / (counts + 10.0 + 1e-6))).astype(np.float32)
    if mode == 'sqrt_count25':
        return (w * np.sqrt(counts / (counts + 25.0 + 1e-6))).astype(np.float32)
    if mode == 'hard5':
        return (w * (counts >= 5)).astype(np.float32)
    if mode == 'hard10':
        return (w * (counts >= 10)).astype(np.float32)
    raise ValueError(mode)


def blend_predictions(base_pred, sub_pred, mode, w_vec):
    w_vec = np.asarray(w_vec, dtype=np.float32)[None, :]
    if mode == 'prob':
        return np.clip((1.0 - w_vec) * base_pred + w_vec * sub_pred, 0.0, 1.0)
    if mode == 'rank':
        return np.clip((1.0 - w_vec) * rank01(base_pred) + w_vec * rank01(sub_pred), 0.0, 1.0)
    raise ValueError(mode)


BASE_CONFIGS = [
    {'base_name': 'v34_w17', 'base_smooth': (17, 0.95), 'base_entropy': True, 'base_cooc_w': 0.12},
    {'base_name': 'v34_w15', 'base_smooth': (15, 0.95), 'base_entropy': True, 'base_cooc_w': 0.12},
    {'base_name': 'v34_w21', 'base_smooth': (21, 0.95), 'base_entropy': True, 'base_cooc_w': 0.12},
]
SUB_SMOOTHS = [(1, 0.0), (9, 0.95), (17, 0.95), (25, 0.95)]
SUB_COOC_W = [0.0, 0.06, 0.12, 0.20]
SUB_ENTROPY = [False, True]
BLEND_W = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
BLEND_MODES = ['prob', 'rank']
CLASS_WEIGHT_MODES = ['global', 'sqrt_count10', 'sqrt_count25', 'hard5', 'hard10']

rows = []
t0 = time.perf_counter()

# Precompute base and sub postprocessed predictions per fold/config to keep the grid fast.
base_cache = {}
sub_cache = {}
for fold_name, payload in fold_payloads.items():
    for bc in BASE_CONFIGS:
        key = (fold_name, bc['base_name'])
        base_cache[key] = base_prediction_for_fold(
            payload,
            base_smooth=bc['base_smooth'],
            base_entropy=bc['base_entropy'],
            base_cooc_w=bc['base_cooc_w'],
        )

    for sub_name in payload['sub_preds'].keys():
        for sub_smooth, sub_cooc_w, sub_entropy in product(SUB_SMOOTHS, SUB_COOC_W, SUB_ENTROPY):
            key = (fold_name, sub_name, sub_smooth[0], sub_smooth[1], sub_cooc_w, sub_entropy)
            sub_cache[key] = sub_prediction_for_fold(
                payload,
                sub_model_name=sub_name,
                sub_smooth=sub_smooth,
                sub_entropy=sub_entropy,
                sub_cooc_w=sub_cooc_w,
            )

print('postprocess cache built | elapsed min:', (time.perf_counter() - t0) / 60)

for bc in BASE_CONFIGS:
    for sub_name in SUBWIN_MODELS.keys():
        for sub_smooth, sub_cooc_w, sub_entropy, w, blend_mode, cw_mode in product(
            SUB_SMOOTHS, SUB_COOC_W, SUB_ENTROPY, BLEND_W, BLEND_MODES, CLASS_WEIGHT_MODES
        ):
            cfg_name = (
                f"{bc['base_name']}|{sub_name}|subw{sub_smooth[0]}a{sub_smooth[1]}|"
                f"subcooc{sub_cooc_w}|subent{sub_entropy}|w{w}|{blend_mode}|{cw_mode}"
            )
            for fold_name, payload in fold_payloads.items():
                base_pred = base_cache[(fold_name, bc['base_name'])]
                sub_pred = sub_cache[(fold_name, sub_name, sub_smooth[0], sub_smooth[1], sub_cooc_w, sub_entropy)]
                w_vec = class_weight_vector(cw_mode, w, payload['Y_sub_train'])
                pred = blend_predictions(base_pred, sub_pred, blend_mode, w_vec)
                rows.append({
                    'config': cfg_name,
                    'base_name': bc['base_name'],
                    'sub_model': sub_name,
                    'sub_smooth_window': sub_smooth[0],
                    'sub_smooth_alpha': sub_smooth[1],
                    'sub_cooc_w': sub_cooc_w,
                    'sub_entropy': sub_entropy,
                    'blend_w': w,
                    'blend_mode': blend_mode,
                    'class_weight_mode': cw_mode,
                    'fold': fold_name,
                    'auc': macro_auc(payload['Y_val'], pred),
                    'ap': macro_ap(payload['Y_val'], pred),
                })

results = pd.DataFrame(rows)
results.to_csv(OUT_DIR / 'v36_all_fold_config_results.csv', index=False)
print('grid rows:', len(results), '| elapsed min:', (time.perf_counter() - t0) / 60)
display(results.head())

# %%

dev = results[results['fold'].isin(DEV_FOLDS)].copy()
hold = results[results['fold'].eq(HOLDOUT_FOLD)].copy()

dev_summary = (
    dev.groupby(['config', 'base_name', 'sub_model', 'sub_smooth_window', 'sub_smooth_alpha',
                 'sub_cooc_w', 'sub_entropy', 'blend_w', 'blend_mode', 'class_weight_mode'], dropna=False)
    .agg(dev_mean_auc=('auc', 'mean'), dev_min_auc=('auc', 'min'), dev_std_auc=('auc', 'std'),
         dev_mean_ap=('ap', 'mean'), dev_min_ap=('ap', 'min'), dev_std_ap=('ap', 'std'))
    .reset_index()
)
dev_summary['dev_safe_auc'] = dev_summary['dev_mean_auc'] - 0.50 * dev_summary['dev_std_auc'].fillna(0)

hold_summary = hold[['config', 'auc', 'ap']].rename(columns={'auc': 'holdout_auc', 'ap': 'holdout_ap'})
summary = dev_summary.merge(hold_summary, on='config', how='left')
summary['holdout_delta_vs_0783_proxy'] = summary['holdout_auc'] - 0.783

summary = summary.sort_values(['dev_safe_auc', 'dev_mean_auc', 'dev_min_auc', 'dev_mean_ap'], ascending=False)
summary.to_csv(OUT_DIR / 'v36_dev_selected_holdout_summary.csv', index=False)

print('Top 40 selected on DEV folds only, with untouched HOLDOUT shown after selection')
display(summary.head(40))

print('\nBest by holdout, diagnostic only. Do not select only from this table.')
display(summary.sort_values(['holdout_auc', 'dev_safe_auc'], ascending=False).head(40))

best = summary.iloc[0].to_dict()
with open(OUT_DIR / 'v36_best_dev_config.json', 'w', encoding='utf-8') as f:
    json.dump(best, f, indent=2, ensure_ascii=False)

print('\nSaved files:')
for p in sorted(OUT_DIR.glob('*')):
    print(p)

# %% [markdown]
#
# ## Comment lire les resultats
#
# Regle de decision:
#
# - Ne pas prendre une config juste parce qu'elle est premiere sur le holdout.
# - Choisir d'abord via `dev_safe_auc`, calcule sur les 4 folds de developpement.
# - Verifier ensuite que le `holdout_auc` ne s'effondre pas.
# - Une bonne piste pour Kaggle doit ameliorer le holdout et rester stable sur les folds.
#
# Si une config depasse `0.89` ou `0.90` localement mais avec un holdout faible, c'est probablement de l'overfit de selection.
#
# Ce notebook ne produit pas de `submission.csv` volontairement. Il sert a decider quelle prochaine version Kaggle construire.