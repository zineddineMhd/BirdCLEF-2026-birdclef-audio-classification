# BirdCLEF+ 2026 experiment export
#
# Clean text export of the original Jupyter/Colab experiment.
# Cell boundaries are preserved with VS Code/Jupytext-style markers.
# Notebook outputs are intentionally omitted; verified metrics are documented in docs/RESULTS.md.

# %% [markdown]
# # BirdCLEF 2026 - V31 Local Sub-Window Soundscape Features Validation
#
# Objectif: tester une piste qui attaque un vrai plafond du modèle classique: les événements courts noyés dans une fenêtre de 5 secondes.
#
# Idée:
# - extraire des features uniquement sur `train_soundscapes` annotés;
# - calculer des statistiques temporelles sur sous-fenêtres de 0.5s / 1s;
# - entraîner un modèle LightGBM soundscape-only;
# - le blender avec V30 global en validation multi-split groupée.
#
# Ce notebook ne crée pas de soumission Kaggle. Si le signal est fort et stable, on décidera ensuite comment convertir cette idée en version Kaggle CPU-safe.

# %%

from pathlib import Path
import json
import time
import warnings

import librosa
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import MultiLabelBinarizer

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 80)
pd.set_option('display.width', 220)

ROOT = Path(r'G:/Mon Drive/projet ML')
DATA_DIR = ROOT / 'birdclef-2026'
WORK_DIR = ROOT / 'birdclef_2026_classical_ml_work'
VAL_DIR = WORK_DIR / 'validation'
CACHE_DIR = VAL_DIR / 'v25_multisplit_cache'
REND_CACHE_DIR = Path.cwd().parent / 'feature_caches'
if not REND_CACHE_DIR.exists():
    REND_CACHE_DIR = Path(r'G:/Mon Drive/projet_ML/feature_caches')
SUBWIN_FEATURE_PATH = next((p for p in [REND_CACHE_DIR / 'v31_soundscape_subwindow_features.parquet', VAL_DIR / 'v31_soundscape_subwindow_features.parquet'] if p.exists()), VAL_DIR / 'v31_soundscape_subwindow_features.parquet')

SOUND_DIR = DATA_DIR / 'train_soundscapes'
LABEL_PATH = DATA_DIR / 'train_soundscapes_labels.csv'
SAMPLE_SUB_PATH = DATA_DIR / 'sample_submission.csv'

SR = 32000
SEGMENT_DURATION = 5.0
SPLIT_SEEDS = [42, 123, 2026]
VAL_SIZE_SOUNDSCAPE_FILES = 0.20
N_ROUNDS_SUBWIN = 80
MIN_POS = 3

print('sound dir:', SOUND_DIR.exists(), SOUND_DIR)
print('labels:', LABEL_PATH.exists(), LABEL_PATH)
print('feature cache:', SUBWIN_FEATURE_PATH)

# %%

sample = pd.read_csv(SAMPLE_SUB_PATH)
label_cols = sample.columns[1:].tolist()
label_set = set(label_cols)
n_classes = len(label_cols)
floor = 1.0 / n_classes

labels_df = pd.read_csv(LABEL_PATH).copy()
labels_df['labels'] = labels_df['primary_label'].astype(str).str.split(';').apply(lambda xs: [x for x in xs if x in label_set])

def time_to_seconds(x):
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x)
    if ':' not in s:
        return float(s)
    parts = [float(p) for p in s.split(':')]
    if len(parts) == 3:
        h, m, sec = parts
        return h * 3600 + m * 60 + sec
    if len(parts) == 2:
        m, sec = parts
        return m * 60 + sec
    return parts[0]

labels_df['start_sec'] = labels_df['start'].apply(time_to_seconds)
labels_df['end_sec'] = labels_df['end'].apply(time_to_seconds)
labels_df['offset_sec'] = labels_df['end_sec'].astype(float)
labels_df['source'] = 'train_soundscapes'
labels_df['stable_key'] = (
    labels_df['source'].astype(str) + '|' + labels_df['filename'].astype(str) + '|' +
    labels_df['offset_sec'].astype(float).round(6).astype(str)
)

print('soundscape label rows:', labels_df.shape)
print('files:', labels_df.filename.nunique())
print('label cardinality:', labels_df.labels.apply(len).describe().to_dict())
display(labels_df.head())

# %%

def stats_1d(x, prefix):
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0 or not np.isfinite(x).any():
        return {f'{prefix}_{k}': 0.0 for k in ['mean', 'std', 'min', 'max', 'p10', 'p50', 'p90']}
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return {
        f'{prefix}_mean': float(np.mean(x)),
        f'{prefix}_std': float(np.std(x)),
        f'{prefix}_min': float(np.min(x)),
        f'{prefix}_max': float(np.max(x)),
        f'{prefix}_p10': float(np.percentile(x, 10)),
        f'{prefix}_p50': float(np.percentile(x, 50)),
        f'{prefix}_p90': float(np.percentile(x, 90)),
    }


def frame_band_energy(power, freqs, lo, hi):
    mask = (freqs >= lo) & (freqs < hi)
    if not np.any(mask):
        return np.zeros(power.shape[1], dtype=np.float32)
    return power[mask].mean(axis=0).astype(np.float32)


def chunk_stats(frame_values, frame_times, prefix, chunk_sec=0.5):
    feats = {}
    values = np.asarray(frame_values, dtype=np.float32)
    chunks = []
    for start in np.arange(0.0, SEGMENT_DURATION, chunk_sec):
        mask = (frame_times >= start) & (frame_times < start + chunk_sec)
        if np.any(mask):
            chunks.append(float(np.mean(values[mask])))
    chunks = np.asarray(chunks, dtype=np.float32)
    feats.update(stats_1d(chunks, prefix))
    if chunks.size:
        feats[f'{prefix}_burst'] = float(np.max(chunks) - np.mean(chunks))
        feats[f'{prefix}_peak_ratio'] = float((np.max(chunks) + 1e-6) / (np.mean(chunks) + 1e-6))
        feats[f'{prefix}_active_frac'] = float(np.mean(chunks > (np.mean(chunks) + np.std(chunks))))
    else:
        feats[f'{prefix}_burst'] = 0.0
        feats[f'{prefix}_peak_ratio'] = 1.0
        feats[f'{prefix}_active_frac'] = 0.0
    return feats


def extract_subwindow_features(row):
    path = SOUND_DIR / row['filename']
    start = float(row['start_sec'])
    duration = float(row['end_sec'] - row['start_sec'])
    y, sr = librosa.load(path, sr=SR, mono=True, offset=start, duration=duration)
    target = int(SR * SEGMENT_DURATION)
    if len(y) < target:
        y = np.pad(y, (0, target - len(y)))
    else:
        y = y[:target]

    feats = {}
    feats.update(stats_1d(y, 'wave'))
    feats['wave_abs_mean'] = float(np.mean(np.abs(y)))
    feats['wave_abs_max'] = float(np.max(np.abs(y)))

    n_fft = 1024
    hop = 320
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop, win_length=n_fft)) ** 2
    freqs = librosa.fft_frequencies(sr=SR, n_fft=n_fft)
    times = librosa.frames_to_time(np.arange(S.shape[1]), sr=SR, hop_length=hop)

    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop)[0]
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop)[0]
    centroid = librosa.feature.spectral_centroid(S=S, sr=SR)[0]
    bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=SR)[0]
    rolloff = librosa.feature.spectral_rolloff(S=S, sr=SR, roll_percent=0.85)[0]
    flatness = librosa.feature.spectral_flatness(S=S)[0]

    for name, vals in [
        ('rms', rms), ('zcr', zcr), ('centroid', centroid),
        ('bandwidth', bandwidth), ('rolloff', rolloff), ('flatness', flatness),
    ]:
        feats.update(stats_1d(vals, name))
        feats.update(chunk_stats(vals, times, f'{name}_chunk05', chunk_sec=0.5))
        feats.update(chunk_stats(vals, times, f'{name}_chunk10', chunk_sec=1.0))

    bands = [
        (40, 250), (250, 500), (500, 1000), (1000, 2000),
        (2000, 4000), (4000, 8000), (8000, 12000), (12000, 15000),
    ]
    band_means = []
    for lo, hi in bands:
        vals = np.log1p(frame_band_energy(S, freqs, lo, hi))
        band_means.append(np.mean(vals))
        prefix = f'band_{lo}_{hi}'
        feats.update(stats_1d(vals, prefix))
        feats.update(chunk_stats(vals, times, f'{prefix}_chunk05', chunk_sec=0.5))
    band_means = np.asarray(band_means, dtype=np.float32)
    total = float(np.sum(np.exp(band_means)) + 1e-6)
    for idx, (lo, hi) in enumerate(bands):
        feats[f'band_{lo}_{hi}_share'] = float(np.exp(band_means[idx]) / total)

    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=n_fft, hop_length=hop, n_mels=48, fmin=40, fmax=15000, power=2.0)
    logmel = librosa.power_to_db(mel, ref=np.max)
    for m in range(logmel.shape[0]):
        vals = logmel[m]
        feats[f'logmel_{m:02d}_mean'] = float(np.mean(vals))
        feats[f'logmel_{m:02d}_max'] = float(np.max(vals))
        feats[f'logmel_{m:02d}_p90'] = float(np.percentile(vals, 90))
        feats[f'logmel_{m:02d}_burst'] = float(np.percentile(vals, 95) - np.percentile(vals, 50))

    return feats

# %%

if SUBWIN_FEATURE_PATH.exists():
    subwin_df = pd.read_parquet(SUBWIN_FEATURE_PATH)
    print('Loaded cached subwindow features:', subwin_df.shape)
else:
    rows = []
    t0 = time.perf_counter()
    for i, row in labels_df.iterrows():
        feats = extract_subwindow_features(row)
        feats.update({
            'filename': row['filename'],
            'start': float(row['start_sec']),
            'end': float(row['end_sec']),
            'offset_sec': float(row['offset_sec']),
            'primary_label': row['primary_label'],
            'labels_json': json.dumps(row['labels']),
            'stable_key': row['stable_key'],
        })
        rows.append(feats)
        if (i + 1) % 100 == 0 or (i + 1) == len(labels_df):
            print(f'extracted {i + 1}/{len(labels_df)} elapsed min={(time.perf_counter() - t0) / 60:.2f}')
    subwin_df = pd.DataFrame(rows)
    subwin_df.to_parquet(SUBWIN_FEATURE_PATH, index=False)
    print('Saved:', SUBWIN_FEATURE_PATH, subwin_df.shape)

display(subwin_df.head())
print('feature columns:', len([c for c in subwin_df.columns if c not in ['filename', 'start', 'end', 'offset_sec', 'primary_label', 'labels_json', 'stable_key']]))

# %%

def make_split(row_df, split_seed):
    gss = GroupShuffleSplit(n_splits=1, test_size=VAL_SIZE_SOUNDSCAPE_FILES, random_state=split_seed)
    _, va_idx = next(gss.split(row_df, groups=row_df['filename'].astype(str)))
    val_keys = set(row_df.iloc[va_idx]['stable_key'])
    train_mask = ~row_df['stable_key'].isin(val_keys)
    val_mask = row_df['stable_key'].isin(val_keys)
    return train_mask.values, val_mask.values


def parse_labels_json(s):
    try:
        return [str(x) for x in json.loads(s) if str(x) in label_set]
    except Exception:
        return []


def safe_macro_auc(y_true, y_pred):
    vals = []
    for j in range(y_true.shape[1]):
        yt = y_true[:, j]
        if yt.min() != yt.max():
            vals.append(roc_auc_score(yt, y_pred[:, j]))
    return float(np.mean(vals)) if vals else np.nan


def safe_macro_ap(y_true, y_pred):
    vals = []
    for j in range(y_true.shape[1]):
        yt = y_true[:, j]
        if yt.sum() > 0:
            vals.append(average_precision_score(yt, y_pred[:, j]))
    return float(np.mean(vals)) if vals else np.nan


def entropy_flatten_array(pred, q=0.5, gamma=1.15):
    eps = 1e-9
    P = np.clip(pred.astype(np.float32).copy(), eps, 1.0)
    prob = P / (P.sum(axis=1, keepdims=True) + eps)
    H = -(prob * np.log(prob + eps)).sum(axis=1) / np.log(P.shape[1])
    thr = np.quantile(H, q)
    mask = H <= thr
    if np.any(mask):
        Z = P[mask]
        s0 = Z.sum(axis=1, keepdims=True)
        Zg = Z ** gamma
        P[mask] = Zg * (s0 / (Zg.sum(axis=1, keepdims=True) + eps))
    return np.clip(P, 0.0, 1.0)


def smooth_array_by_file(pred, val_meta, window=15, alpha=0.95):
    if window <= 1 or alpha <= 0:
        return pred.copy()
    out = pred.copy()
    tmp = val_meta.reset_index().rename(columns={'index': 'pos'})
    for _, grp in tmp.groupby('filename', sort=False):
        grp = grp.sort_values('offset_sec')
        idx = grp['pos'].values
        if len(idx) < 2:
            continue
        block = out[idx]
        roll = pd.DataFrame(block).rolling(window, center=True, min_periods=1).mean().values
        out[idx] = (1 - alpha) * block + alpha * roll
    return np.clip(out, 0.0, 1.0)


def load_v30_global_for_fold(fold, val_keys):
    # Recreate the exact V30 global prediction on the same validation rows from cached V25 predictions.
    raw = np.load(CACHE_DIR / f'raw_v7_{fold}_lgbm_r80.npz')['pred'].astype(np.float32)
    lm = np.load(CACHE_DIR / f'logmel64_{fold}_lgbm_r50.npz')['pred'].astype(np.float32)
    # Prototype was tested in V30, but not cached here. For this notebook, compare subwindow against raw+logmel+postprocess.
    # This is slightly conservative. If subwindow helps here, it is worth retesting with prototype included.
    pred = 0.97 * raw + 0.03 * lm
    pred = entropy_flatten_array(np.clip(pred, 0.0, 1.0), q=0.5, gamma=1.15)
    meta = subwin_df[subwin_df['stable_key'].isin(val_keys)].sort_values(['filename', 'offset_sec']).reset_index(drop=True)
    return smooth_array_by_file(pred, meta, window=15, alpha=0.95)

# %%

feature_cols = [c for c in subwin_df.columns if c not in ['filename', 'start', 'end', 'offset_sec', 'primary_label', 'labels_json', 'stable_key']]
X_all = subwin_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32).values
y_all = MultiLabelBinarizer(classes=label_cols).fit_transform([parse_labels_json(s) for s in subwin_df['labels_json']]).astype(np.uint8)

LGB_PARAMS = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'learning_rate': 0.04,
    'num_leaves': 63,
    'feature_fraction': 0.85,
    'bagging_fraction': 0.80,
    'bagging_freq': 5,
    'min_data_in_leaf': 4,
    'lambda_l1': 0.1,
    'lambda_l2': 0.3,
    'verbose': -1,
    'n_jobs': -1,
    'force_col_wise': True,
    'seed': 42,
}

rows = []
blend_grid = [0.00, 0.03, 0.05, 0.08, 0.12, 0.20, 0.30, 0.40]

for split_seed in SPLIT_SEEDS:
    fold = f'seed{split_seed}'
    train_mask, val_mask = make_split(subwin_df, split_seed)
    X_train, X_val = X_all[train_mask], X_all[val_mask]
    y_train, y_val = y_all[train_mask], y_all[val_mask]
    val_meta = subwin_df.loc[val_mask, ['filename', 'offset_sec', 'stable_key']].sort_values(['filename', 'offset_sec']).reset_index(drop=True)

    # Align rows after sorting validation metadata.
    order = subwin_df.loc[val_mask].sort_values(['filename', 'offset_sec']).index
    X_val = subwin_df.loc[order, feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32).values
    y_val = y_all[order]
    val_keys = set(subwin_df.loc[order, 'stable_key'])

    pred_sub = np.full((len(X_val), n_classes), floor, dtype=np.float32)
    trainable = np.where(y_train.sum(axis=0) >= MIN_POS)[0]
    print(f'{fold}: train rows={len(X_train)} val rows={len(X_val)} trainable={len(trainable)}')
    for k, j in enumerate(trainable, 1):
        yj = y_train[:, j]
        n_pos = int(yj.sum())
        n_neg = len(yj) - n_pos
        params = dict(LGB_PARAMS)
        params['scale_pos_weight'] = min(80.0, max(1.0, np.sqrt(n_neg / max(1, n_pos))))
        ds = lgb.Dataset(X_train, label=yj)
        model = lgb.train(params, ds, num_boost_round=N_ROUNDS_SUBWIN)
        pred_sub[:, j] = model.predict(X_val).astype(np.float32)
    pred_sub = smooth_array_by_file(pred_sub, val_meta, window=15, alpha=0.95)

    pred_base = load_v30_global_for_fold(fold, val_keys)
    if pred_base.shape[0] != pred_sub.shape[0]:
        print('Warning: base/subwindow row mismatch; skipping base blend for', fold, pred_base.shape, pred_sub.shape)
        pred_base = pred_sub.copy()

    rows.append({'fold': fold, 'method': 'subwindow_only', 'blend_w': 1.0, 'auc': safe_macro_auc(y_val, pred_sub), 'ap': safe_macro_ap(y_val, pred_sub)})
    rows.append({'fold': fold, 'method': 'base_proxy', 'blend_w': 0.0, 'auc': safe_macro_auc(y_val, pred_base), 'ap': safe_macro_ap(y_val, pred_base)})
    for w in blend_grid:
        pred = np.clip((1 - w) * pred_base + w * pred_sub, 0.0, 1.0)
        rows.append({'fold': fold, 'method': 'base_plus_subwindow', 'blend_w': w, 'auc': safe_macro_auc(y_val, pred), 'ap': safe_macro_ap(y_val, pred)})

results_df = pd.DataFrame(rows)
summary_df = (
    results_df.groupby(['method', 'blend_w'], dropna=False)
    .agg(mean_auc=('auc', 'mean'), min_auc=('auc', 'min'), std_auc=('auc', 'std'),
         mean_ap=('ap', 'mean'), min_ap=('ap', 'min'), std_ap=('ap', 'std'))
    .reset_index()
    .sort_values(['mean_auc', 'min_auc', 'mean_ap'], ascending=False)
)
print('Subwindow validation summary')
display(summary_df)
display(results_df.sort_values(['method', 'blend_w', 'fold']))
