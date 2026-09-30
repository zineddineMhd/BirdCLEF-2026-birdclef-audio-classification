# BirdCLEF+ 2026 experiment export
#
# Clean text export of the original Jupyter/Colab experiment.
# Cell boundaries are preserved with VS Code/Jupytext-style markers.
# Notebook outputs are intentionally omitted; verified metrics are documented in docs/RESULTS.md.

# %% [markdown]
# # BirdCLEF+ 2026 - V35 Submission V34 + Subwindow Branch
#
# Kaggle submission candidate based on V34 public 0.767, with a soundscape subwindow branch validated locally under disjoint GroupKFold.

# %%
from pathlib import Path
import json
import time
import warnings
import gc

import numpy as np
import pandas as pd
import librosa
import lightgbm as lgb
from scipy.signal import butter, filtfilt
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler

warnings.filterwarnings('ignore')
t0_notebook = time.perf_counter()

DATA_DIR_CANDIDATES = [
    Path('/kaggle/input/competitions/birdclef-2026'),
    Path('/kaggle/input/birdclef-2026'),
    Path(r'G:/Mon Drive/projet ML/birdclef-2026'),
]
DATA_DIR = next((p for p in DATA_DIR_CANDIDATES if (p / 'sample_submission.csv').exists()), DATA_DIR_CANDIDATES[0])
TEST_DIR = DATA_DIR / 'test_soundscapes'
TRAIN_SOUNDSCAPES_DIR = DATA_DIR / 'train_soundscapes'
TRAIN_SOUNDSCAPE_LABELS_PATH = DATA_DIR / 'train_soundscapes_labels.csv'
SAMPLE_SUB_PATH = DATA_DIR / 'sample_submission.csv'
OUTPUT_PATH = Path('/kaggle/working/submission.csv') if Path('/kaggle/working').exists() else Path(r'G:/Mon Drive/projet ML/submission_v35_v34_plus_subwindow.csv')

RAW_FEATURE_CANDIDATES = [
    Path.cwd().parent / 'feature_caches' / 'train_features_science_full.parquet',
    Path(r'G:/Mon Drive/projet_ML/feature_caches/train_features_science_full.parquet'),
    Path('/kaggle/input/datasets/zineddinemohammedi/birdclef-2026-classical-features/train_features_science_full.parquet'),
    Path('/kaggle/input/birdclef-2026-classical-features/train_features_science_full.parquet'),
    Path('/kaggle/input/birdclef-2026-classical-ml-work/features/train_features_science_full.parquet'),
    Path('/kaggle/input/birdclef-features-full/train_features_science_full.parquet'),
    Path('/kaggle/input/birdclef-v7-raw-features/train_features_science_full.parquet'),
    Path(r'G:/Mon Drive/projet ML/birdclef_2026_classical_ml_work/features/train_features_science_full.parquet'),
]

LOGMEL_FEATURE_CANDIDATES = [
    Path.cwd().parent / 'feature_caches' / 'v21_logmel_pcen64_features.parquet',
    Path(r'G:/Mon Drive/projet_ML/feature_caches/v21_logmel_pcen64_features.parquet'),
    Path('/kaggle/input/datasets/zineddinemohammedi/v21-logmel-pcen64-features/v21_logmel_pcen64_features.parquet'),
    Path('/kaggle/input/birdclef-v21-logmel-pcen64-features/v21_logmel_pcen64_features.parquet'),
    Path('/kaggle/input/birdclef-logmel-pcen64-features/v21_logmel_pcen64_features.parquet'),
    Path('/kaggle/input/v21-logmel-pcen64-features/v21_logmel_pcen64_features.parquet'),
    Path('/kaggle/input/datasets/zineddinemohammedi/birdclef-v21-logmel-pcen64-features/v21_logmel_pcen64_features.parquet'),
    Path(r'G:/Mon Drive/projet ML/birdclef_2026_classical_ml_work/validation/v21_logmel_pcen64_features.parquet'),
]

SUBWIN_FEATURE_CANDIDATES = [
    Path.cwd().parent / 'feature_caches' / 'v31_soundscape_subwindow_features.parquet',
    Path(r'G:/Mon Drive/projet_ML/feature_caches/v31_soundscape_subwindow_features.parquet'),
    Path('/kaggle/input/datasets/zineddinemohammedi/v31-soundscape-subwindow-features/v31_soundscape_subwindow_features.parquet'),
    Path('/kaggle/input/birdclef-v31-soundscape-subwindow-features/v31_soundscape_subwindow_features.parquet'),
    Path('/kaggle/input/v31-soundscape-subwindow-features/v31_soundscape_subwindow_features.parquet'),
    Path('/kaggle/input/birdclef-subwindow-features/v31_soundscape_subwindow_features.parquet'),
    Path(r'G:/Mon Drive/projet ML/birdclef_2026_classical_ml_work/validation/v31_soundscape_subwindow_features.parquet'),
]

SR = 32000
SEGMENT_SEC = 5.0
N_SAMPLES = int(SR * SEGMENT_SEC)
MIN_POS = 3
SEEDS = [42]
N_ROUNDS_RAW = 80
N_ROUNDS_LOGMEL = 50
N_ROUNDS_SUBWIN = 80
SOURCE_WEIGHT_SS = 1.0

W_RAW = 0.95
W_LOGMEL = 0.03
W_PROTO = 0.02
SUBWIN_BLEND_W = 0.40

SMOOTH_WINDOW = 17
SMOOTH_ALPHA = 0.95
LOGMEL_PCEN_N_MELS = 64
HIGHPASS_HZ = 150

ENTROPY_Q = 0.50
ENTROPY_GAMMA = 1.15
COOC_ALPHA = 0.5
COOC_BLEND_W = 0.12
COOC_DIAG_ZERO = True

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
}

print('DATA_DIR:', DATA_DIR)
print('TEST_DIR:', TEST_DIR)
print('OUTPUT_PATH:', OUTPUT_PATH)
print('weights:', W_RAW, W_LOGMEL, W_PROTO, '| subwindow blend:', SUBWIN_BLEND_W)
print('entropy:', ENTROPY_Q, ENTROPY_GAMMA)
print('cooccurrence:', COOC_ALPHA, COOC_BLEND_W, 'diag_zero=', COOC_DIAG_ZERO)

# %%
def resolve_parquet(candidates, expected_names):
    for p in candidates:
        if p.exists():
            return p
    input_root = Path('/kaggle/input')
    if input_root.exists():
        for name in expected_names:
            matches = list(input_root.rglob(name))
            if matches:
                print('Auto-found parquet:', matches[0])
                return matches[0]
        available = list(input_root.rglob('*.parquet'))[:80]
        msg = 'Required parquet not found. Expected one of: ' + ', '.join(expected_names)
        if available:
            msg += '\nAvailable parquet files:\n' + '\n'.join(str(x) for x in available)
        raise FileNotFoundError(msg)
    raise FileNotFoundError('Required parquet not found. Expected one of: ' + ', '.join(expected_names))


def parse_labels(s, label_set):
    try:
        arr = json.loads(s)
        return [str(z) for z in arr if str(z) in label_set]
    except Exception:
        return []


def parse_row_id(row_id):
    stem, sec = row_id.rsplit('_', 1)
    return f'{stem}.ogg', float(sec)


def stats_1d(vec, prefix):
    vec = np.asarray(vec, dtype=np.float32)
    if vec.size == 0:
        vec = np.zeros(1, dtype=np.float32)
    return {
        f'{prefix}_mean': float(np.mean(vec)),
        f'{prefix}_std': float(np.std(vec)),
        f'{prefix}_min': float(np.min(vec)),
        f'{prefix}_max': float(np.max(vec)),
        f'{prefix}_median': float(np.median(vec)),
        f'{prefix}_p10': float(np.percentile(vec, 10)),
        f'{prefix}_p90': float(np.percentile(vec, 90)),
    }


def stats_2d(mat, prefix):
    out = {}
    mat = np.asarray(mat)
    if mat.ndim == 1:
        mat = mat[None, :]
    for i in range(mat.shape[0]):
        out.update(stats_1d(mat[i], f'{prefix}_{i:02d}'))
    return out


def trim_mean_10(v):
    v = np.sort(np.asarray(v, dtype=np.float32))
    if len(v) == 0:
        return 0.0
    k = int(0.10 * len(v))
    if 2 * k >= len(v):
        return float(np.mean(v))
    return float(np.mean(v[k:len(v)-k]))


def logmel_stats_1d(vec, prefix):
    v = np.asarray(vec, dtype=np.float32)
    if v.size == 0:
        v = np.zeros(1, dtype=np.float32)
    p10, p25, p50, p75, p90 = np.percentile(v, [10, 25, 50, 75, 90])
    vmin = float(np.min(v))
    vmax = float(np.max(v))
    return {
        f'{prefix}_mean': float(np.mean(v)),
        f'{prefix}_std': float(np.std(v)),
        f'{prefix}_min': vmin,
        f'{prefix}_max': vmax,
        f'{prefix}_median': float(np.median(v)),
        f'{prefix}_p10': float(p10),
        f'{prefix}_p25': float(p25),
        f'{prefix}_p50': float(p50),
        f'{prefix}_p75': float(p75),
        f'{prefix}_p90': float(p90),
        f'{prefix}_trim10': trim_mean_10(v),
        f'{prefix}_iqr': float(p75 - p25),
        f'{prefix}_dyn_p90_p10': float(p90 - p10),
        f'{prefix}_range': float(vmax - vmin),
    }


def logmel_stats_2d(mat, prefix):
    out = {}
    m = np.asarray(mat)
    if m.ndim == 1:
        m = m[None, :]
    for i in range(m.shape[0]):
        out.update(logmel_stats_1d(m[i], f'{prefix}_{i:02d}'))
    return out


def prepare_fixed_segment(y):
    if len(y) < int(0.2 * SR):
        return None
    if len(y) < N_SAMPLES:
        y = np.pad(y, (0, N_SAMPLES - len(y)))
    elif len(y) > N_SAMPLES:
        y = y[:N_SAMPLES]
    return np.asarray(y, dtype=np.float32)


def highpass_segment(y):
    x = np.asarray(y, dtype=np.float32)
    if len(x) > 64:
        try:
            b, a = butter(3, HIGHPASS_HZ / (SR / 2), btype='highpass')
            x = filtfilt(b, a, x).astype(np.float32)
        except Exception:
            pass
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def extract_raw_v7_features_prepared(y):
    feats = {}
    n_fft = 2048
    hop = 512
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=20, n_fft=n_fft, hop_length=hop)
    feats.update(stats_2d(mfcc, 'mfcc'))
    feats.update(stats_2d(librosa.feature.delta(mfcc), 'mfcc_d1'))

    centroid = librosa.feature.spectral_centroid(y=y, sr=SR, n_fft=n_fft, hop_length=hop)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=SR, n_fft=n_fft, hop_length=hop)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=SR, n_fft=n_fft, hop_length=hop)
    flatness = librosa.feature.spectral_flatness(y=y, n_fft=n_fft, hop_length=hop)
    contrast = librosa.feature.spectral_contrast(y=y, sr=SR, n_fft=n_fft, hop_length=hop)
    chroma = librosa.feature.chroma_stft(y=y, sr=SR, n_fft=n_fft, hop_length=hop)
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop)
    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop)

    feats.update(stats_2d(centroid, 'centroid'))
    feats.update(stats_2d(bandwidth, 'bandwidth'))
    feats.update(stats_2d(rolloff, 'rolloff'))
    feats.update(stats_2d(flatness, 'flatness'))
    feats.update(stats_2d(contrast, 'contrast'))
    feats.update(stats_2d(chroma, 'chroma'))
    feats.update(stats_2d(zcr, 'zcr'))
    feats.update(stats_2d(rms, 'rms'))

    spec = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop)) ** 2
    freqs = librosa.fft_frequencies(sr=SR, n_fft=n_fft)
    for lo, hi in [(40, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 8000), (8000, 15000)]:
        mask = (freqs >= lo) & (freqs < hi)
        if mask.sum() == 0:
            continue
        band_energy = spec[mask, :].mean(axis=0)
        feats.update(stats_1d(band_energy, f'band_{lo}_{hi}'))
    return feats


def extract_logmel_pcen64_features_prepared(y):
    y = highpass_segment(y)
    n_fft = 2048
    hop = 512
    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=n_fft, hop_length=hop,
        n_mels=LOGMEL_PCEN_N_MELS, fmin=40, fmax=15000, power=2.0,
    )
    logmel = librosa.power_to_db(mel, ref=np.max)
    pcen = librosa.pcen(
        mel * (2**31), sr=SR, hop_length=hop,
        gain=0.98, bias=2, power=0.5, time_constant=0.4,
    )
    feats = {}
    feats.update(logmel_stats_2d(logmel, 'logmel'))
    feats.update(logmel_stats_2d(pcen, 'pcen'))
    return feats


def smooth_per_file(pred_df, label_cols, window=SMOOTH_WINDOW, alpha=SMOOTH_ALPHA):
    out = pred_df.copy()
    tmp = pred_df.copy()
    parsed = tmp['row_id'].map(parse_row_id)
    tmp['filename'] = parsed.map(lambda x: x[0])
    tmp['offset_sec'] = parsed.map(lambda x: x[1]).astype(float)
    for _, grp in tmp.groupby('filename', sort=False):
        grp = grp.sort_values('offset_sec')
        idx = grp.index.values
        if len(idx) < 2:
            continue
        block = out.loc[idx, label_cols].values
        roll = pd.DataFrame(block).rolling(window, center=True, min_periods=1).mean().values
        out.loc[idx, label_cols] = (1 - alpha) * block + alpha * roll
    return out


def scale_pos_weight(n_neg, n_pos):
    ratio = n_neg / max(1, n_pos)
    return min(80.0, max(1.0, np.sqrt(ratio)))


def entropy_flatten_array(pred, q=ENTROPY_Q, gamma=ENTROPY_GAMMA):
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


def train_cooc_matrix(y_train, alpha=COOC_ALPHA, diag_zero=COOC_DIAG_ZERO):
    # C[i, j] approximates P(label j present | label i present).
    y = y_train.astype(np.float32)
    counts = y.sum(axis=0)
    co = y.T @ y
    C = (co + alpha) / (counts[:, None] + alpha * y.shape[1])
    if diag_zero:
        np.fill_diagonal(C, 0.0)
    return C.astype(np.float32)


def apply_cooccurrence_raw_sum(pred, C, blend_w=COOC_BLEND_W):
    P = np.clip(pred.astype(np.float32), 0.0, 1.0)
    support = np.clip(P @ C, 0.0, 1.0)
    return np.clip((1.0 - blend_w) * P + blend_w * support, 0.0, 1.0).astype(np.float32)



def find_optional_parquet(candidates, expected_name):
    for p in candidates:
        if p.exists():
            return p
    input_root = Path('/kaggle/input')
    if input_root.exists():
        matches = list(input_root.rglob(expected_name))
        if matches:
            print('Auto-found optional parquet:', matches[0])
            return matches[0]
    return None


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


def subwin_stats_1d(x, prefix):
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


def subwin_frame_band_energy(power, freqs, lo, hi):
    mask = (freqs >= lo) & (freqs < hi)
    if not np.any(mask):
        return np.zeros(power.shape[1], dtype=np.float32)
    return power[mask].mean(axis=0).astype(np.float32)


def subwin_chunk_stats(frame_values, frame_times, prefix, chunk_sec=0.5):
    feats = {}
    values = np.asarray(frame_values, dtype=np.float32)
    chunks = []
    for start in np.arange(0.0, SEGMENT_SEC, chunk_sec):
        mask = (frame_times >= start) & (frame_times < start + chunk_sec)
        if np.any(mask):
            chunks.append(float(np.mean(values[mask])))
    chunks = np.asarray(chunks, dtype=np.float32)
    feats.update(subwin_stats_1d(chunks, prefix))
    if chunks.size:
        feats[f'{prefix}_burst'] = float(np.max(chunks) - np.mean(chunks))
        feats[f'{prefix}_peak_ratio'] = float((np.max(chunks) + 1e-6) / (np.mean(chunks) + 1e-6))
        feats[f'{prefix}_active_frac'] = float(np.mean(chunks > (np.mean(chunks) + np.std(chunks))))
    else:
        feats[f'{prefix}_burst'] = 0.0
        feats[f'{prefix}_peak_ratio'] = 1.0
        feats[f'{prefix}_active_frac'] = 0.0
    return feats


def extract_subwindow_features_prepared(y):
    y = prepare_fixed_segment(y)
    if y is None:
        y = np.zeros(N_SAMPLES, dtype=np.float32)
    y = np.nan_to_num(np.asarray(y, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    feats = {}
    feats.update(subwin_stats_1d(y, 'wave'))
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
        feats.update(subwin_stats_1d(vals, name))
        feats.update(subwin_chunk_stats(vals, times, f'{name}_chunk05', chunk_sec=0.5))
        feats.update(subwin_chunk_stats(vals, times, f'{name}_chunk10', chunk_sec=1.0))

    bands = [
        (40, 250), (250, 500), (500, 1000), (1000, 2000),
        (2000, 4000), (4000, 8000), (8000, 12000), (12000, 15000),
    ]
    band_means = []
    for lo, hi in bands:
        vals = np.log1p(subwin_frame_band_energy(S, freqs, lo, hi))
        band_means.append(np.mean(vals))
        prefix = f'band_{lo}_{hi}'
        feats.update(subwin_stats_1d(vals, prefix))
        feats.update(subwin_chunk_stats(vals, times, f'{prefix}_chunk05', chunk_sec=0.5))
    band_means = np.asarray(band_means, dtype=np.float32)
    total = float(np.sum(np.exp(band_means)) + 1e-6)
    for idx, (lo, hi) in enumerate(bands):
        feats[f'band_{lo}_{hi}_share'] = float(np.exp(band_means[idx]) / total)

    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=n_fft, hop_length=hop, n_mels=48,
        fmin=40, fmax=15000, power=2.0
    )
    logmel = librosa.power_to_db(mel, ref=np.max)
    for m in range(logmel.shape[0]):
        vals = logmel[m]
        feats[f'logmel_{m:02d}_mean'] = float(np.mean(vals))
        feats[f'logmel_{m:02d}_max'] = float(np.max(vals))
        feats[f'logmel_{m:02d}_p90'] = float(np.percentile(vals, 90))
        feats[f'logmel_{m:02d}_burst'] = float(np.percentile(vals, 95) - np.percentile(vals, 50))

    return feats


def build_train_subwindow_features():
    if not TRAIN_SOUNDSCAPE_LABELS_PATH.exists():
        raise FileNotFoundError(f'Missing train_soundscapes_labels.csv: {TRAIN_SOUNDSCAPE_LABELS_PATH}')
    labels_df = pd.read_csv(TRAIN_SOUNDSCAPE_LABELS_PATH).copy()
    labels_df['labels'] = labels_df['primary_label'].astype(str).str.split(';').apply(
        lambda xs: [str(x) for x in xs if str(x) in label_set]
    )
    labels_df['start_sec'] = labels_df['start'].apply(time_to_seconds)
    labels_df['end_sec'] = labels_df['end'].apply(time_to_seconds)
    labels_df['offset_sec'] = labels_df['end_sec'].astype(float)

    rows = []
    t0 = time.perf_counter()
    for i, row in labels_df.iterrows():
        path = TRAIN_SOUNDSCAPES_DIR / row['filename']
        y, _ = librosa.load(
            path, sr=SR, mono=True,
            offset=float(row['start_sec']),
            duration=float(row['end_sec'] - row['start_sec'])
        )
        feats = extract_subwindow_features_prepared(y)
        feats.update({
            'filename': row['filename'],
            'start': float(row['start_sec']),
            'end': float(row['end_sec']),
            'offset_sec': float(row['offset_sec']),
            'primary_label': row['primary_label'],
            'labels_json': json.dumps(row['labels']),
            'stable_key': 'train_soundscapes|' + str(row['filename']) + '|' + str(round(float(row['offset_sec']), 6)),
        })
        rows.append(feats)
        if (i + 1) % 250 == 0 or (i + 1) == len(labels_df):
            print(f'computed train subwindow {i + 1}/{len(labels_df)} | elapsed min={(time.perf_counter() - t0) / 60:.2f}')
    return pd.DataFrame(rows)

# %%
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
label_cols = sample_sub.columns[1:].tolist()
label_set = set(label_cols)
n_classes = len(label_cols)
floor = 1.0 / n_classes

raw_file = resolve_parquet(RAW_FEATURE_CANDIDATES, ['train_features_science_full.parquet'])
logmel_file = resolve_parquet(LOGMEL_FEATURE_CANDIDATES, ['v21_logmel_pcen64_features.parquet'])
subwin_file = find_optional_parquet(SUBWIN_FEATURE_CANDIDATES, 'v31_soundscape_subwindow_features.parquet')
print('Using raw V7 train features:', raw_file)
print('Using logmel/PCEN64 train features:', logmel_file)
print('Using subwindow train features:', subwin_file if subwin_file is not None else 'compute from train_soundscapes')

raw_df = pd.read_parquet(raw_file)
logmel_df = pd.read_parquet(logmel_file)
subwin_df = pd.read_parquet(subwin_file) if subwin_file is not None else build_train_subwindow_features()
for df in [raw_df, logmel_df]:
    for c in ['row_index', 'split']:
        if c in df.columns:
            df.drop(columns=[c], inplace=True)

META_COLS = ['source', 'filename', 'offset_sec', 'labels', 'primary_label']
for name, df in [('raw', raw_df), ('logmel', logmel_df)]:
    missing = [c for c in META_COLS if c not in df.columns]
    if missing:
        raise RuntimeError(f'Missing metadata in {name}: {missing}')
    df.sort_values(META_COLS, inplace=True)
    df.reset_index(drop=True, inplace=True)

print('raw rows:', len(raw_df), '| logmel rows:', len(logmel_df), '| subwindow rows:', len(subwin_df))

Y_raw = MultiLabelBinarizer(classes=label_cols).fit_transform([parse_labels(s, label_set) for s in raw_df['labels']]).astype(np.uint8)
Y_logmel = MultiLabelBinarizer(classes=label_cols).fit_transform([parse_labels(s, label_set) for s in logmel_df['labels']]).astype(np.uint8)
Y_subwin = MultiLabelBinarizer(classes=label_cols).fit_transform([
    parse_labels(s, label_set) for s in subwin_df['labels_json']
]).astype(np.uint8)

raw_feature_cols = [c for c in raw_df.columns if c not in META_COLS]
logmel_feature_cols = [c for c in logmel_df.columns if c not in META_COLS]
SUBWIN_META_COLS = ['filename', 'start', 'end', 'offset_sec', 'primary_label', 'labels_json', 'stable_key']
subwin_feature_cols = [c for c in subwin_df.columns if c not in SUBWIN_META_COLS]

X_raw_raw = raw_df[raw_feature_cols].replace([np.inf, -np.inf], np.nan).astype(np.float32)
raw_imp = SimpleImputer(strategy='median')
X_raw_imp = raw_imp.fit_transform(X_raw_raw)
raw_vt = VarianceThreshold(threshold=1e-8)
X_raw = raw_vt.fit_transform(X_raw_imp).astype(np.float32)

X_logmel_raw = logmel_df[logmel_feature_cols].replace([np.inf, -np.inf], np.nan).astype(np.float32)
logmel_imp = SimpleImputer(strategy='median')
X_logmel_imp = logmel_imp.fit_transform(X_logmel_raw)
logmel_vt = VarianceThreshold(threshold=1e-8)
X_logmel = logmel_vt.fit_transform(X_logmel_imp).astype(np.float32)

X_subwin_raw = subwin_df[subwin_feature_cols].replace([np.inf, -np.inf], np.nan).astype(np.float32)
subwin_imp = SimpleImputer(strategy='median')
X_subwin_imp = subwin_imp.fit_transform(X_subwin_raw)
subwin_vt = VarianceThreshold(threshold=1e-8)
X_subwin = subwin_vt.fit_transform(X_subwin_imp).astype(np.float32)

# Prototype branch on raw V7 selected features.
proto_scaler = StandardScaler()
X_raw_std = proto_scaler.fit_transform(X_raw).astype(np.float32)
X_raw_norm = X_raw_std / np.maximum(np.linalg.norm(X_raw_std, axis=1, keepdims=True), 1e-12)
class_counts = Y_raw.sum(axis=0).astype(np.float32)
proto = np.zeros((n_classes, X_raw_norm.shape[1]), dtype=np.float32)
for j in range(n_classes):
    idx = np.where(Y_raw[:, j] > 0)[0]
    if len(idx) > 0:
        proto[j] = X_raw_norm[idx].mean(axis=0)
proto_norm = proto / np.maximum(np.linalg.norm(proto, axis=1, keepdims=True), 1e-12)

print('raw matrix:', X_raw.shape)
print('logmel matrix:', X_logmel.shape)
print('prototype matrix:', proto_norm.shape)
print('subwindow matrix:', X_subwin.shape)
print('trainable raw:', int((Y_raw.sum(axis=0) >= MIN_POS).sum()), '/', n_classes)
print('trainable logmel:', int((Y_logmel.sum(axis=0) >= MIN_POS).sum()), '/', n_classes)
print('trainable subwindow:', int((Y_subwin.sum(axis=0) >= MIN_POS).sum()), '/', n_classes)

# %%
def train_predict_ovr_lgbm(X_train, Y_train, source_values, X_test, num_rounds, model_name):
    """Train one class model, predict immediately, then free it.

    This produces the same kind of OVR prediction as storing all models,
    but avoids holding hundreds of LightGBM boosters in RAM.
    """
    trainable = np.where(Y_train.sum(axis=0) >= MIN_POS)[0]
    sample_weight = np.ones(len(source_values), dtype=np.float32)
    if SOURCE_WEIGHT_SS != 1.0:
        sample_weight[np.asarray(source_values).astype(str) == 'train_soundscapes'] = SOURCE_WEIGHT_SS

    pred_acc = np.zeros((len(X_test), n_classes), dtype=np.float32)
    t0 = time.perf_counter()

    for seed in SEEDS:
        print(f'\nTraining/predicting {model_name} seed {seed}: {len(trainable)}/{n_classes} classes')
        params_seed = dict(LGB_BASE_PARAMS)
        params_seed['seed'] = seed
        params_seed['bagging_seed'] = seed + 11
        params_seed['feature_fraction_seed'] = seed + 17
        params_seed['data_random_seed'] = seed + 23

        pred_seed = np.full((len(X_test), n_classes), floor, dtype=np.float32)
        for idx, j in enumerate(trainable, 1):
            yj = Y_train[:, j]
            n_pos = int(yj.sum())
            n_neg = len(yj) - n_pos
            params = dict(params_seed)
            params['scale_pos_weight'] = scale_pos_weight(n_neg, n_pos)

            ds = lgb.Dataset(X_train, label=yj, weight=sample_weight)
            model = lgb.train(params, ds, num_boost_round=num_rounds)
            pred_seed[:, j] = model.predict(X_test).astype(np.float32)

            del model, ds
            if idx % 40 == 0 or idx == len(trainable):
                gc.collect()
                print(f'  {model_name}: {idx}/{len(trainable)}')

        pred_acc += pred_seed
        del pred_seed
        gc.collect()

    pred = pred_acc / len(SEEDS)
    print(f'{model_name} train+predict elapsed min:', (time.perf_counter() - t0) / 60)
    return pred



def train_predict_subwindow_lgbm(X_train, Y_train, X_test):
    trainable = np.where(Y_train.sum(axis=0) >= MIN_POS)[0]
    pred_acc = np.zeros((len(X_test), n_classes), dtype=np.float32)
    t0 = time.perf_counter()

    for seed in SEEDS:
        print(f'\nTraining/predicting subwindow seed {seed}: {len(trainable)}/{n_classes} classes')
        params_seed = dict(LGB_BASE_PARAMS)
        params_seed['num_leaves'] = 63
        params_seed['seed'] = seed
        params_seed['bagging_seed'] = seed + 11
        params_seed['feature_fraction_seed'] = seed + 17
        params_seed['data_random_seed'] = seed + 23

        pred_seed = np.full((len(X_test), n_classes), floor, dtype=np.float32)
        for idx, j in enumerate(trainable, 1):
            yj = Y_train[:, j]
            n_pos = int(yj.sum())
            n_neg = len(yj) - n_pos
            params = dict(params_seed)
            params['scale_pos_weight'] = scale_pos_weight(n_neg, n_pos)

            ds = lgb.Dataset(X_train, label=yj)
            model = lgb.train(params, ds, num_boost_round=N_ROUNDS_SUBWIN)
            pred_seed[:, j] = model.predict(X_test).astype(np.float32)

            del model, ds
            if idx % 40 == 0 or idx == len(trainable):
                gc.collect()
                print(f'  subwindow: {idx}/{len(trainable)}')

        pred_acc += pred_seed
        del pred_seed
        gc.collect()

    pred = pred_acc / len(SEEDS)
    print('subwindow train+predict elapsed min:', (time.perf_counter() - t0) / 60)
    return pred

# %%
# Build raw and logmel test features in one audio pass.
test_rows = sample_sub[['row_id']].copy()
parsed = test_rows['row_id'].map(parse_row_id)
test_rows['filename'] = parsed.map(lambda x: x[0])
test_rows['offset_sec'] = parsed.map(lambda x: x[1])

raw_records = []
logmel_records = []
subwin_records = []
t0 = time.perf_counter()

for file_idx, (filename, grp) in enumerate(test_rows.groupby('filename', sort=False), 1):
    audio_path = TEST_DIR / filename
    if not audio_path.exists():
        for row_id in grp['row_id'].tolist():
            raw_records.append({'row_id': row_id})
            logmel_records.append({'row_id': row_id})
            subwin_records.append({'row_id': row_id})
        continue

    y, _ = librosa.load(audio_path, sr=SR, mono=True)
    for row in grp.itertuples(index=False):
        start = int(float(row.offset_sec) * SR)
        seg = y[start:start + N_SAMPLES] if start < len(y) else np.zeros(0, dtype=np.float32)
        seg = prepare_fixed_segment(seg)

        if seg is None:
            f_raw = {}
            f_logmel = {}
            f_subwin = {}
        else:
            f_raw = extract_raw_v7_features_prepared(seg)
            f_logmel = extract_logmel_pcen64_features_prepared(seg)
            f_subwin = extract_subwindow_features_prepared(seg)

        f_raw['row_id'] = row.row_id
        f_logmel['row_id'] = row.row_id
        f_subwin['row_id'] = row.row_id
        raw_records.append(f_raw)
        logmel_records.append(f_logmel)
        subwin_records.append(f_subwin)

    if file_idx % 25 == 0:
        print(f'Extracted test features for {file_idx} files | elapsed min {(time.perf_counter() - t0) / 60:.1f}')

raw_test_df = pd.DataFrame(raw_records).set_index('row_id').reindex(sample_sub['row_id']).reset_index()
logmel_test_df = pd.DataFrame(logmel_records).set_index('row_id').reindex(sample_sub['row_id']).reset_index()
subwin_test_df = pd.DataFrame(subwin_records).set_index('row_id').reindex(sample_sub['row_id']).reset_index()

for c in raw_feature_cols:
    if c not in raw_test_df.columns:
        raw_test_df[c] = np.nan
for c in logmel_feature_cols:
    if c not in logmel_test_df.columns:
        logmel_test_df[c] = np.nan
for c in subwin_feature_cols:
    if c not in subwin_test_df.columns:
        subwin_test_df[c] = np.nan

X_raw_test_raw = raw_test_df[raw_feature_cols].replace([np.inf, -np.inf], np.nan).astype(np.float32)
X_raw_test = raw_vt.transform(raw_imp.transform(X_raw_test_raw)).astype(np.float32)

X_logmel_test_raw = logmel_test_df[logmel_feature_cols].replace([np.inf, -np.inf], np.nan).astype(np.float32)
X_logmel_test = logmel_vt.transform(logmel_imp.transform(X_logmel_test_raw)).astype(np.float32)

X_subwin_test_raw = subwin_test_df[subwin_feature_cols].replace([np.inf, -np.inf], np.nan).astype(np.float32)
X_subwin_test = subwin_vt.transform(subwin_imp.transform(X_subwin_test_raw)).astype(np.float32)

X_test_std = proto_scaler.transform(X_raw_test).astype(np.float32)
X_test_norm = X_test_std / np.maximum(np.linalg.norm(X_test_std, axis=1, keepdims=True), 1e-12)
pred_proto = X_test_norm @ proto_norm.T
pred_proto = ((pred_proto + 1.0) / 2.0).astype(np.float32)
pred_proto[:, class_counts == 0] = floor
pred_proto = np.clip(pred_proto, 0.0, 1.0)

print('test rows:', len(sample_sub))
print('raw test matrix:', X_raw_test.shape)
print('logmel test matrix:', X_logmel_test.shape)
print('subwindow test matrix:', X_subwin_test.shape)
print('test feature extraction elapsed min:', (time.perf_counter() - t0) / 60)

# %%
pred_raw = train_predict_ovr_lgbm(
    X_raw, Y_raw, raw_df['source'].values, X_raw_test, N_ROUNDS_RAW, 'raw_v7'
)
# Free raw training matrix before logmel training to reduce memory pressure.
del X_raw, X_raw_raw, X_raw_imp
try:
    del raw_df
except NameError:
    pass
gc.collect()

pred_logmel = train_predict_ovr_lgbm(
    X_logmel, Y_logmel, logmel_df['source'].values, X_logmel_test, N_ROUNDS_LOGMEL, 'logmel64'
)
# Free logmel training matrix before final blend.
del X_logmel, X_logmel_raw, X_logmel_imp
try:
    del logmel_df
except NameError:
    pass
gc.collect()

# V33 best local post-process:
# blend -> entropy -> cooccurrence -> smoothing.
pred_final = W_RAW * pred_raw + W_LOGMEL * pred_logmel + W_PROTO * pred_proto
pred_final = np.clip(pred_final, 0.0, 1.0)
pred_final = entropy_flatten_array(pred_final, q=ENTROPY_Q, gamma=ENTROPY_GAMMA)

cooc_C = train_cooc_matrix(Y_raw, alpha=COOC_ALPHA, diag_zero=COOC_DIAG_ZERO)
pred_final = apply_cooccurrence_raw_sum(pred_final, cooc_C, blend_w=COOC_BLEND_W)

submission = pd.DataFrame(pred_final, columns=label_cols)
submission.insert(0, 'row_id', sample_sub['row_id'])
submission_base_final = smooth_per_file(submission[['row_id'] + label_cols], label_cols, window=SMOOTH_WINDOW, alpha=SMOOTH_ALPHA)

pred_subwin = train_predict_subwindow_lgbm(X_subwin, Y_subwin, X_subwin_test)
del X_subwin, X_subwin_raw, X_subwin_imp
gc.collect()

pred_subwin = entropy_flatten_array(pred_subwin, q=ENTROPY_Q, gamma=ENTROPY_GAMMA)
subwin_cooc_C = train_cooc_matrix(Y_subwin, alpha=COOC_ALPHA, diag_zero=COOC_DIAG_ZERO)
pred_subwin = apply_cooccurrence_raw_sum(pred_subwin, subwin_cooc_C, blend_w=COOC_BLEND_W)
subwin_submission = pd.DataFrame(pred_subwin, columns=label_cols)
subwin_submission.insert(0, 'row_id', sample_sub['row_id'])
subwin_submission_final = smooth_per_file(subwin_submission[['row_id'] + label_cols], label_cols, window=SMOOTH_WINDOW, alpha=SMOOTH_ALPHA)

base_values = submission_base_final[label_cols].values.astype(np.float32)
subwin_values = subwin_submission_final[label_cols].values.astype(np.float32)
final_values = np.clip((1.0 - SUBWIN_BLEND_W) * base_values + SUBWIN_BLEND_W * subwin_values, 0.0, 1.0)
submission_final = pd.DataFrame(final_values, columns=label_cols)
submission_final.insert(0, 'row_id', sample_sub['row_id'])
submission_final.to_csv(OUTPUT_PATH, index=False)

print('Saved:', OUTPUT_PATH)
print('shape:', submission_final.shape)
print('blend:', W_RAW, '*raw +', W_LOGMEL, '*logmel +', W_PROTO, '*prototype')
print('entropy:', ENTROPY_Q, ENTROPY_GAMMA)
print('cooccurrence before smoothing:', 'alpha=', COOC_ALPHA, 'blend_w=', COOC_BLEND_W, 'diag_zero=', COOC_DIAG_ZERO)
print('smoothing:', SMOOTH_WINDOW, SMOOTH_ALPHA)
print('subwindow blend:', SUBWIN_BLEND_W, '| rounds:', N_ROUNDS_SUBWIN)
print('total elapsed min:', (time.perf_counter() - t0_notebook) / 60)
submission_final.head()

# %%
check = pd.read_csv(OUTPUT_PATH)
assert list(check.columns) == list(sample_sub.columns), 'Column mismatch with sample_submission'
assert len(check) == len(sample_sub), 'Row count mismatch with sample_submission'
assert check.isna().sum().sum() == 0, 'NaN values in submission'
assert np.isfinite(check.iloc[:, 1:].values).all(), 'Non-finite values in predictions'
assert (check.iloc[:, 1:].values >= 0).all() and (check.iloc[:, 1:].values <= 1).all(), 'Predictions out of [0,1]'
print('Submission check OK')
print(check.iloc[:3, :8])