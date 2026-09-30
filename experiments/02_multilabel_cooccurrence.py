# BirdCLEF+ 2026 experiment export
#
# Clean text export of the original Jupyter/Colab experiment.
# Cell boundaries are preserved with VS Code/Jupytext-style markers.
# Notebook outputs are intentionally omitted; verified metrics are documented in docs/RESULTS.md.

# %% [markdown]
# # BirdCLEF 2026 - V30 Local Multi-Label Cooccurrence Validation
#
# Objectif: tester localement des méthodes classiques qui corrigent le défaut du one-vs-rest: chaque classe est prédite séparément alors que les soundscapes sont multi-label.
#
# Ce notebook ne crée pas de soumission Kaggle. Il réutilise les prédictions multi-split déjà en cache:
# - `raw_v7` LightGBM;
# - `logmel64` LightGBM;
# - prototype/cosine;
# - splits groupés par fichier soundscape.
#
# Méthodes testées:
# - baseline V27-like;
# - baseline globale V30 candidate: entropy + smoothing plus long;
# - correction par matrice de cooccurrence apprise sur le train du fold;
# - stacking/calibration classique par classe sur les probabilités et features de cooccurrence.
#
# Règle: si une méthode ne gagne pas en leave-one-fold-out multi-split, on ne la convertit pas en notebook Kaggle.

# %%

from pathlib import Path
import json
import time
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 80)
pd.set_option('display.width', 220)

ROOT = Path(r'G:/Mon Drive/projet ML')
DATA_DIR = ROOT / 'birdclef-2026'
WORK_DIR = ROOT / 'birdclef_2026_classical_ml_work'
FEATURE_DIR = WORK_DIR / 'features'
VAL_DIR = WORK_DIR / 'validation'
CACHE_DIR = VAL_DIR / 'v25_multisplit_cache'

REND_CACHE_DIR = Path.cwd().parent / 'feature_caches'
if not REND_CACHE_DIR.exists():
    REND_CACHE_DIR = Path(r'G:/Mon Drive/projet_ML/feature_caches')
RAW_V7_FEATURE_PATH = next((p for p in [REND_CACHE_DIR / 'train_features_science_full.parquet', FEATURE_DIR / 'train_features_science_full.parquet'] if p.exists()), FEATURE_DIR / 'train_features_science_full.parquet')
LOGMEL_PCEN64_FEATURE_PATH = next((p for p in [REND_CACHE_DIR / 'v21_logmel_pcen64_features.parquet', VAL_DIR / 'v21_logmel_pcen64_features.parquet'] if p.exists()), VAL_DIR / 'v21_logmel_pcen64_features.parquet')
SAMPLE_SUB_PATH = DATA_DIR / 'sample_submission.csv'
TAXONOMY_PATH = DATA_DIR / 'taxonomy.csv'

SPLIT_SEEDS = [42, 123, 2026, 777, 999]
VAL_SIZE_SOUNDSCAPE_FILES = 0.20
N_ROUNDS_BASE = 80
N_ROUNDS_LOGMEL = 50
MIN_POS = 3
SOURCE_WEIGHT_SS = 1.0

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

print('raw:', RAW_V7_FEATURE_PATH.exists(), RAW_V7_FEATURE_PATH)
print('logmel:', LOGMEL_PCEN64_FEATURE_PATH.exists(), LOGMEL_PCEN64_FEATURE_PATH)
print('cache:', CACHE_DIR.exists(), CACHE_DIR)

# %%

sample = pd.read_csv(SAMPLE_SUB_PATH)
label_cols = sample.columns[1:].tolist()
label_set = set(label_cols)
n_classes = len(label_cols)
floor = 1.0 / n_classes
META_COLS = ['source', 'filename', 'offset_sec', 'labels', 'primary_label']

taxonomy = pd.read_csv(TAXONOMY_PATH)
label_to_group = taxonomy.set_index('primary_label')['class_name'].astype(str).to_dict()
class_groups = np.array([label_to_group.get(c, 'Unknown') for c in label_cols])
group_names = sorted(pd.Series(class_groups).unique())
group_to_idx = {g: i for i, g in enumerate(group_names)}
class_group_idx = np.array([group_to_idx[g] for g in class_groups])


def parse_labels(s):
    try:
        return [str(x) for x in json.loads(s) if str(x) in label_set]
    except Exception:
        return []


def load_feature_df(path):
    df = pd.read_parquet(path).copy()
    for c in ['row_index', 'split']:
        if c in df.columns:
            df = df.drop(columns=[c])
    df = df.sort_values(META_COLS).reset_index(drop=True)
    df['stable_key'] = (
        df['source'].astype(str) + '|' + df['filename'].astype(str) + '|' +
        df['offset_sec'].astype(float).round(6).astype(str) + '|' + df['primary_label'].astype(str)
    )
    return df


raw_df = load_feature_df(RAW_V7_FEATURE_PATH)
logmel_df = load_feature_df(LOGMEL_PCEN64_FEATURE_PATH)
assert len(raw_df) == len(logmel_df)
assert np.all(raw_df['stable_key'].values == logmel_df['stable_key'].values)

print('raw_df:', raw_df.shape, raw_df['source'].value_counts().to_dict())
print('taxonomy groups:', pd.Series(class_groups).value_counts().to_dict())

# %%

def make_split(row_df, split_seed):
    ss = row_df[row_df['source'].eq('train_soundscapes')].copy()
    gss = GroupShuffleSplit(n_splits=1, test_size=VAL_SIZE_SOUNDSCAPE_FILES, random_state=split_seed)
    _, va_idx = next(gss.split(ss, groups=ss['filename'].astype(str)))
    val_keys = set(ss.iloc[va_idx]['stable_key'])
    train_mask = ~row_df['stable_key'].isin(val_keys)
    val_mask = row_df['stable_key'].isin(val_keys)
    return train_mask.values, val_mask.values


def prepare_xy(df, train_mask, val_mask):
    y_train = MultiLabelBinarizer(classes=label_cols).fit_transform(
        [parse_labels(s) for s in df.loc[train_mask, 'labels']]
    ).astype(np.uint8)
    y_val = MultiLabelBinarizer(classes=label_cols).fit_transform(
        [parse_labels(s) for s in df.loc[val_mask, 'labels']]
    ).astype(np.uint8)
    feature_cols = [c for c in df.columns if c not in META_COLS + ['stable_key']]
    x_train_raw = df.loc[train_mask, feature_cols].replace([np.inf, -np.inf], np.nan).astype(np.float32)
    x_val_raw = df.loc[val_mask, feature_cols].replace([np.inf, -np.inf], np.nan).astype(np.float32)
    imp = SimpleImputer(strategy='median')
    x_train_imp = imp.fit_transform(x_train_raw)
    x_val_imp = imp.transform(x_val_raw)
    vt = VarianceThreshold(threshold=1e-8)
    x_train = vt.fit_transform(x_train_imp).astype(np.float32)
    x_val = vt.transform(x_val_imp).astype(np.float32)
    val_meta = df.loc[val_mask, META_COLS + ['stable_key']].reset_index(drop=True)
    train_source = df.loc[train_mask, 'source'].astype(str).values
    return x_train, x_val, y_train, y_val, val_meta, train_source


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


def per_class_auc(y_true, y_pred):
    out = np.full(y_true.shape[1], np.nan, dtype=np.float32)
    for j in range(y_true.shape[1]):
        yt = y_true[:, j]
        if yt.min() != yt.max():
            out[j] = roc_auc_score(yt, y_pred[:, j])
    return out


def entropy_flatten_array(pred, q=0.5, gamma=1.15):
    eps = 1e-9
    P = np.clip(pred.astype(np.float32).copy(), eps, 1.0)
    K = P.shape[1]
    prob = P / (P.sum(axis=1, keepdims=True) + eps)
    H = -(prob * np.log(prob)).sum(axis=1) / np.log(K)
    thr = np.quantile(H, q)
    mask = H <= thr
    if np.any(mask):
        Z = P[mask]
        s0 = Z.sum(axis=1, keepdims=True)
        Zg = Z ** gamma
        P[mask] = Zg * (s0 / (Zg.sum(axis=1, keepdims=True) + eps))
    return np.clip(P, 0.0, 1.0)


def smooth_array_by_file(pred, val_meta, window=5, alpha=0.7):
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


def prototype_predict(x_train, y_train, x_val):
    scaler = StandardScaler()
    xt = scaler.fit_transform(x_train).astype(np.float32)
    xv = scaler.transform(x_val).astype(np.float32)
    xt = xt / np.maximum(np.linalg.norm(xt, axis=1, keepdims=True), 1e-12)
    xv = xv / np.maximum(np.linalg.norm(xv, axis=1, keepdims=True), 1e-12)
    proto = np.zeros((n_classes, xt.shape[1]), dtype=np.float32)
    counts = y_train.sum(axis=0)
    for j in range(n_classes):
        idx = np.where(y_train[:, j] > 0)[0]
        if len(idx):
            proto[j] = xt[idx].mean(axis=0)
    proto = proto / np.maximum(np.linalg.norm(proto, axis=1, keepdims=True), 1e-12)
    pred = ((xv @ proto.T) + 1.0) / 2.0
    pred[:, counts == 0] = floor
    return np.clip(pred, 0.0, 1.0).astype(np.float32)


def scale_pos_weight(n_neg, n_pos):
    ratio = n_neg / max(1, n_pos)
    return min(80.0, max(1.0, np.sqrt(ratio)))


def train_or_load_lgbm_cache(x_train, y_train, x_val, train_source, cache_path, rounds, name):
    if cache_path.exists():
        pred = np.load(cache_path)['pred'].astype(np.float32)
        print('loaded cache:', cache_path.name, pred.shape)
        return pred

    trainable = np.where(y_train.sum(axis=0) >= MIN_POS)[0]
    sample_weight = np.ones(len(train_source), dtype=np.float32)
    sample_weight[np.asarray(train_source).astype(str) == 'train_soundscapes'] = SOURCE_WEIGHT_SS
    pred = np.full((len(x_val), n_classes), floor, dtype=np.float32)
    seed = 42
    params_seed = dict(LGB_BASE_PARAMS)
    params_seed['seed'] = seed
    params_seed['bagging_seed'] = seed + 11
    params_seed['feature_fraction_seed'] = seed + 17
    params_seed['data_random_seed'] = seed + 23
    t0 = time.perf_counter()
    print(f'training missing cache {name}: {len(trainable)}/{n_classes} classes, rounds={rounds}')
    for k, j in enumerate(trainable, 1):
        yj = y_train[:, j]
        n_pos = int(yj.sum())
        n_neg = len(yj) - n_pos
        params = dict(params_seed)
        params['scale_pos_weight'] = scale_pos_weight(n_neg, n_pos)
        ds = lgb.Dataset(x_train, label=yj, weight=sample_weight)
        model = lgb.train(params, ds, num_boost_round=rounds)
        pred[:, j] = model.predict(x_val).astype(np.float32)
        if k % 40 == 0 or k == len(trainable):
            print(f'  {name}: {k}/{len(trainable)}')
    np.savez_compressed(cache_path, pred=pred)
    print('saved cache:', cache_path, 'elapsed min:', (time.perf_counter() - t0) / 60)
    return pred

# %%

def make_v27_like(fd):
    pred = 0.95 * fd['pred_raw'] + 0.03 * fd['pred_logmel'] + 0.02 * fd['pred_proto']
    pred = np.clip(pred, 0.0, 1.0)
    return smooth_array_by_file(pred, fd['val_meta'], window=9, alpha=0.95)


def make_v30_global(fd):
    # Best global candidate from V29/V30 grid:
    # raw 0.95 + logmel 0.03 + proto 0.02, entropy q=0.5 gamma=1.15, smoothing 15/0.95.
    pred = 0.95 * fd['pred_raw'] + 0.03 * fd['pred_logmel'] + 0.02 * fd['pred_proto']
    pred = entropy_flatten_array(np.clip(pred, 0.0, 1.0), q=0.5, gamma=1.15)
    return smooth_array_by_file(pred, fd['val_meta'], window=15, alpha=0.95)


fold_data = {}
t0 = time.perf_counter()
for split_seed in SPLIT_SEEDS:
    fold = f'seed{split_seed}'
    train_mask, val_mask = make_split(raw_df, split_seed)
    x_train, x_val, y_train, y_val, val_meta, train_source = prepare_xy(raw_df, train_mask, val_mask)
    raw_cache = CACHE_DIR / f'raw_v7_{fold}_lgbm_r{N_ROUNDS_BASE}.npz'
    pred_raw = train_or_load_lgbm_cache(
        x_train, y_train, x_val, train_source, raw_cache, N_ROUNDS_BASE, f'raw_v7_{fold}'
    )

    lm_train_mask, lm_val_mask = make_split(logmel_df, split_seed)
    x_lm_train, x_lm_val, y_lm_train, y_lm_val, lm_val_meta, lm_train_source = prepare_xy(logmel_df, lm_train_mask, lm_val_mask)
    assert np.array_equal(y_val, y_lm_val), 'Y mismatch raw/logmel for validation fold'
    lm_cache = CACHE_DIR / f'logmel64_{fold}_lgbm_r{N_ROUNDS_LOGMEL}.npz'
    pred_logmel = train_or_load_lgbm_cache(
        x_lm_train, y_lm_train, x_lm_val, lm_train_source, lm_cache, N_ROUNDS_LOGMEL, f'logmel64_{fold}'
    )
    pred_proto = prototype_predict(x_train, y_train, x_val)
    fold_data[fold] = {
        'train_mask': train_mask,
        'val_mask': val_mask,
        'y_train': y_train,
        'y_val': y_val,
        'val_meta': val_meta,
        'pred_raw': pred_raw,
        'pred_logmel': pred_logmel,
        'pred_proto': pred_proto,
    }
    fold_data[fold]['pred_v27'] = make_v27_like(fold_data[fold])
    fold_data[fold]['pred_v30_global'] = make_v30_global(fold_data[fold])
    print(fold,
          'v27_auc:', safe_macro_auc(y_val, fold_data[fold]['pred_v27']),
          'v30_global_auc:', safe_macro_auc(y_val, fold_data[fold]['pred_v30_global']),
          'v30_global_ap:', safe_macro_ap(y_val, fold_data[fold]['pred_v30_global']))

print('load/cache elapsed min:', (time.perf_counter() - t0) / 60)

# %%

def train_cooc_matrix(y_train, alpha=1.0, diag_mode='zero'):
    # C[i, j] ~= P(label j present | label i present).
    y = y_train.astype(np.float32)
    counts = y.sum(axis=0)
    co = y.T @ y
    C = (co + alpha) / (counts[:, None] + alpha * n_classes)
    if diag_mode == 'zero':
        np.fill_diagonal(C, 0.0)
    elif diag_mode == 'keep':
        pass
    else:
        raise ValueError(diag_mode)
    return C.astype(np.float32)


def cooc_support(pred, C, mode='weighted_mean'):
    P = np.clip(pred.astype(np.float32), 0.0, 1.0)
    if mode == 'weighted_mean':
        return (P @ C) / (P.sum(axis=1, keepdims=True) + 1e-6)
    if mode == 'raw_sum':
        return np.clip(P @ C, 0.0, 1.0)
    if mode == 'max_pair':
        # Max over possible source labels. Slower but captures "if A then B" edges.
        out = np.zeros_like(P)
        for i in range(P.shape[0]):
            out[i] = np.max(P[i, :, None] * C, axis=0)
        return np.clip(out, 0.0, 1.0)
    raise ValueError(mode)


cooc_rows = []
cooc_preds = {}
for alpha in [0.1, 0.5, 1.0, 2.0, 5.0, 20.0]:
    for diag_mode in ['zero', 'keep']:
        for support_mode in ['weighted_mean', 'raw_sum']:
            for blend_w in [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.22, 0.25]:
                fold_aucs = []
                fold_aps = []
                key = f'a{alpha}_diag{diag_mode}_mode{support_mode}_w{blend_w}'
                for fold, fd in fold_data.items():
                    C = train_cooc_matrix(fd['y_train'], alpha=alpha, diag_mode=diag_mode)
                    sup = cooc_support(fd['pred_v30_global'], C, mode=support_mode)
                    pred = np.clip((1 - blend_w) * fd['pred_v30_global'] + blend_w * sup, 0.0, 1.0)
                    fold_aucs.append(safe_macro_auc(fd['y_val'], pred))
                    fold_aps.append(safe_macro_ap(fd['y_val'], pred))
                    cooc_preds[(key, fold)] = pred
                cooc_rows.append({
                    'key': key,
                    'alpha': alpha,
                    'diag_mode': diag_mode,
                    'support_mode': support_mode,
                    'blend_w': blend_w,
                    'mean_auc': float(np.mean(fold_aucs)),
                    'min_auc': float(np.min(fold_aucs)),
                    'std_auc': float(np.std(fold_aucs, ddof=1)),
                    'mean_ap': float(np.mean(fold_aps)),
                    'min_ap': float(np.min(fold_aps)),
                    'std_ap': float(np.std(fold_aps, ddof=1)),
                })

cooc_df = pd.DataFrame(cooc_rows).sort_values(['mean_auc', 'min_auc', 'mean_ap'], ascending=False)
print('Cooccurrence correction results')
display(cooc_df.head(30))

# %%

def build_meta_features(pred_base, y_train_for_cooc):
    C = train_cooc_matrix(y_train_for_cooc, alpha=1.0, diag_mode='zero')
    sup = cooc_support(pred_base, C, mode='weighted_mean')

    row_sum = pred_base.sum(axis=1)
    row_max = pred_base.max(axis=1)
    row_top5 = -np.sort(-pred_base, axis=1)[:, :5].mean(axis=1)
    eps = 1e-9
    prob = pred_base / (pred_base.sum(axis=1, keepdims=True) + eps)
    entropy = -(prob * np.log(prob + eps)).sum(axis=1) / np.log(pred_base.shape[1])

    group_sums = np.zeros((pred_base.shape[0], len(group_names)), dtype=np.float32)
    group_maxs = np.zeros_like(group_sums)
    for gi in range(len(group_names)):
        cols = np.where(class_group_idx == gi)[0]
        group_sums[:, gi] = pred_base[:, cols].sum(axis=1)
        group_maxs[:, gi] = pred_base[:, cols].max(axis=1)
    return {
        'support': sup,
        'row_sum': row_sum.astype(np.float32),
        'row_max': row_max.astype(np.float32),
        'row_top5': row_top5.astype(np.float32),
        'entropy': entropy.astype(np.float32),
        'group_sums': group_sums,
        'group_maxs': group_maxs,
    }


def per_class_meta_matrix(meta, pred_base, class_j):
    gi = class_group_idx[class_j]
    return np.column_stack([
        pred_base[:, class_j],
        meta['support'][:, class_j],
        meta['row_sum'],
        meta['row_max'],
        meta['row_top5'],
        meta['entropy'],
        meta['group_sums'][:, gi],
        meta['group_maxs'][:, gi],
    ]).astype(np.float32)


def lofo_stacking(blend_w=0.25, min_pos=3, C=0.3):
    rows = []
    all_pred = {}
    fold_names = list(fold_data.keys())

    # Meta-features use only each fold's train labels for the cooccurrence support.
    meta_by_fold = {
        fold: build_meta_features(fd['pred_v30_global'], fd['y_train'])
        for fold, fd in fold_data.items()
    }

    for held_fold in fold_names:
        other_folds = [f for f in fold_names if f != held_fold]
        held_fd = fold_data[held_fold]
        pred_meta = held_fd['pred_v30_global'].copy()
        trained = 0

        for j in range(n_classes):
            x_parts = []
            y_parts = []
            for f in other_folds:
                fd = fold_data[f]
                x_parts.append(per_class_meta_matrix(meta_by_fold[f], fd['pred_v30_global'], j))
                y_parts.append(fd['y_val'][:, j])
            X = np.vstack(x_parts)
            y = np.concatenate(y_parts)
            n_pos = int(y.sum())
            n_neg = int(len(y) - n_pos)
            if n_pos < min_pos or n_neg < min_pos:
                continue
            model = LogisticRegression(
                C=C,
                class_weight='balanced',
                solver='liblinear',
                max_iter=500,
                random_state=42,
            )
            model.fit(X, y)
            Xh = per_class_meta_matrix(meta_by_fold[held_fold], held_fd['pred_v30_global'], j)
            pred_meta[:, j] = model.predict_proba(Xh)[:, 1].astype(np.float32)
            trained += 1

        pred_final = np.clip((1 - blend_w) * held_fd['pred_v30_global'] + blend_w * pred_meta, 0.0, 1.0)
        all_pred[held_fold] = pred_final
        rows.append({
            'fold': held_fold,
            'blend_w': blend_w,
            'min_pos': min_pos,
            'C': C,
            'trained_classes': trained,
            'auc': safe_macro_auc(held_fd['y_val'], pred_final),
            'ap': safe_macro_ap(held_fd['y_val'], pred_final),
        })
    return pd.DataFrame(rows), all_pred


stack_rows = []
stack_preds = {}
t0 = time.perf_counter()
for blend_w in [0.05, 0.10, 0.15, 0.20, 0.30, 0.40]:
    for min_pos in [3, 5, 8]:
        for C in [0.05, 0.1, 0.3, 1.0]:
            df_stack, preds = lofo_stacking(blend_w=blend_w, min_pos=min_pos, C=C)
            summary = {
                'blend_w': blend_w,
                'min_pos': min_pos,
                'C': C,
                'mean_auc': df_stack['auc'].mean(),
                'min_auc': df_stack['auc'].min(),
                'std_auc': df_stack['auc'].std(),
                'mean_ap': df_stack['ap'].mean(),
                'min_ap': df_stack['ap'].min(),
                'std_ap': df_stack['ap'].std(),
                'mean_trained_classes': df_stack['trained_classes'].mean(),
            }
            stack_rows.append(summary)
            stack_preds[(blend_w, min_pos, C)] = preds

stack_df = pd.DataFrame(stack_rows).sort_values(['mean_auc', 'min_auc', 'mean_ap'], ascending=False)
print('stacking elapsed min:', (time.perf_counter() - t0) / 60)
print('Stacking/calibration results')
display(stack_df.head(30))

# %%

baseline_rows = []
for name in ['pred_v27', 'pred_v30_global']:
    aucs, aps = [], []
    for fold, fd in fold_data.items():
        aucs.append(safe_macro_auc(fd['y_val'], fd[name]))
        aps.append(safe_macro_ap(fd['y_val'], fd[name]))
    baseline_rows.append({
        'method': name,
        'mean_auc': np.mean(aucs),
        'min_auc': np.min(aucs),
        'std_auc': np.std(aucs, ddof=1),
        'mean_ap': np.mean(aps),
        'min_ap': np.min(aps),
        'std_ap': np.std(aps, ddof=1),
    })

best_cooc = cooc_df.iloc[0].copy()
best_stack = stack_df.iloc[0].copy()

final_compare = pd.concat([
    pd.DataFrame(baseline_rows),
    pd.DataFrame([{
        'method': 'best_cooccurrence',
        'mean_auc': best_cooc['mean_auc'],
        'min_auc': best_cooc['min_auc'],
        'std_auc': best_cooc['std_auc'],
        'mean_ap': best_cooc['mean_ap'],
        'min_ap': best_cooc['min_ap'],
        'std_ap': best_cooc['std_ap'],
    }]),
    pd.DataFrame([{
        'method': 'best_stacking_lofo',
        'mean_auc': best_stack['mean_auc'],
        'min_auc': best_stack['min_auc'],
        'std_auc': best_stack['std_auc'],
        'mean_ap': best_stack['mean_ap'],
        'min_ap': best_stack['min_ap'],
        'std_ap': best_stack['std_ap'],
    }]),
], ignore_index=True).sort_values(['mean_auc', 'min_auc', 'mean_ap'], ascending=False)

print('Final comparison')
display(final_compare)

print('Best cooccurrence config')
display(best_cooc.to_frame().T)

print('Best stacking config')
display(best_stack.to_frame().T)

print('Decision rule:')
print('- If best_stacking improves mean AUC but hurts min AUC or is unstable, treat as overfit.')
print('- If cooccurrence improves both mean/min AUC by >0.003, it is a candidate for a safe Kaggle postprocess.')
print('- If neither improves, do not add multi-label postprocess; focus next on sub-window audio features/pseudo-labeling.')