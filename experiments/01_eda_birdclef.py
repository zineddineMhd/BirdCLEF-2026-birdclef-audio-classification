# BirdCLEF+ 2026 experiment export
#
# Clean text export of the original Jupyter/Colab experiment.
# Cell boundaries are preserved with VS Code/Jupytext-style markers.
# Notebook outputs are intentionally omitted; verified metrics are documented in docs/RESULTS.md.

# %% [markdown]
# # BirdCLEF 2026 - EDA notebook
#
# This notebook is designed to understand the dataset before modeling:
# - inspect the folder structure and CSV schemas
# - measure label imbalance and metadata quality
# - verify that metadata matches the audio files on disk
# - visualize waveform and mel spectrogram examples
# - inspect soundscape labels and the submission format
# - turn the EDA into concrete modeling decisions

# %%
from pathlib import Path
import ast
import re
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from IPython.display import Audio, display

try:
    import librosa
    import librosa.display
    import soundfile as sf
except ImportError:
    get_ipython().run_line_magic('pip', 'install librosa soundfile seaborn -q')
    import librosa
    import librosa.display
    import soundfile as sf

sns.set_theme(style='whitegrid', context='talk')
pd.set_option('display.max_columns', 120)
pd.set_option('display.max_colwidth', 120)

IN_COLAB = 'google.colab' in sys.modules

if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')

project_candidates = [
    Path('/content/drive/MyDrive/projet ML'),
    Path('/content/drive/MyDrive/Colab Notebooks/projet ML'),
    Path(r'G:/Mon Drive/projet ML'),
    Path.cwd(),
]

PROJECT_DIR = next((p for p in project_candidates if p.exists()), project_candidates[0])
DATA_DIR = PROJECT_DIR / 'birdclef-2026'
TRAIN_AUDIO_DIR = DATA_DIR / 'train_audio'
TRAIN_SOUNDSCAPES_DIR = DATA_DIR / 'train_soundscapes'
TEST_SOUNDSCAPES_DIR = DATA_DIR / 'test_soundscapes'

assert DATA_DIR.exists(), (
    f'Dataset folder not found: {DATA_DIR}\n'
    "If you are on Colab, place the dataset under /content/drive/MyDrive/projet ML/birdclef-2026"
)
print('PROJECT_DIR =', PROJECT_DIR)
print('DATA_DIR =', DATA_DIR)

# %%
def parse_list_string(value):
    if pd.isna(value):
        return []
    if isinstance(value, list):
        return value
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
        return parsed if isinstance(parsed, list) else [parsed]
    except Exception:
        return [x.strip() for x in text.split(';') if x.strip()]

def parse_semicolon_labels(value):
    if pd.isna(value):
        return []
    return [x.strip() for x in str(value).split(';') if x.strip()]

def time_to_seconds(value):
    hh, mm, ss = map(int, str(value).split(':'))
    return hh * 3600 + mm * 60 + ss

def add_duration_column(file_df, file_col='path'):
    durations = []
    for path in file_df[file_col]:
        try:
            durations.append(sf.info(path).duration)
        except Exception:
            durations.append(np.nan)
    out = file_df.copy()
    out['duration_sec'] = durations
    return out

def show_audio_example(path, offset_sec=0.0, duration_sec=10.0, sr=32000):
    y, sr = librosa.load(path, sr=sr, mono=True, offset=offset_sec, duration=duration_sec)
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), constrained_layout=True)
    librosa.display.waveshow(y, sr=sr, ax=axes[0], color='teal')
    axes[0].set_title(f'Waveform: {Path(path).name}')

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=sr // 2)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    img = librosa.display.specshow(mel_db, x_axis='time', y_axis='mel', sr=sr, ax=axes[1], cmap='magma')
    axes[1].set_title('Mel spectrogram (dB)')
    fig.colorbar(img, ax=axes[1], format='%+2.0f dB')
    plt.show()
    display(Audio(y, rate=sr))

# %% [markdown]
# ## 0. File-system overview
#
# This section answers the first practical questions:
# - what files are available?
# - how many audio folders and audio files are there?
# - are the metadata tables aligned with the files on disk?

# %%
overview = []
for path in sorted(DATA_DIR.iterdir()):
    overview.append({
        'name': path.name,
        'kind': 'dir' if path.is_dir() else 'file',
        'size_mb': round(path.stat().st_size / 1024**2, 3) if path.is_file() else np.nan,
    })

overview_df = pd.DataFrame(overview)
display(overview_df)

audio_files = list(TRAIN_AUDIO_DIR.glob('*/*.ogg'))
soundscape_files = list(TRAIN_SOUNDSCAPES_DIR.glob('*.ogg'))
test_files = list(TEST_SOUNDSCAPES_DIR.glob('*'))

print('train_audio folders :', sum(p.is_dir() for p in TRAIN_AUDIO_DIR.iterdir()))
print('train_audio files   :', len(audio_files))
print('train_soundscapes   :', len(soundscape_files))
print('test_soundscapes    :', len(test_files))

# %%
train = pd.read_csv(DATA_DIR / 'train.csv')
taxonomy = pd.read_csv(DATA_DIR / 'taxonomy.csv')
sample_submission = pd.read_csv(DATA_DIR / 'sample_submission.csv')
train_soundscapes_labels = pd.read_csv(DATA_DIR / 'train_soundscapes_labels.csv')

print('train shape                 :', train.shape)
print('taxonomy shape              :', taxonomy.shape)
print('sample_submission shape     :', sample_submission.shape)
print('train_soundscapes_labels    :', train_soundscapes_labels.shape)

# %% [markdown]
# ## 1. Understand the metadata tables
#
# This is where we verify what the labels really mean, how many classes we have, and whether the task is strongly imbalanced.

# %%
for name, df in {
    'train': train,
    'taxonomy': taxonomy,
    'sample_submission': sample_submission,
    'train_soundscapes_labels': train_soundscapes_labels,
}.items():
    print(f'\n=== {name} ===')
    print('shape   :', df.shape)
    print('columns :', list(df.columns))
    display(df.head(3))

# %%
train = train.copy()
train['secondary_labels_list'] = train['secondary_labels'].apply(parse_list_string)
train['type_list'] = train['type'].apply(parse_list_string)
train['n_secondary_labels'] = train['secondary_labels_list'].apply(len)
train['n_types'] = train['type_list'].apply(len)

missing_pct = (train.isna().mean() * 100).sort_values(ascending=False).rename('missing_pct')
display(missing_pct.to_frame().head(15))

print('Unique primary labels :', train['primary_label'].nunique())
print('Unique scientific names:', train['scientific_name'].nunique())
print('Unique collections    :', train['collection'].nunique())
print('Rows with secondary labels > 0 :', (train['n_secondary_labels'] > 0).mean().round(4))
print('Rows with type tags > 0        :', (train['n_types'] > 0).mean().round(4))

# %%
label_counts = train['primary_label'].value_counts().rename_axis('primary_label').reset_index(name='n_clips')
label_counts = label_counts.merge(taxonomy[['primary_label', 'common_name', 'scientific_name', 'class_name']], on='primary_label', how='left')

display(label_counts.head(10))
display(label_counts.tail(10))

fig, axes = plt.subplots(1, 2, figsize=(18, 6), constrained_layout=True)
sns.barplot(data=label_counts.head(20), x='n_clips', y='primary_label', ax=axes[0], palette='viridis')
axes[0].set_title('Top 20 labels by number of clips')
axes[0].set_xlabel('Number of clips')
axes[0].set_ylabel('Primary label')

sns.histplot(label_counts['n_clips'], bins=40, kde=True, ax=axes[1], color='coral')
axes[1].set_title('Distribution of clips per label')
axes[1].set_xlabel('Clips per label')
plt.show()

print('Median clips per label :', label_counts['n_clips'].median())
print('Mean clips per label   :', round(label_counts['n_clips'].mean(), 2))
print('Labels with <= 5 clips :', int((label_counts['n_clips'] <= 5).sum()))

# %%
class_counts = taxonomy['class_name'].value_counts().rename_axis('class_name').reset_index(name='n_labels')
display(class_counts)

plt.figure(figsize=(10, 5))
sns.barplot(data=class_counts, x='class_name', y='n_labels', palette='deep')
plt.title('Taxonomy mix by biological class')
plt.xlabel('Class')
plt.ylabel('Number of labels')
plt.xticks(rotation=30, ha='right')
plt.show()

# %% [markdown]
# ## 2. Verify audio files on disk
#
# The goal here is to catch data issues early: missing files, wrong folders, duplicates, abnormal file sizes, and likely duration outliers.

# %%
audio_inventory = pd.DataFrame({
    'path': audio_files,
})
audio_inventory['path'] = audio_inventory['path'].astype(str)
audio_inventory['filename'] = audio_inventory['path'].map(lambda x: Path(x).relative_to(TRAIN_AUDIO_DIR).as_posix())
audio_inventory['folder_label'] = audio_inventory['filename'].str.split('/').str[0]
audio_inventory['audio_file'] = audio_inventory['filename'].str.split('/').str[1]
audio_inventory['size_mb'] = audio_inventory['path'].map(lambda x: Path(x).stat().st_size / 1024**2)

train['file_exists'] = train['filename'].apply(lambda x: (TRAIN_AUDIO_DIR / x).exists())
print('Missing audio files referenced by train.csv :', int((~train['file_exists']).sum()))

merged_inventory = audio_inventory.merge(
    train[['filename', 'primary_label', 'scientific_name', 'common_name', 'rating', 'collection']],
    on='filename',
    how='left'
)
display(merged_inventory.head())

print('Audio files not matched in train.csv :', int(merged_inventory['primary_label'].isna().sum()))

# %%
sample_audio_stats = merged_inventory.sample(min(300, len(merged_inventory)), random_state=42).copy()
sample_audio_stats = add_duration_column(sample_audio_stats, file_col='path')

fig, axes = plt.subplots(1, 3, figsize=(20, 5), constrained_layout=True)
sns.histplot(merged_inventory['size_mb'], bins=40, ax=axes[0], color='steelblue')
axes[0].set_title('Audio file size distribution')
axes[0].set_xlabel('Size (MB)')

sns.histplot(sample_audio_stats['duration_sec'].dropna(), bins=40, ax=axes[1], color='olive')
axes[1].set_title('Sampled audio duration distribution')
axes[1].set_xlabel('Duration (seconds)')

top_label_sizes = merged_inventory.groupby('folder_label')['size_mb'].mean().sort_values(ascending=False).head(20)
sns.barplot(x=top_label_sizes.values, y=top_label_sizes.index, ax=axes[2], palette='crest')
axes[2].set_title('Top 20 labels by mean file size')
axes[2].set_xlabel('Mean size (MB)')
axes[2].set_ylabel('Folder label')
plt.show()

display(sample_audio_stats[['filename', 'duration_sec', 'size_mb']].sort_values('duration_sec', ascending=False).head(10))

# %%
example_row = train.sample(1, random_state=42).iloc[0]
example_path = TRAIN_AUDIO_DIR / example_row['filename']

print('Example label      :', example_row['primary_label'])
print('Scientific name    :', example_row['scientific_name'])
print('Common name        :', example_row['common_name'])
print('Collection         :', example_row['collection'])
print('Rating             :', example_row['rating'])
print('Audio path         :', example_path)

show_audio_example(example_path, duration_sec=10.0)

# %% [markdown]
# ## 3. Geographic and source metadata
#
# This block helps assess domain shift risk: different collections, coordinates, and recording conditions may bias the model.

# %%
geo = train[['primary_label', 'scientific_name', 'class_name', 'latitude', 'longitude', 'collection', 'rating']].copy()
geo['latitude'] = pd.to_numeric(geo['latitude'], errors='coerce')
geo['longitude'] = pd.to_numeric(geo['longitude'], errors='coerce')
geo['rating'] = pd.to_numeric(geo['rating'], errors='coerce')

fig, axes = plt.subplots(1, 2, figsize=(18, 6), constrained_layout=True)
sns.scatterplot(
    data=geo.sample(min(5000, len(geo)), random_state=42),
    x='longitude', y='latitude', hue='class_name', alpha=0.6, s=40, ax=axes[0]
)
axes[0].set_title('Geographic spread of training clips (sample)')

collection_counts = train['collection'].value_counts().rename_axis('collection').reset_index(name='n_clips')
sns.barplot(data=collection_counts, x='n_clips', y='collection', ax=axes[1], palette='flare')
axes[1].set_title('Clips by source collection')
axes[1].set_xlabel('Number of clips')
axes[1].set_ylabel('Collection')
plt.show()

display(train.groupby('collection')['rating'].agg(['count', 'mean', 'median']).sort_values('count', ascending=False))

# %% [markdown]
# ## 4. Soundscapes and multi-label targets
#
# This section is critical because the competition prediction target is closer to soundscape detection than to isolated-clip classification.

# %%
ssl = train_soundscapes_labels.copy()
ssl['label_list'] = ssl['primary_label'].apply(parse_semicolon_labels)
ssl['n_species'] = ssl['label_list'].apply(len)
ssl['start_sec'] = ssl['start'].apply(time_to_seconds)
ssl['end_sec'] = ssl['end'].apply(time_to_seconds)
ssl['segment_duration_sec'] = ssl['end_sec'] - ssl['start_sec']

display(ssl.head())
print('Unique soundscape files:', ssl['filename'].nunique())
print('Mean species per segment:', round(ssl['n_species'].mean(), 2))
print('Median species per segment:', round(ssl['n_species'].median(), 2))
print('Unique segment durations:', sorted(ssl['segment_duration_sec'].dropna().unique().tolist())[:10])

# %%
exploded_ssl = ssl[['filename', 'start', 'end', 'label_list']].explode('label_list').rename(columns={'label_list': 'primary_label'})
exploded_ssl = exploded_ssl.merge(taxonomy[['primary_label', 'common_name', 'scientific_name', 'class_name']], on='primary_label', how='left')

fig, axes = plt.subplots(1, 2, figsize=(18, 6), constrained_layout=True)
sns.histplot(ssl['n_species'], bins=20, ax=axes[0], color='purple')
axes[0].set_title('Number of species per soundscape segment')
axes[0].set_xlabel('Species count in 5-second segment')

top_soundscape_labels = exploded_ssl['primary_label'].value_counts().head(20)
sns.barplot(x=top_soundscape_labels.values, y=top_soundscape_labels.index, ax=axes[1], palette='rocket')
axes[1].set_title('Top 20 species in soundscape labels')
axes[1].set_xlabel('Number of labeled segments')
axes[1].set_ylabel('Primary label')
plt.show()

display(exploded_ssl.head(10))

# %%
example_soundscape_name = ssl['filename'].iloc[0]
example_soundscape_path = TRAIN_SOUNDSCAPES_DIR / example_soundscape_name
print('Example soundscape:', example_soundscape_path)
display(ssl[ssl['filename'] == example_soundscape_name].head(10))
show_audio_example(example_soundscape_path, duration_sec=15.0)

# %% [markdown]
# ## 5. Understand the submission format
#
# We need to confirm the model output shape and how row identifiers are constructed.

# %%
submission_label_cols = [c for c in sample_submission.columns if c != 'row_id']
taxonomy_labels = set(taxonomy['primary_label'].astype(str))
submission_labels = set(map(str, submission_label_cols))

print('Submission rows          :', len(sample_submission))
print('Submission label columns :', len(submission_label_cols))
print('Labels in taxonomy but not submission:', len(taxonomy_labels - submission_labels))
print('Labels in submission but not taxonomy:', len(submission_labels - taxonomy_labels))

row_id_parts = sample_submission['row_id'].str.extract(
    r'(?P<prefix>.+)_(?P<segment_start>\d+)$'
)
display(pd.concat([sample_submission[['row_id']].head(5), row_id_parts.head(5)], axis=1))

display(sample_submission.iloc[:3, :10])

# %% [markdown]
# ## 6. Practical conclusions for modeling
#
# What this EDA should help you decide:
#
# 1. Use isolated clips (`train_audio`) for supervised pretraining, but do not assume the final task is single-label.
# 2. Treat `train_soundscapes_labels.csv` as the bridge toward the competition target: segment-level multi-label detection.
# 3. Expect strong class imbalance, so plan weighted losses, careful validation splits, and robust metrics.
# 4. Validate file integrity before training: missing files, broken audio, abnormal durations, and low-quality labels.
# 5. Build a first baseline around mel spectrograms + CNN/efficient audio backbone, then move to soundscape inference.
#
# Suggested next notebook after EDA:
# - audio preprocessing and resampling checks
# - label encoding for both clip classification and soundscape detection
# - stratified validation split by label and, if possible, by source / location
# - baseline model training
# - inference pipeline that maps 5-second windows to submission rows