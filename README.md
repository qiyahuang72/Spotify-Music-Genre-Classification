# 🎵 Music Genre Classification

> Predicting Spotify song genres from audio features using dimensionality reduction and machine learning — **best model achieved 88.3% weighted AUC**.

---

## Overview

This project builds a multi-class classification pipeline to predict the genre of a song (out of 10 possible genres) using audio features from a dataset of 50,000 Spotify songs. It covers the full ML workflow: data cleaning, feature engineering, dimensionality reduction, model training, and evaluation.

This was completed as a capstone project for an Introduction to Machine Learning course at NYU.

---

## Dataset

- **Source:** Spotify API (via course-provided dataset `musicData.csv`)
- **Size:** 50,000 songs × 18 columns
- **Target:** Music genre (10 classes: Electronic, Anime, Jazz, Alternative, Country, Rap, Blues, Rock, Classical, Hip-Hop)

**Audio features include:** popularity, acousticness, danceability, duration, energy, instrumentalness, key, liveness, loudness, mode, speechiness, tempo, valence

---

## Project Structure

```
music-genre-classification/
│
├── classification.ipynb     # Full analysis notebook (recommended entry point)
├── classification.py        # Python script version
├── ML_capstone.pdf          # Written report with visualizations
└── requirements.txt         # Dependencies
```

---

## Methodology

### 1. Data Preprocessing
| Challenge | Solution |
|---|---|
| Missing values in `tempo` and `duration_ms` | Median imputation |
| Non-normal distribution of `acousticness` | Box-Cox transformation |
| String-format `key` column | Label encoding |
| Categorical `mode` column | Dummy coding |
| Genre labels as strings | Categorical label encoding |
| Feature scale differences | StandardScaler on all continuous features |

Dropped irrelevant columns: `track_name`, `artist_name`, `instance_id`, `obtained_date`.  
**Final dataset: 50,000 samples × 13 features.**

### 2. Train/Test Split
- 500 randomly selected songs per genre → **5,000-song test set**
- Remaining 4,500 songs per genre → **45,000-song training set**
- Stratified by genre to ensure balanced evaluation

### 3. Dimensionality Reduction
Compared LDA, PCA (2D, 3D, 5D, 6D, 7D), and t-SNE:

| Method | Components | Variance Explained |
|---|---|---|
| **LDA** | **3** | **~92%** ✅ best |
| PCA | 6 | ~80% |
| t-SNE | 2 | — (non-linear) |

LDA was selected as the primary reduction method due to its superior variance explanation with the fewest components. The 3D LDA projection revealed significant overlap between genre clusters — indicating the inherent difficulty of the classification task.

### 4. Classification Models

| Model | Weighted AUC |
|---|---|
| **Feedforward Neural Network (ReLU)** | **88.3%** 🏆 |
| Random Forest | 87.3% |
| SVM | 83.7% |

**Random Forest config:** `n_estimators=1000`, `criterion='gini'`, `bootstrap=True`  
**Neural Network:** 2 hidden layers, ReLU activation — also tested sigmoid activation

---

## Results

The feedforward neural network with ReLU activation achieved the best performance at **88.3% weighted AUC**, evaluated via per-class ROC curves across all 10 genres.

Key finding: dimensionality reduction strategy had the largest impact on performance — LDA significantly outperformed PCA and t-SNE across all models. Hyperparameter tuning (especially epoch count for the neural network) was also critical.

---

## Interesting Observations

- **Pop, Rock, and Hip-Hop** tend to score higher in popularity; **Anime, Blues, and Classical** tend to score lower
- **Jazz** shows the widest spread in popularity — some tracks are widely known, others are very obscure
- Songs with **both high danceability and high energy** tend to be more popular, but popularity plateaus once both features exceed a certain threshold

---

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run the script
python classification.py

# Or open the notebook
jupyter notebook classification.ipynb
```

> ⚠️ **Note:** The original code uses a student N-number as the random seed. Replace `random.seed(YOUR_N_NUMBER)` with any integer seed of your choice to reproduce results.

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![pandas](https://img.shields.io/badge/pandas-1.5+-green)

- **ML:** scikit-learn, PyTorch
- **Data:** pandas, numpy, scipy
- **Visualization:** matplotlib, seaborn
- **Preprocessing:** imbalanced-learn

---

## Report

See [`ML_capstone.pdf`](./ML_capstone.pdf) for the full written report including ROC curve plots, the 3D LDA visualization, and discussion of design choices.
