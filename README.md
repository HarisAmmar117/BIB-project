# BIB (BeforeItBreaks) 🔧

> Predicting industrial machine failure before it happens — a supervised learning project built on real sensor data.

## 📌 Overview

BIB is a machine learning system that predicts whether an industrial machine is likely to fail based on live sensor readings — air temperature, process temperature, rotational speed, torque, and tool wear. The goal is to move maintenance from *reactive* ("fix it after it breaks") to *predictive* ("flag it before it breaks"), which is one of the highest-value applications of ML in manufacturing.

The project is being built end-to-end: from a trained classification model, to a serving API, to a live risk dashboard.

## 🎯 Problem Statement

Unplanned machine downtime is expensive and disruptive. Rather than waiting for a failure to happen or servicing equipment on a fixed schedule regardless of actual condition, BIB uses historical sensor data to learn the patterns that precede a failure — and flags at-risk machines in advance.

Two prediction tasks:
1. **Binary classification** — will this machine fail? (Yes/No)
2. **Multi-label classification** — *which type* of failure is likely? (Tool Wear Failure, Heat Dissipation Failure, Power Failure, Overstrain Failure, Random Failure)

## 📊 Dataset

- **Source:** [AI4I 2020 Predictive Maintenance Dataset](https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020) (Kaggle)
- **Size:** 10,000 rows × 14 columns
- **Features:** Product quality type (L/M/H), air temperature, process temperature, rotational speed, torque, tool wear
- **Targets:** `Machine failure` (binary) + 5 failure-type flags (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`)
- **Note:** The dataset is realistically imbalanced — only ~3.4% of rows represent an actual failure. Evaluation uses precision, recall, F1, and PR-AUC rather than raw accuracy.

## 🧠 Approach

- **Baseline models:** Logistic Regression → Random Forest → XGBoost
- **Imbalance handling:** class weighting, with SMOTE as a comparison approach
- **Feature engineering:** derived features such as temperature differential (process − air temp) and torque-to-speed ratio
- **Evaluation:** stratified train/test split, precision/recall/F1/PR-AUC (not accuracy)
- **Extension:** multi-label classification for failure type once the binary model is solid

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Modeling | Python, pandas, scikit-learn, XGBoost, imbalanced-learn |
| Environment | Conda |
| API | FastAPI |
| Dashboard | React |
| Version control | Git + GitHub |

## 📁 Project Structure

```
bib/
├── data/               # Raw and processed datasets
├── notebooks/          # Exploration and modeling notebooks
├── src/                # Reusable Python modules (preprocessing, training, evaluation)
├── models/             # Saved trained model artifacts
├── api/                # FastAPI service exposing predictions
├── frontend/           # React dashboard
├── environment.yml     # Reproducible conda environment
└── README.md
```

## 🚀 Getting Started

**Prerequisites:** Anaconda/Miniconda, Git

```bash
# Clone the repo
git clone https://github.com/<your-username>/bib.git
cd bib

# Create and activate the environment
conda env create -f environment.yml
conda activate bib

# Launch the exploration notebook
jupyter notebook notebooks/01_exploration.ipynb
```

## 📈 Model Performance

_To be filled in as models are trained and evaluated._

| Model | Precision | Recall | F1-Score | PR-AUC |
|---|---|---|---|---|
| Logistic Regression | — | — | — | — |
| Random Forest | — | — | — | — |
| XGBoost | — | — | — | — |

## 🗺️ Roadmap

- [ ] Exploratory data analysis
- [ ] Baseline binary failure classifier
- [ ] Class imbalance handling & model comparison
- [ ] Multi-label failure-type classifier
- [ ] FastAPI prediction service
- [ ] React risk dashboard
- [ ] Deployment (API + frontend)

## 🙋 Author

Built by Ammar — BSc Information Technology undergraduate, SLIIT.

## 📄 License

MIT
