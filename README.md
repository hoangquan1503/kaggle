# Internet Advertisements Classification Analysis

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Latest-F7931E.svg)](https://scikit-learn.org/)

## Overview

This repository contains a comprehensive analysis of the **Internet Advertisements** dataset from the UCI Machine Learning Repository. The goal is to classify web page elements as either advertisements (`ad`) or non‑advertisements (`non-ad`) using features such as image dimensions, URL keywords, anchor text, alt text, and captions.

The notebook covers:
- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- Statistical hypothesis testing
- Two classification models (Lasso Logistic Regression and Random Forest)
- Performance comparison with and without outlier removal

## Dataset

- **Source**: UCI Machine Learning Repository – [Internet Advertisements Data Set](https://archive.ics.uci.edu/ml/datasets/Internet+Advertisements)
- **Original samples**: 3,279  
- **After removing duplicates**: 2,422
- **Features**: 1,558 (including `height`, `width`, `local`, binary features for URLs, anchor text, alt text, captions)
- **Target**: `Class` (0 = non‑ad, 1 = ad) – imbalanced (84% non‑ad, 16% ad)

## Key Steps

1. **Preprocessing**  
   - Column naming based on dataset documentation  
   - Convert `?` to `NaN`, trim whitespace, coerce to numeric  
   - Drop `ratio` (redundant)  
   - Impute missing values: `height`/`width` with median, `local` with 1  
   - Remove duplicate rows

2. **Exploratory Data Analysis**  
   - Distributions of `height`, `width`, `local`  
   - Class distribution, boxplots by class, proportion of `local` by class  
   - Top‑10 binary features with largest mean differences between classes  
   - Correlation heatmap

3. **Statistical Tests**  
   - Welch’s t‑test and Mann‑Whitney U test for `width` across classes → significant difference  
   - Two‑proportion z‑test for `local` → ads have significantly higher proportion of internal (`local=1`) images  
   - Shapiro‑Wilk test shows `width` is not normally distributed

4. **Modeling**  
   - Train/test split (70/30), stratified by class  
   - **Lasso Logistic Regression** (`LogisticRegressionCV` with L1 penalty) using sample weights to handle class imbalance  
   - **Random Forest** (500 trees, `max_features='sqrt'`)  
   - Evaluation: confusion matrix, sensitivity, specificity, ROC‑AUC  
   - Repeat models after removing outliers (IQR method for `height` and `width`)

## Results

| Model                     | ROC‑AUC |
|---------------------------|---------|
| Lasso – original data     | 0.9606  |
| Random Forest – original  | 0.9628  |
| Lasso – outliers removed  | 0.9103  |
| Random Forest – outliers removed | 0.9546  |

**Observations**  
- Random Forest consistently outperforms Lasso on both datasets.  
- Outlier removal dramatically reduces Lasso’s performance, but only slightly affects Random Forest.  
- Both models achieve high accuracy (>94%) and very high specificity (97‑99%).

## Requirements

Install the required packages using:

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn statsmodels
```
## How to Run
1. Clone the repository or download the notebook.
2. Open internet_advertisements_analysis.ipynb with Jupyter Notebook / JupyterLab / Google Colab.
3. Update the data path in the Cell 1 – set DATA_PATH to the location of ad.data or add.csv on your system.
4. Run all cells sequentially.
