# Machine Learning & Data Visualization Lab

**Submitted by:** Aruni Mishra  
**Course:** Machine Learning & Data Visualization Lab

This repository contains nine machine learning experiments and eight data visualization experiments. Each Python script is self-contained and uses built-in or clearly defined sample data; none requires downloading a dataset.

## Repository structure

```text
.
├── ML/
│   ├── Exp1_Simple_Linear_Regression.py
│   ├── Exp2_Multiple_Linear_Regression.py
│   ├── Exp3_Logistic_Regression.py
│   ├── Exp4_Regression_with_Regularization.py
│   ├── Exp5_Dimensionality_Reduction_PCA.py
│   ├── Exp6_SVM.py
│   ├── Exp7_Decision_Tree.py
│   ├── Exp8_Naive_Bayes.py
│   └── Exp9_KMeans_Clustering.py
└── DV/
    ├── Exp1_Basic_Charts.py
    ├── Exp2_Data_Cleaning_Transformation.py
    ├── Exp3_Bar_and_Line_Charts.py
    ├── Exp4_Interactive_Dashboard_Filters.py
    ├── Exp5_Geo_Visualization.py
    ├── Exp6_Pie_and_Donut_Charts.py
    ├── Exp7_Heatmaps_and_Treemaps.py
    └── Exp8_Time_Series_Forecasting.py
```

## Machine Learning experiments

1. **Simple Linear Regression** — Predict diabetes disease progression from the BMI feature.
2. **Multiple Linear Regression** — Predict diabetes progression using all dataset features.
3. **Logistic Regression** — Classify benign and malignant tumors with scaled breast cancer data.
4. **Regression with Regularization** — Compare linear, Ridge, and Lasso regression.
5. **Dimensionality Reduction (PCA)** — Reduce standardized Iris measurements to two components.
6. **Support Vector Machine** — Classify breast cancer tumors with a scaled RBF-kernel SVM.
7. **Decision Tree** — Classify Iris species and print the learned tree.
8. **Naive Bayes** — Classify wine cultivars using Gaussian Naive Bayes.
9. **K-means Clustering** — Cluster Iris measurements and report unsupervised metrics.

## Data Visualization experiments

1. **Basic Charts** — Create a scatter plot, histogram, and box plot.
2. **Data Cleaning and Transformation** — Remove duplicates, handle missing values, derive revenue, and summarize categories.
3. **Bar and Line Charts** — Compare and chart monthly product sales.
4. **Interactive Dashboard Filters** — Filter a sample sales dashboard by product and region using command-line options.
5. **Geographical Visualization** — Plot sample US city coordinates with population-scaled markers.
6. **Pie and Donut Charts** — Compare monthly budget category shares.
7. **Heatmaps and Treemaps** — Explore metric correlations and relative category revenue.
8. **Time Series Forecasting** — Evaluate a seasonal trend model and forecast six months of sales.

## Requirements and running

Use Python 3.9 or newer. Install the dependencies with:

```bash
python -m pip install numpy pandas matplotlib seaborn scikit-learn
```

Run any experiment from the repository root, for example:

```bash
python ML/Exp1_Simple_Linear_Regression.py
python DV/Exp1_Basic_Charts.py
python DV/Exp4_Interactive_Dashboard_Filters.py --region North --product Tea
```

The visualization scripts open their charts in a Matplotlib window. The dashboard accepts `--region` (`all`, `North`, `South`, or `West`) and `--product` (`all`, `Tea`, `Coffee`, or `Juice`) filters.
