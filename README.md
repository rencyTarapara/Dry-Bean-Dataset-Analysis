# Dry-Bean-Dataset-Analysis
# Dry Bean Dataset Classification Project

## Overview

This project focuses on the classification of seven different types of dry beans using various machine learning algorithms. The objective is to build accurate predictive models that identify bean types based on their morphological features.

The dataset contains physical measurements of dry beans, and multiple preprocessing and modeling techniques are applied to analyze and improve classification performance.

---

## Bean Classes

The dataset includes the following seven dry bean types:

* Barbunya
* Bombay
* Cali
* Dermason
* Horoz
* Sira
* Seker

---

## Keywords (SEO Optimized)

Dry Bean Dataset
Machine Learning Classification
Agricultural Data Analysis
Data Science Project
Predictive Modeling
Bean Type Prediction
SVM Classifier
Random Forest Classification
KNN Algorithm
Feature Engineering
PCA Analysis
Outlier Detection
Scikit-learn Python

---

## Dataset Description

The Dry Bean Dataset contains morphological features extracted from images of beans.

### Features include:

* Area
* Perimeter
* MajorAxisLength
* MinorAxisLength
* AspectRatio
* Eccentricity
* ConvexArea
* EquivDiameter
* Extent
* Solidity
* Roundness
* Compactness
* ShapeFactor1
* ShapeFactor2
* ShapeFactor3
* ShapeFactor4

Target Variable:

* Class (Type of Dry Bean)

---

## Methodology

### Data Loading and Exploration

* Loaded the dataset from an Excel file using pandas
* Checked for missing values and duplicate entries
* Generated descriptive statistics
* Visualized class distribution

---

### Exploratory Data Analysis

* Outlier Detection

  * Used box plots and the Interquartile Range (IQR) method
  * Removed extreme outliers to create a cleaned dataset
* Feature Distribution Comparison

  * Compared histograms before and after outlier removal
* Correlation Analysis

  * Heatmaps used to detect multicollinearity in raw and cleaned datasets

---

### Data Preprocessing and Feature Engineering

* Split data into training and testing sets
* Applied StandardScaler for feature normalization
* Used Principal Component Analysis (PCA)

  * Retained 95 percent variance
  * Reduced dimensionality and improved efficiency

---

### Model Training and Evaluation

The following models were trained and evaluated:

* Support Vector Machine (SVM) with RBF kernel
* Random Forest Classifier
* K-Nearest Neighbors (KNN)

Evaluation Metrics:

* Accuracy Score
* Confusion Matrix for cleaned dataset

Models were tested on:

* Raw dataset
* Cleaned and PCA-transformed dataset

---

## Results

### Model Accuracy Comparison

| Model         | Before Cleaning | After Cleaning |
| ------------- | --------------- | -------------- |
| SVM           | 63.72 percent   | 89.38 percent  |
| Random Forest | 92.03 percent   | 91.50 percent  |
| KNN           | 72.42 percent   | 88.67 percent  |

---

## Key Observations

* SVM and KNN showed significant performance improvement after cleaning and PCA
* Random Forest performed consistently well even without extensive preprocessing
* Highlights the importance of data preprocessing for distance-based and margin-based models

---

## Technologies and Libraries

* Python
* pandas
* numpy
* scikit-learn
* matplotlib
* seaborn

---

## How to Run the Project

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/dry-bean-classification.git
   ```

2. Install dependencies:

   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
   ```

3. Place the dataset file:

   ```
   Dry_Bean_Dataset.xlsx
   ```

4. Run the notebook:

   * Open in Google Colab or Jupyter Notebook
   * Execute cells sequentially

---

## Conclusion

This project demonstrates how data cleaning, feature scaling, and dimensionality reduction can significantly improve machine learning classification performance. It serves as a strong example of applying machine learning techniques to agricultural data analysis.

---

If you find this project useful, consider starring the repository. Contributions and suggestions are welcome.
