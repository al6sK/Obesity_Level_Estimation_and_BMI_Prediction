# 📊 Obesity Level Estimation & BMI Prediction

This repository contains a comprehensive Data Analytics and Machine Learning project aimed at analyzing and predicting obesity levels and Body Mass Index (BMI) based on individuals' eating habits and physical conditions. 

This project was developed as part of a "Data Analytics" university course.

## 📑 Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
  - [1. Data Preprocessing & EDA](#1-data-preprocessing--eda)
  - [2. Clustering](#2-clustering)
  - [3. Classification & Regression](#3-classification--regression)
- [Results & Conclusion](#results--conclusion)
- [Technologies Used](#technologies-used)
- [How to Run](#how-to-run)

## 🔍 Project Overview
The goal of this project is to apply various machine learning techniques to understand how dietary habits and physical activity correlate with obesity. The project is divided into three main phases:
1. **Data Preprocessing & Exploratory Data Analysis (EDA)**
2. **Unsupervised Learning (Clustering)** to find hidden patterns.
3. **Supervised Learning (Classification & Regression)** to predict the specific obesity category and the exact BMI value.

## 📁 Dataset
The project uses the **"Estimation of Obesity Levels Based On Eating Habits and Physical Condition"** dataset. 
- **Size:** 2,111 records.
- **Features:** Attributes related to eating habits (e.g., frequency of high-caloric food, water intake), physical condition (e.g., physical activity frequency), and physical characteristics (Age, Height, Weight).

## ⚙️ Project Workflow

### 1. Data Preprocessing & EDA
- **Data Cleaning:** Checked for missing (null) or erroneous values.
- **Visualizations:** Generated distribution plots (Histograms with KDE), density plots, and box plots (using log scaling) to understand feature distributions and detect outliers.
- **Normalization:** Applied `MinMaxScaler` to numerical features to scale them between 0 and 1.
- **Encoding:** Converted binary categorical variables (e.g., Gender, Smoke) to 1/0 and applied `LabelEncoder` to multi-class categorical variables.
- **Correlation Analysis:** Generated a Pearson correlation heatmap to check for multicollinearity. No features were dropped as correlations were not high enough to justify removal.

### 2. Clustering
Focused on features strictly related to food and drink consumption. 
- **PCA:** Applied Principal Component Analysis (PCA) retaining 95% of the variance to reduce dimensionality.
- **K-Means:** Clustered the data using $k$ equal to the number of unique obesity classes.
- **DBSCAN:** Utilized `NearestNeighbors` and the `KneeLocator` algorithm to dynamically find the optimal `eps` (maximum distance) for density-based clustering.

### 3. Classification & Regression
The dataset was split into **Train (70%)**, **Validation/Dev (20%)**, and **Test (10%)** sets using stratified sampling to maintain class proportions.

* **Classification (Target: Obesity Category - `NObeyesdad`)**
  * **Random Forest Classifier:** Trained with 500 estimators and entropy criterion.
  * **Neural Network (Custom Keras Model):** A 3-layer Dense network (128-16-7) utilizing ReLU and Softmax activations, trained with Early Stopping to prevent overfitting.

* **Regression (Target: BMI)**
  * Calculated the BMI mathematically ($Weight / Height^2$) and added it as a new feature.
  * **Neural Network (Custom Keras Model):** A 4-layer Dense network (64-32-8-1) utilizing ReLU and Linear activations, optimized with Adam and Mean Squared Error (MSE).

## 📈 Results & Conclusion

* **Clustering Comparison:** **DBSCAN** outperformed K-Means. DBSCAN achieved a Silhouette Score of **0.445** compared to K-Means' **0.358** (a ~24.3% improvement). DBSCAN was also significantly faster computationally.
* **Classification Performance:** Both models performed exceptionally well in predicting the obesity category. 
  * **Random Forest** slightly outperformed the Neural Network, achieving a Weighted F1-Score of **0.945**.
  * **Neural Network** achieved a Weighted F1-Score of **0.934**.
* **Regression Performance:** The Neural Network successfully predicted exact BMI values with a very low error rate. The Mean Absolute Percentage Error (**MAPE**) on the test set was just **3.82%**, proving the model's high reliability.

## 💻 Technologies Used
- **Language:** Python 3.x
- **Data Manipulation:** `pandas`, `numpy`
- **Machine Learning:** `scikit-learn`, `tensorflow`, `keras`
- **Data Visualization:** `matplotlib`, `seaborn`, `missingno`
- **Utilities:** `kneed` (for Knee point detection in DBSCAN)

## 🚀 How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/obesity-level-estimation.git
   cd obesity-level-estimation
   ```

2. **Directory Structure:**
   Ensure your project matches the following structure so the script can locate the data and save the plots properly:
   ```text
   project/
   ├── code/
   │   ├── ergasia.py
   │   └── resources.txt
   ├── data/
   │   └── ObesityDataSet_raw_and_data_sinthetic.csv
   ├── plots/
   │   ├── step1/
   │   ├── step2/
   │   └── step3/
   ├── p22062.pdf
   └── README.md
   ```

3. **Install dependencies:**
   You can easily install all required libraries using the provided `resources.txt` file:
   ```bash
   pip install -r code/resources.txt
   ```

4. **Run the script:**
   From the root `project` directory, run the Python file located in the `code` folder:
   ```bash
   python code/ergasia.py
   ```
   *(**Note:** Make sure the file paths inside your `ergasia.py` script have been updated to point to `"data/ObesityDataSet..."` and `"plots/step..."` instead of the old `"ergasia/..."` paths).*
