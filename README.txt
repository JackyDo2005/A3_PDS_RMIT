================================================================================
ASSIGNMENT 3: PRACTICAL DATA SCIENCE GROUP PROJECT
================================================================================

Course:        COSC2789: Practical Data Science
Assignment:    Assignment 3 - Group Project
Group:         UG2
Date:          January 15, 2026
Institution:   RMIT University, SSET

Dataset:       Online Shoppers Purchasing Intention Dataset
Problem Type:  Problem Type 1 - Data Modeling (Classification and Clustering)

================================================================================
TEAM MEMBERS
================================================================================

1. Do Duy Hung
   Student ID: s3991053
   Email:      s3991053@rmit.edu.vn

2. Ho Dinh Gia Bao
   Student ID: s4028938
   Email:      s4028938@rmit.edu.vn


================================================================================
INSTALLATION & SETUP
================================================================================

Step 1: Install Required Dependencies
--------------------------------------
Run the following command to install all required packages:

    pip install -r requirements.txt

Required Packages:
- pandas>=2.0.0
- numpy>=1.24.0
- scipy>=1.10.0
- matplotlib>=3.7.0
- seaborn>=0.12.0
- jupyter>=1.0.0
- ipykernel>=6.25.0
- scikit-learn>=1.3.0
- skfeature-chappers>=1.0.0
- xgboost>=2.0.0
- joblib>=1.3.0
- threadpoolctl>=3.0.0
- streamlit>=1.31.0
- plotly>=5.18.0

================================================================================
EXECUTION INSTRUCTIONS
================================================================================

Option 1: Run Jupyter Notebook (Recommended)
---------------------------------------------
1. Open terminal/command prompt
2. Navigate to the project directory
3. Launch Jupyter Notebook:
   
    jupyter notebook

4. Open "Assignment3.ipynb" in the browser
5. Run all cells sequentially (Cell > Run All)

Note: The notebook contains the complete data analysis pipeline including:
      - Data Preprocessing
      - Feature Engineering
      - Feature Selection
      - Classification Models (Logistic Regression, Random Forest, XGBoost, CRAE)
      - Clustering Models (K-Means, DBSCAN, Agglomerative Clustering)
      - Model Evaluation & Visualization

================================================================================
STREAMLIT DEPLOYMENT
================================================================================

Live Demo: https://ug2pdsa3rmit.streamlit.app/

The deployed application provides:
- Interactive prediction interface
- Real-time model inference
- Visualization of predictions
- Model performance metrics

================================================================================
PROJECT STRUCTURE
================================================================================

Task 1: Exploratory Data Analysis (EDA)
- Data loading and initial inspection
- Statistical analysis and distribution visualization
- Correlation analysis and feature relationships

Task 2: Data Preprocessing
- Outlier detection and treatment
- Feature engineering (log transformations, ratios, seasonal indicators)
- Feature scaling and encoding
- Feature selection (Filter-based, RFECV, MCFS)

Task 3: Data Modeling
- Classification: Logistic Regression, Random Forest, XGBoost, CRAE
- Clustering: K-Means, DBSCAN, Agglomerative Clustering
- Model evaluation and comparison
- Final model recommendation with justification

================================================================================
CONTRIBUTIONS (50% - 50% Split)
================================================================================

Do Duy Hung (s3991053) - 50% Contribution
-------------------------------------------
Task 1: Retrieving and Preparing the Data
- Data import and initial inspection
- Missing value detection and handling
- Outlier detection and handling
- Univariate analysis
- Bivariate analysis
- Correlation analysis

Task 2: Feature Engineering
- Feature Selection Method 1: Filter-Based (Variance + Correlation)
- Classification Feature Selection (Supervised Methods)
  - Mutual Information Ranking (Filter Method)
  - Recursive Feature Elimination with CV (Wrapper Method)
  - Final Feature Selection Decision for Classification

Task 3: Data Modeling
- Classification Task
  - Baseline Model 1: Logistic Regression
  - Baseline Model 2: Random Forest
  - Advanced Model: XGBoost
  - Innovative Model: CRAE (Confidence-Routed Adaptive Ensemble)
  - Model Comparison & Final Recommendation for Classification

Deliverables:
- Report (contribution)
- Presentation Slides (contribution)

Ho Dinh Gia Bao (s4028938) - 50% Contribution
----------------------------------------------
Task 1: Retrieving and Preparing the Data
- Univariate analysis
- Correlation analysis

Task 2: Feature Engineering
- Advanced Feature Engineering
  - Log Transformation for skewed features
  - Behavioral Feature Engineering (Total_Pages, Avg_Time_Per_Page, Product_Focus_Ratio)
  - Temporal Pattern Encoding (Is_Peak_Season)
  - Redundant Feature Removal
- Clustering Feature Selection (Unsupervised - MCFS)
  - Correlation Analysis of Selected Clustering Features
  - Clustering Feature Summary

Task 3: Data Modeling
- Clustering Task
  - Baseline Model 1: K-Means Clustering
  - Baseline Model 2: DBSCAN (Density-Based)
  - Advanced Model: Agglomerative Clustering (Hierarchical)
  - Cluster Profiling & Interpretation
  - Model Comparison & Final Recommendation for Clustering

Deliverables:
- Report (contribution)

================================================================================
END OF README
================================================================================
