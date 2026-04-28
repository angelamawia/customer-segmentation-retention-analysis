# customer-segmentation-retention-analysis

## Project Overview
This project performs end-to-end customer segmentation and retention analysis on a real-world online retail dataset. The goal is to understand customer behavior, identify at-risk customers, calculate customer lifetime value, and build a machine learning model to predict customer churn.

## Dataset

. Source: UCI Machine Learning Repository — Online Retail Dataset
. Size: 541,909 transactions
. Period: December 2010 — December 2011
. Description: Transactional data from a UK-based online retail store containing all purchases made between 2010 and 2011

## Project Goals

1. Customer Segmentation — Group customers based on their purchasing behavior
2. RFM Analysis — Score customers based on Recency, Frequency and Monetary value
3. Customer Lifetime Value (CLV) — Calculate the long term value of each customer
4. Churn Definition — Define and label churned vs active customers
5. Churn Prediction Model — Build a machine learning model to predict which customers are likely to churn

## Project Workflow

Raw Data
   ↓
Data Cleaning & Preprocessing
   ↓
Customer Level Feature Engineering
   ↓
RFM Analysis & Scoring
   ↓
CLV Calculation
   ↓
Churn Definition & Labeling
   ↓
Machine Learning Model (XGBoost)
   ↓
Model Evaluation & Interpretation

## Churn Definition
    A customer is defined as churned if they have not made a purchase in the last 90 days relative to the most recent  transaction date in the dataset.

    Churn Label = 1  if Recency > 90 days
    Churn Label = 0  if Recency ≤ 90 days

## RFM Segmentation
Customers are scored 1-4 on each RFM component and combined into a final RFM score ranging from 3 to 12:

RFM Score                    Customer Segment
10 — 12                      🏆 Champions
7 — 9                        ⭐ Loyal Customers
5 — 6                        ⚠️ At Risk
3 — 4                        ❌ Lost Customers

## Machine Learning Models
### Models Trained
Model                       Accuracy             Churned Recall
Random Forest (baseline)    67.5%                44%
Random Forest (balanced)    67.5%                44%
Random Forest (n=500)       67.7%                44%
XGBoost (final model)       69.7%                75%

## Final Model — XGBoost Classifier
    Accuracy:  69.7%

## Class          ## Precision        ## Recall          ## F1-Score
Active (0)         0.83                 0.67               0.74
Churned (1)        0.55                 0.75               0.64

## Confusion Matrix
    [[376  185]
    [ 78  229]]

## Key Result
The final XGBoost model successfully identifies 75% of churned customers — meaning the business can proactively reach out to 3 out of every 4 customers likely to churn before they are lost.

## Tools & Libraries
python
pandas        # data manipulation
numpy         # numerical computing
matplotlib    # data visualization
seaborn       # statistical visualization
sklearn       # machine learning & evaluation
xgboost       # gradient boosting model
jupyter       # development environment

## How to Run This Project
### 1. Clone the repository
    bashgit clone https://github.com/angelamawia/customer-segmentation-retention-analysis.git
### 2. Install required libraries
    bashpip install pandas numpy matplotlib seaborn scikit-learn xgboost jupyter
### 3. Download the dataset
    Download the Online Retail dataset from UCI Repository
    Place it in the project folder

### 4. Run the notebook
    bashjupyter notebook

    Open customer_segmentation_retention_analysis.ipynb
    Run all cells from top to bottom


## Key Business Insights
  . Customers with high RFM scores should be prioritized for loyalty rewards and retention campaigns
  . Customers with low recency and low frequency are at highest risk of churn
  . The bulk buyer segment (high spend, frequency of 1) requires a different retention strategy than regular customers
  . 75% of churning customers can be identified early enough for the business to intervene


## Project Status
    ✅ Complete

## Author
    Angela Mawia

    GitHub: angelamawia
    LinkedIn: [Your LinkedIn URL]

## Acknowledgements
Dataset sourced from the UCI Machine Learning Repository
Project completed as part of a customer analytics and data science project



