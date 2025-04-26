# Customer Segmentation using RFM Analysis

This project performs customer segmentation using the RFM (Recency, Frequency, Monetary) analysis method on an eCommerce dataset.

## Project Overview

RFM analysis is a powerful technique used by businesses to group customers based on their recent purchasing behavior, purchase frequency, and monetary value. This segmentation enables more targeted marketing and customer engagement strategies.

### Key Features

- Data preprocessing and cleaning
- RFM metric calculation
- Customer segmentation based on RFM scores
- K-means clustering of customers
- Visualization of results and segments
- Marketing recommendations for each segment

## Project Structure

- `src/`: Source code modules
  - `data_processing.py`: Data cleaning and preprocessing
  - `rfm_analysis.py`: RFM calculation and segmentation
  - `clustering.py`: Clustering algorithms
  - `visualization.py`: Visualization functions
  - `utils.py`: Utility functions
- `notebooks/`: Jupyter notebooks
- `data/`: Data files
  - `raw/`: Original dataset
  - `processed/`: Cleaned and processed data
- `results/`: Output visualizations and findings
- `main.py`: Main script to run analysis

## Setup and Installation

1. Clone this repository

2. Install required packages:
 - pip install -r requirements.txt

3. Download the dataset:
- Get the eCommerce dataset from [Kaggle](https://www.kaggle.com/datasets/carrie1/ecommerce-data)
- Place the dataset in the `data/raw/` directory as `ecommerce_data.csv`

4. Run the analysis