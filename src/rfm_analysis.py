import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def calculate_rfm(df, reference_date=None):
    """
    Calculate RFM metrics for each customer.
    
    Parameters:
    df (DataFrame): Preprocessed dataframe with customer purchase data
    reference_date (datetime, optional): Reference date for recency calculation
    
    Returns:
    DataFrame: DataFrame with RFM metrics for each customer
    """
    # If no reference date provided, use the day after the max date in the dataset
    if reference_date is None:
        reference_date = df['InvoiceDate'].max() + pd.DateOffset(days=1)
    
    # Group by customer ID
    rfm_data = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (reference_date - x.max()).days,  # Recency
        'InvoiceNo': 'nunique',                                    # Frequency
        'TotalPrice': 'sum'                                        # Monetary
    }).reset_index()
    
    # Rename columns
    rfm_data.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
    
    return rfm_data

def assign_rfm_scores(rfm_data, num_segments=5):
    """
    Assign RFM scores based on quartiles.
    
    Parameters:
    rfm_data (DataFrame): DataFrame with RFM metrics
    num_segments (int): Number of segments to create
    
    Returns:
    DataFrame: DataFrame with RFM scores
    """
    # Create copy of the data
    rfm_scores = rfm_data.copy()
    
    # Assign Recency score (lower recency is better)
    rfm_scores['R_Score'] = pd.qcut(rfm_scores['Recency'], num_segments, 
                                    labels=range(num_segments, 0, -1))
    
    # Assign Frequency score (higher frequency is better)
    rfm_scores['F_Score'] = pd.qcut(rfm_scores['Frequency'].rank(method='first'), 
                                    num_segments, labels=range(1, num_segments+1))
    
    # Assign Monetary score (higher monetary is better)
    rfm_scores['M_Score'] = pd.qcut(rfm_scores['Monetary'], num_segments, 
                                    labels=range(1, num_segments+1))
    
    # Combine RFM scores
    rfm_scores['RFM_Score'] = rfm_scores['R_Score'].astype(str) + \
                             rfm_scores['F_Score'].astype(str) + \
                             rfm_scores['M_Score'].astype(str)
    
    # Calculate RFM score as numeric
    rfm_scores['RFM_Numeric'] = rfm_scores['R_Score'].astype(int) * 100 + \
                               rfm_scores['F_Score'].astype(int) * 10 + \
                               rfm_scores['M_Score'].astype(int)
    
    return rfm_scores

def segment_customers(rfm_scores):
    """
    Segment customers based on RFM scores.
    
    Parameters:
    rfm_scores (DataFrame): DataFrame with RFM scores
    
    Returns:
    DataFrame: DataFrame with customer segments
    """
    # Define segmentation conditions
    segments = {
        'Champions': (rfm_scores['R_Score'] >= 4) & (rfm_scores['F_Score'] >= 4) & (rfm_scores['M_Score'] >= 4),
        'Loyal Customers': (rfm_scores['R_Score'] >= 3) & (rfm_scores['F_Score'] >= 3) & (rfm_scores['M_Score'] >= 3),
        'Potential Loyalists': (rfm_scores['R_Score'] >= 3) & (rfm_scores['F_Score'] >= 1) & (rfm_scores['M_Score'] >= 2),
        'New Customers': (rfm_scores['R_Score'] >= 4) & (rfm_scores['F_Score'] <= 2),
        'Promising': (rfm_scores['R_Score'] >= 3) & (rfm_scores['F_Score'] <= 2) & (rfm_scores['M_Score'] <= 2),
        'Needs Attention': (rfm_scores['R_Score'] >= 2) & (rfm_scores['R_Score'] < 4) & (rfm_scores['F_Score'] >= 2) & (rfm_scores['F_Score'] < 4) & (rfm_scores['M_Score'] >= 2) & (rfm_scores['M_Score'] < 4),
        'About to Sleep': (rfm_scores['R_Score'] >= 2) & (rfm_scores['R_Score'] < 4) & (rfm_scores['F_Score'] < 2),
        'At Risk': (rfm_scores['R_Score'] < 3) & (rfm_scores['F_Score'] >= 3) & (rfm_scores['M_Score'] >= 3),
        'Can\'t Lose Them': (rfm_scores['R_Score'] < 2) & (rfm_scores['F_Score'] >= 4) & (rfm_scores['M_Score'] >= 4),
        'Hibernating': (rfm_scores['R_Score'] < 2) & (rfm_scores['F_Score'] >= 2) & (rfm_scores['F_Score'] < 4),
        'Lost': (rfm_scores['R_Score'] < 2) & (rfm_scores['F_Score'] < 2)
    }
    
    # Create a new column for segments
    rfm_scores['Segment'] = 'Unknown'
    
    # Assign segments based on conditions
    for segment, condition in segments.items():
        rfm_scores.loc[condition, 'Segment'] = segment
    
    return rfm_scores
