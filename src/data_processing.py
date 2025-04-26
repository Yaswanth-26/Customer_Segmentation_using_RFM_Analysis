import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def load_data(file_path):
    """Load the eCommerce dataset."""
    return pd.read_csv(file_path, encoding='ISO-8859-1')

def clean_data(df):
    """Clean and preprocess the dataset."""
    # Handle missing values
    print(f"Initial shape: {df.shape}")
    
    # Drop rows with missing customer IDs
    df = df.dropna(subset=['CustomerID', 'Description'])
    
    # Handle duplicates
    df = df.drop_duplicates()
    
    # Convert data types
    df['CustomerID'] = df['CustomerID'].astype('int64')
    df['InvoiceDate'] = pd.to_datetime(df.InvoiceDate)
    
    # Filter out cancelled transactions
    df_positive = df[df['Quantity'] > 0]
    
    # Remove items with 0 unit price
    df_positive = df_positive[df_positive['UnitPrice'] > 0]
    
    # Calculate total price
    df_positive['TotalPrice'] = df_positive['Quantity'] * df_positive['UnitPrice']
    
    print(f"Cleaned shape: {df_positive.shape}")
    return df_positive
