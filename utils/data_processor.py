import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
import re

class DataProcessor:
    """Handles CSV data loading, validation, and preprocessing for economic analysis"""
    
    def __init__(self):
        self.supported_date_formats = [
            '%Y-%m-%d', '%Y/%m/%d', '%m/%d/%Y', '%d/%m/%Y',
            '%Y-%m', '%Y/%m', '%m/%Y', '%Y'
        ]
    
    def load_and_validate_csv(self, uploaded_file):
        """
        Load and validate the uploaded CSV file
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            pandas.DataFrame: Processed dataframe with datetime index
        """
        try:
            # Read CSV file
            df = pd.read_csv(uploaded_file)
            
            if df.empty:
                st.error("The uploaded file is empty.")
                return None
            
            # Basic validation
            if len(df.columns) < 2:
                st.error("The CSV file must contain at least 2 columns (date and one expenditure column).")
                return None
            
            # Attempt to identify and set date column as index
            df = self._process_date_column(df)
            
            # Clean and validate numeric columns
            df = self._clean_numeric_columns(df)
            
            # Check for minimum data requirements
            if len(df) < 8:
                st.warning("Dataset contains fewer than 8 periods. Analysis results may be limited.")
            
            return df
            
        except Exception as e:
            st.error(f"Error reading CSV file: {str(e)}")
            return None
    
    def _process_date_column(self, df):
        """
        Identify and process the date column to create a datetime index
        
        Args:
            df: Input dataframe
            
        Returns:
            pandas.DataFrame: Dataframe with datetime index
        """
        date_col = None
        
        # Try to identify date column by name
        for col in df.columns:
            col_lower = col.lower()
            if any(date_word in col_lower for date_word in ['date', 'time', 'year', 'period', 'quarter', 'month']):
                date_col = col
                break
        
        # If no date column found by name, try to identify by content
        if date_col is None:
            for col in df.columns:
                if self._is_date_column(df[col]):
                    date_col = col
                    break
        
        if date_col is not None:
            try:
                # Convert to datetime
                df[date_col] = pd.to_datetime(df[date_col], infer_datetime_format=True)
                df = df.set_index(date_col)
                df.index.name = 'Date'
                
                # Sort by date
                df = df.sort_index()
                
                st.info(f"✅ Date column '{date_col}' identified and set as index.")
                
            except Exception as e:
                st.warning(f"Could not convert '{date_col}' to datetime: {str(e)}")
                # Create a simple integer index
                df.index = range(len(df))
                df.index.name = 'Period'
        else:
            # Create a simple integer index if no date column found
            df.index = range(len(df))
            df.index.name = 'Period'
            st.warning("No date column identified. Using sequential period numbers as index.")
        
        return df
    
    def _is_date_column(self, series):
        """
        Check if a pandas series contains date-like data
        
        Args:
            series: Pandas series to check
            
        Returns:
            bool: True if series appears to contain dates
        """
        # Convert to string and check patterns
        str_series = series.astype(str)
        
        # Check for date patterns
        date_patterns = [
            r'\d{4}-\d{1,2}-\d{1,2}',  # YYYY-MM-DD
            r'\d{4}/\d{1,2}/\d{1,2}',  # YYYY/MM/DD
            r'\d{1,2}/\d{1,2}/\d{4}',  # MM/DD/YYYY
            r'\d{1,2}-\d{1,2}-\d{4}',  # MM-DD-YYYY
            r'\d{4}-\d{1,2}',          # YYYY-MM
            r'\d{4}/\d{1,2}',          # YYYY/MM
            r'\d{4}',                  # YYYY
            r'Q[1-4]\s\d{4}',          # Q1 2023
            r'\d{4}Q[1-4]'             # 2023Q1
        ]
        
        matches = 0
        for value in str_series.head(10):  # Check first 10 values
            for pattern in date_patterns:
                if re.match(pattern, value.strip()):
                    matches += 1
                    break
        
        return matches >= len(str_series.head(10)) * 0.5  # At least 50% match
    
    def _clean_numeric_columns(self, df):
        """
        Clean and convert numeric columns
        
        Args:
            df: Input dataframe
            
        Returns:
            pandas.DataFrame: Dataframe with cleaned numeric columns
        """
        numeric_columns = []
        
        for col in df.columns:
            # Try to convert to numeric
            original_series = df[col].copy()
            
            # Remove currency symbols and commas
            if df[col].dtype == 'object':
                cleaned_series = df[col].astype(str).str.replace(r'[$,€£¥₹]', '', regex=True)
                cleaned_series = cleaned_series.str.replace(r'[^\d.-]', '', regex=True)
                
                # Try to convert to numeric
                try:
                    numeric_series = pd.to_numeric(cleaned_series, errors='coerce')
                    
                    # Check if conversion was successful for most values
                    non_null_ratio = numeric_series.notna().sum() / len(numeric_series)
                    
                    if non_null_ratio >= 0.7:  # At least 70% of values are numeric
                        df[col] = numeric_series
                        numeric_columns.append(col)
                    else:
                        # Keep original if conversion failed
                        df[col] = original_series
                        
                except Exception:
                    # Keep original if conversion failed
                    df[col] = original_series
            else:
                # Already numeric
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    numeric_columns.append(col)
                except Exception:
                    pass
        
        if numeric_columns:
            st.info(f"✅ Identified {len(numeric_columns)} numeric columns for analysis.")
        else:
            st.warning("No numeric columns identified. Please check your data format.")
        
        return df
    
    def validate_column_mapping(self, df, column_mapping):
        """
        Validate that the mapped columns exist and contain numeric data
        
        Args:
            df: Input dataframe
            column_mapping: Dictionary mapping expenditure types to column names
            
        Returns:
            dict: Validation results
        """
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        for expenditure_type, column_name in column_mapping.items():
            if column_name is not None:
                if column_name not in df.columns:
                    validation_results['valid'] = False
                    validation_results['errors'].append(f"Column '{column_name}' not found in dataset")
                
                elif not pd.api.types.is_numeric_dtype(df[column_name]):
                    validation_results['valid'] = False
                    validation_results['errors'].append(f"Column '{column_name}' is not numeric")
                
                elif df[column_name].isna().all():
                    validation_results['valid'] = False
                    validation_results['errors'].append(f"Column '{column_name}' contains no valid data")
                
                elif df[column_name].isna().sum() > len(df) * 0.5:
                    validation_results['warnings'].append(f"Column '{column_name}' has many missing values")
        
        return validation_results
    
    def handle_missing_values(self, df, method='interpolate'):
        """
        Handle missing values in the dataset
        
        Args:
            df: Input dataframe
            method: Method for handling missing values ('interpolate', 'forward_fill', 'drop')
            
        Returns:
            pandas.DataFrame: Dataframe with missing values handled
        """
        if method == 'interpolate':
            # Use linear interpolation for numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].interpolate(method='linear')
            
        elif method == 'forward_fill':
            df = df.fillna(method='ffill')
            
        elif method == 'drop':
            df = df.dropna()
        
        return df
    
    def detect_frequency(self, df):
        """
        Detect the frequency of the time series data
        
        Args:
            df: Input dataframe with datetime index
            
        Returns:
            str: Detected frequency ('Annual', 'Quarterly', 'Monthly', 'Unknown')
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            return 'Unknown'
        
        try:
            inferred_freq = pd.infer_freq(df.index)
            
            if inferred_freq:
                if 'A' in inferred_freq or 'Y' in inferred_freq:
                    return 'Annual'
                elif 'Q' in inferred_freq:
                    return 'Quarterly'
                elif 'M' in inferred_freq:
                    return 'Monthly'
                elif 'D' in inferred_freq:
                    return 'Daily'
            
            # Fallback: analyze time differences
            time_diffs = df.index.to_series().diff().dropna()
            median_diff = time_diffs.median()
            
            if median_diff >= pd.Timedelta(days=300):
                return 'Annual'
            elif median_diff >= pd.Timedelta(days=60):
                return 'Quarterly'
            elif median_diff >= pd.Timedelta(days=20):
                return 'Monthly'
            else:
                return 'High Frequency'
                
        except Exception:
            return 'Unknown'
    
    def prepare_analysis_subset(self, df, start_date=None, end_date=None):
        """
        Prepare a subset of data for analysis based on date range
        
        Args:
            df: Input dataframe
            start_date: Start date for analysis
            end_date: End date for analysis
            
        Returns:
            pandas.DataFrame: Subset of data for analysis
        """
        if isinstance(df.index, pd.DatetimeIndex) and start_date and end_date:
            try:
                return df.loc[start_date:end_date]
            except Exception:
                st.warning("Could not filter by date range. Using full dataset.")
                return df
        
        return df
