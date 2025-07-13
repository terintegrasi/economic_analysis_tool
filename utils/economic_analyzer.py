import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.stats.diagnostic import acorr_ljungbox
import warnings
warnings.filterwarnings('ignore')

class EconomicAnalyzer:
    """Performs economic analysis to identify expenditure drivers of GDP growth"""
    
    def __init__(self):
        self.expenditure_categories = {
            'consumption': 'Consumption',
            'investment': 'Investment', 
            'government': 'Government',
            'exports': 'Exports',
            'imports': 'Imports'
        }
    
    def prepare_analysis_data(self, df, column_mapping):
        """
        Prepare data for economic analysis by mapping columns and calculating derived metrics
        
        Args:
            df: Input dataframe
            column_mapping: Dictionary mapping expenditure types to column names
            
        Returns:
            pandas.DataFrame: Prepared analysis dataframe
        """
        try:
            analysis_df = pd.DataFrame(index=df.index)
            
            # Map expenditure columns
            for exp_type, col_name in column_mapping.items():
                if col_name is not None and col_name in df.columns:
                    analysis_df[exp_type] = df[col_name]
            
            # Calculate net exports if both exports and imports are available
            if 'exports' in analysis_df.columns and 'imports' in analysis_df.columns:
                analysis_df['net_exports'] = analysis_df['exports'] - analysis_df['imports']
            
            # Calculate total GDP if not provided
            if 'gdp' not in analysis_df.columns or (analysis_df['gdp'].isna().all() if 'gdp' in analysis_df.columns else True):
                # Calculate GDP as sum of available components
                gdp_components = []
                for component in ['consumption', 'investment', 'government', 'net_exports']:
                    if component in analysis_df.columns:
                        gdp_components.append(component)
                
                if gdp_components:
                    analysis_df['gdp'] = analysis_df[gdp_components].sum(axis=1, skipna=True)
                    st.info(f"✅ GDP calculated as sum of: {', '.join(gdp_components)}")
            
            # Remove rows with all NaN values
            analysis_df = analysis_df.dropna(how='all')
            
            # Fill missing values using interpolation
            numeric_cols = analysis_df.select_dtypes(include=[np.number]).columns
            analysis_df[numeric_cols] = analysis_df[numeric_cols].interpolate(method='linear')
            
            return analysis_df
            
        except Exception as e:
            st.error(f"Error preparing analysis data: {str(e)}")
            return None
    
    def calculate_growth_rates(self, df):
        """
        Calculate various growth rates for all expenditure components
        
        Args:
            df: Analysis dataframe
            
        Returns:
            dict: Dictionary containing growth rate calculations
        """
        growth_rates = {}
        
        for column in df.columns:
            if df[column].notna().sum() > 1:  # Need at least 2 non-null values
                
                # Year-over-year growth rates
                yoy_growth = df[column].pct_change() * 100
                
                # Period-over-period growth rates
                pop_growth = df[column].pct_change() * 100
                
                # Compound Annual Growth Rate (if we have enough data)
                if len(df[column].dropna()) >= 2:
                    start_value = df[column].dropna().iloc[0]
                    end_value = df[column].dropna().iloc[-1]
                    periods = len(df[column].dropna()) - 1
                    
                    if start_value > 0 and periods > 0:
                        cagr = ((end_value / start_value) ** (1/periods) - 1) * 100
                    else:
                        cagr = np.nan
                else:
                    cagr = np.nan
                
                growth_rates[column] = {
                    'yoy_growth': yoy_growth,
                    'pop_growth': pop_growth,
                    'cagr': cagr,
                    'avg_growth': yoy_growth.mean(),
                    'volatility': yoy_growth.std()
                }
        
        return growth_rates
    
    def calculate_correlations(self, df):
        """
        Calculate correlations between expenditure components and GDP
        
        Args:
            df: Analysis dataframe
            
        Returns:
            dict: Correlation results
        """
        correlations = {}
        
        if 'gdp' in df.columns:
            gdp_growth = df['gdp'].pct_change().dropna()
            
            for column in df.columns:
                if column != 'gdp' and df[column].notna().sum() > 3:
                    component_growth = df[column].pct_change().dropna()
                    
                    # Align the series
                    aligned_gdp, aligned_component = gdp_growth.align(component_growth, join='inner')
                    
                    if len(aligned_gdp) > 3:
                        try:
                            correlation, p_value = stats.pearsonr(aligned_component, aligned_gdp)
                            correlations[column] = {
                                'correlation': correlation,
                                'p_value': p_value,
                                'significant': p_value < 0.05
                            }
                        except Exception:
                            correlations[column] = {
                                'correlation': np.nan,
                                'p_value': np.nan,
                                'significant': False
                            }
        
        return correlations
    
    def calculate_contributions(self, df):
        """
        Calculate the contribution of each expenditure component to GDP growth
        
        Args:
            df: Analysis dataframe
            
        Returns:
            dict: Contribution analysis results
        """
        contributions = {}
        
        if 'gdp' not in df.columns:
            return contributions
        
        # Calculate shares of each component in GDP
        for column in df.columns:
            if column != 'gdp' and df[column].notna().sum() > 1:
                
                # Calculate average share in GDP
                share = (df[column] / df['gdp']).mean()
                
                # Calculate growth rates
                component_growth = df[column].pct_change()
                gdp_growth = df['gdp'].pct_change()
                
                # Calculate contribution to GDP growth
                # Contribution = (Component Growth × Component Share)
                contribution_series = component_growth * share
                
                # Average contribution
                avg_contribution = contribution_series.mean()
                
                # Relative contribution (as percentage of total GDP growth)
                if gdp_growth.mean() != 0:
                    relative_contribution = (avg_contribution / gdp_growth.mean()) * 100
                else:
                    relative_contribution = 0
                
                contributions[column] = {
                    'avg_share': share * 100,
                    'avg_contribution': avg_contribution,
                    'relative_contribution': relative_contribution,
                    'contribution_series': contribution_series
                }
        
        return contributions
    
    def perform_regression_analysis(self, df):
        """
        Perform regression analysis for each expenditure component against GDP
        
        Args:
            df: Analysis dataframe
            
        Returns:
            dict: Regression analysis results
        """
        regression_results = {}
        
        if 'gdp' not in df.columns:
            return regression_results
        
        gdp_growth = df['gdp'].pct_change().dropna()
        
        for column in df.columns:
            if column != 'gdp' and df[column].notna().sum() > 5:
                component_growth = df[column].pct_change().dropna()
                
                # Align the series
                aligned_gdp, aligned_component = gdp_growth.align(component_growth, join='inner')
                
                if len(aligned_gdp) > 5:
                    try:
                        # Prepare data for regression
                        X = aligned_component.values.reshape(-1, 1)
                        y = aligned_gdp.values
                        
                        # Remove any infinite or NaN values
                        mask = np.isfinite(X.flatten()) & np.isfinite(y)
                        X = X[mask].reshape(-1, 1)
                        y = y[mask]
                        
                        if len(X) > 3:
                            # Fit linear regression
                            reg = LinearRegression()
                            reg.fit(X, y)
                            
                            # Calculate statistics
                            y_pred = reg.predict(X)
                            r_squared = r2_score(y, y_pred)
                            
                            # Calculate t-statistic and p-value for coefficient
                            n = len(X)
                            mse = np.mean((y - y_pred) ** 2)
                            var_coeff = mse / np.sum((X.flatten() - np.mean(X.flatten())) ** 2)
                            t_stat = reg.coef_[0] / np.sqrt(var_coeff)
                            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
                            
                            regression_results[column] = {
                                'coefficient': reg.coef_[0],
                                'intercept': reg.intercept_,
                                'r_squared': r_squared,
                                't_statistic': t_stat,
                                'p_value': p_value,
                                'significant': p_value < 0.05
                            }
                    
                    except Exception as e:
                        st.warning(f"Regression analysis failed for {column}: {str(e)}")
                        continue
        
        return regression_results
    
    def identify_primary_driver(self, correlations, contributions, regression_results):
        """
        Identify the primary driver of economic growth based on multiple criteria
        
        Args:
            correlations: Correlation analysis results
            contributions: Contribution analysis results  
            regression_results: Regression analysis results
            
        Returns:
            str: Name of primary growth driver
        """
        scores = {}
        
        # Get all available expenditure categories
        all_categories = set()
        all_categories.update(correlations.keys())
        all_categories.update(contributions.keys())
        all_categories.update(regression_results.keys())
        
        for category in all_categories:
            score = 0
            
            # Correlation score (0-40 points)
            if category in correlations:
                corr = abs(correlations[category].get('correlation', 0))
                if not np.isnan(corr):
                    score += corr * 40
                
                # Bonus for statistical significance
                if correlations[category].get('significant', False):
                    score += 10
            
            # Contribution score (0-30 points)
            if category in contributions:
                rel_contrib = abs(contributions[category].get('relative_contribution', 0))
                if not np.isnan(rel_contrib):
                    score += min(rel_contrib / 100 * 30, 30)  # Cap at 30 points
            
            # Regression score (0-30 points)
            if category in regression_results:
                r_squared = regression_results[category].get('r_squared', 0)
                if not np.isnan(r_squared):
                    score += r_squared * 30
                
                # Bonus for statistical significance
                if regression_results[category].get('significant', False):
                    score += 10
            
            scores[category] = score
        
        if scores:
            primary_driver = max(scores.keys(), key=lambda k: scores[k])
            return self.expenditure_categories.get(primary_driver, primary_driver)
        else:
            return "Insufficient data"
    
    def analyze_growth_drivers(self, df, method='correlation_contribution'):
        """
        Perform comprehensive analysis to identify growth drivers
        
        Args:
            df: Analysis dataframe
            method: Analysis method ('correlation_contribution', 'log_log_regression', 'average_elasticity')
            
        Returns:
            dict: Comprehensive analysis results
        """
        results = {}
        
        # Calculate growth rates
        growth_rates = self.calculate_growth_rates(df)
        results['growth_rates'] = growth_rates
        
        # Calculate correlations
        correlations = self.calculate_correlations(df)
        results['correlations'] = {k: v['correlation'] for k, v in correlations.items() if not np.isnan(v['correlation'])}
        results['correlation_details'] = correlations
        
        # Calculate contributions
        contributions = self.calculate_contributions(df)
        results['growth_contributions'] = {k: v['relative_contribution'] for k, v in contributions.items()}
        results['contribution_details'] = contributions
        
        # Perform regression analysis
        regression_results = self.perform_regression_analysis(df)
        results['regression_results'] = regression_results
        
        # Calculate overall GDP growth rate
        if 'gdp' in df.columns:
            gdp_growth_series = df['gdp'].pct_change() * 100
            results['gdp_growth_rate'] = gdp_growth_series.mean()
        else:
            results['gdp_growth_rate'] = 0
        
        # Find strongest correlation
        if results['correlations']:
            strongest_corr = max(results['correlations'].values(), key=abs)
            results['strongest_correlation'] = strongest_corr
        else:
            results['strongest_correlation'] = 0
        
        # Identify primary driver using selected method and add method-specific results
        if method == 'log_log_regression':
            primary_driver = self._identify_driver_log_log_regression(df)
            # Add log-log regression specific results
            log_log_results = self._calculate_log_log_elasticities(df)
            results['log_log_elasticities'] = log_log_results
        elif method == 'average_elasticity':
            primary_driver = self._identify_driver_average_elasticity(df)
            # Add average elasticity specific results
            elasticity_results = self._calculate_average_elasticities(df)
            results['average_elasticities'] = elasticity_results
        else:  # correlation_contribution (default)
            primary_driver = self.identify_primary_driver(correlations, contributions, regression_results)
        
        results['primary_driver'] = primary_driver
        results['analysis_method'] = method
        
        # Create summary statistics
        summary_data = []
        for category in df.columns:
            if category in growth_rates:
                gr = growth_rates[category]
                corr = correlations.get(category, {}).get('correlation', np.nan)
                contrib = contributions.get(category, {}).get('relative_contribution', np.nan)
                
                summary_data.append({
                    'Expenditure Type': self.expenditure_categories.get(category, category.title()),
                    'Avg Growth Rate (%)': round(gr['avg_growth'], 2) if not np.isnan(gr['avg_growth']) else None,
                    'Volatility (%)': round(gr['volatility'], 2) if not np.isnan(gr['volatility']) else None,
                    'GDP Correlation': round(corr, 3) if not np.isnan(corr) else None,
                    'Growth Contribution (%)': round(contrib, 2) if not np.isnan(contrib) else None,
                    'CAGR (%)': round(gr['cagr'], 2) if not np.isnan(gr['cagr']) else None
                })
        
        results['summary_stats'] = pd.DataFrame(summary_data)
        
        # Add trend analysis
        results['trend_analysis'] = self._analyze_trends(df, growth_rates)
        
        return results
    
    def forecast_growth_drivers(self, df, forecast_years=5, method='linear_regression'):
        """
        Forecast GDP and expenditure components using selected method
        
        Args:
            df: Analysis dataframe
            forecast_years: Number of years to forecast
            method: Forecasting method ('linear_regression', 'polynomial', 'arima', 'random_forest', 'svm')
            
        Returns:
            dict: Forecast results with predictions and confidence intervals
        """
        forecast_results = {}
        
        # Prepare time index for forecasting
        if isinstance(df.index, pd.DatetimeIndex):
            # For datetime index, extend by years
            last_date = df.index[-1]
            if hasattr(last_date, 'year'):
                forecast_dates = pd.date_range(
                    start=last_date + pd.DateOffset(years=1),
                    periods=forecast_years,
                    freq='Y'
                )
            else:
                forecast_dates = pd.date_range(
                    start=last_date,
                    periods=forecast_years + 1,
                    freq='Y'
                )[1:]
        else:
            # For numeric index, extend sequentially
            last_index = df.index[-1]
            forecast_dates = pd.Index(range(last_index + 1, last_index + forecast_years + 1))
        
        # Forecast each component
        for column in df.columns:
            if df[column].notna().sum() > 5:  # Need sufficient data points
                try:
                    predictions, confidence_lower, confidence_upper = self._apply_forecast_method(
                        df[column].dropna(), method, forecast_years
                    )
                    
                    forecast_results[column] = {
                        'predictions': predictions,
                        'confidence_lower': confidence_lower,
                        'confidence_upper': confidence_upper,
                        'forecast_dates': forecast_dates[:len(predictions)],
                        'method': method
                    }
                    
                except Exception as e:
                    st.warning(f"Forecasting failed for {column}: {str(e)}")
                    continue
        
        return forecast_results
    
    def _apply_forecast_method(self, series, method, forecast_periods):
        """
        Apply the selected forecasting method to a time series
        
        Args:
            series: Time series data
            method: Forecasting method
            forecast_periods: Number of periods to forecast
            
        Returns:
            tuple: (predictions, confidence_lower, confidence_upper)
        """
        # Prepare data
        y = series.values
        X = np.arange(len(y)).reshape(-1, 1)
        X_forecast = np.arange(len(y), len(y) + forecast_periods).reshape(-1, 1)
        
        if method == 'linear_regression':
            # Simple linear regression
            model = LinearRegression()
            model.fit(X, y)
            predictions = model.predict(X_forecast)
            
            # Calculate prediction intervals (simplified)
            residuals = y - model.predict(X)
            mse = np.mean(residuals ** 2)
            std_error = np.sqrt(mse)
            
            confidence_lower = predictions - 1.96 * std_error
            confidence_upper = predictions + 1.96 * std_error
            
        elif method == 'polynomial':
            # Polynomial regression (degree 2)
            X_poly = np.column_stack([X.flatten(), X.flatten() ** 2])
            X_forecast_poly = np.column_stack([X_forecast.flatten(), X_forecast.flatten() ** 2])
            
            model = LinearRegression()
            model.fit(X_poly, y)
            predictions = model.predict(X_forecast_poly)
            
            residuals = y - model.predict(X_poly)
            mse = np.mean(residuals ** 2)
            std_error = np.sqrt(mse)
            
            confidence_lower = predictions - 1.96 * std_error
            confidence_upper = predictions + 1.96 * std_error
            
        elif method == 'random_forest':
            # Random Forest regression
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
            predictions = model.predict(X_forecast)
            
            # For confidence intervals, use prediction variance from trees
            tree_predictions = np.array([tree.predict(X_forecast) for tree in model.estimators_])
            std_predictions = np.std(tree_predictions, axis=0)
            
            confidence_lower = predictions - 1.96 * std_predictions
            confidence_upper = predictions + 1.96 * std_predictions
            
        elif method == 'svm':
            # Support Vector Machine regression
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X_forecast_scaled = scaler.transform(X_forecast)
            
            model = SVR(kernel='rbf', C=1.0, gamma='scale')
            model.fit(X_scaled, y)
            predictions = model.predict(X_forecast_scaled)
            
            # Simple confidence intervals for SVM
            residuals = y - model.predict(X_scaled)
            std_error = np.std(residuals)
            
            confidence_lower = predictions - 1.96 * std_error
            confidence_upper = predictions + 1.96 * std_error
            
        elif method == 'moving_average':
            # Simple moving average
            window = min(5, len(y) // 2)  # Use 5-period moving average or half the data
            if window < 2:
                window = 2
            
            # Calculate moving average
            ma = np.convolve(y, np.ones(window)/window, mode='valid')
            last_ma = ma[-1]
            
            # Simple trend adjustment
            if len(ma) > 1:
                trend = (ma[-1] - ma[0]) / (len(ma) - 1)
            else:
                trend = 0
            
            predictions = np.array([last_ma + trend * i for i in range(1, forecast_periods + 1)])
            
            # Calculate volatility for confidence intervals
            volatility = np.std(y[-window:]) if len(y) >= window else np.std(y)
            confidence_lower = predictions - 1.96 * volatility
            confidence_upper = predictions + 1.96 * volatility
            
        elif method == 'var_model':
            # Vector Autoregression (VAR) - requires multivariate data
            # For univariate case, we'll use AR(1) approximation
            if len(y) > 10:
                try:
                    # Create lagged variables for AR model
                    y_series = pd.Series(y)
                    y_lag1 = y_series.shift(1).dropna()
                    y_current = y_series[1:]
                    
                    # Fit simple AR(1) model
                    model = LinearRegression()
                    model.fit(y_lag1.values.reshape(-1, 1), y_current.values)
                    
                    # Generate forecasts iteratively
                    predictions = []
                    last_value = y[-1]
                    
                    for i in range(forecast_periods):
                        next_pred = model.predict([[last_value]])[0]
                        predictions.append(next_pred)
                        last_value = next_pred
                    
                    predictions = np.array(predictions)
                    
                    # Calculate confidence intervals based on residuals
                    residuals = y_current.values - model.predict(y_lag1.values.reshape(-1, 1))
                    std_error = np.std(residuals)
                    confidence_lower = predictions - 1.96 * std_error
                    confidence_upper = predictions + 1.96 * std_error
                    
                except Exception:
                    # Fallback to linear regression
                    model = LinearRegression()
                    model.fit(X, y)
                    predictions = model.predict(X_forecast)
                    residuals = y - model.predict(X)
                    mse = np.mean(residuals ** 2)
                    std_error = np.sqrt(mse)
                    confidence_lower = predictions - 1.96 * std_error
                    confidence_upper = predictions + 1.96 * std_error
            else:
                # Fallback for insufficient data
                model = LinearRegression()
                model.fit(X, y)
                predictions = model.predict(X_forecast)
                residuals = y - model.predict(X)
                mse = np.mean(residuals ** 2)
                std_error = np.sqrt(mse)
                confidence_lower = predictions - 1.96 * std_error
                confidence_upper = predictions + 1.96 * std_error
                
        elif method == 'gmm':
            # Generalized Method of Moments (simplified implementation)
            # Using instrumental variables approach with lagged values
            if len(y) > 15:
                try:
                    # Create instruments (lagged values)
                    y_series = pd.Series(y)
                    y_lag1 = y_series.shift(1).dropna()
                    y_lag2 = y_series.shift(2).dropna()
                    y_current = y_series[2:]
                    
                    # Use two-stage least squares approximation
                    # First stage: regress current on lag2
                    first_stage = LinearRegression()
                    first_stage.fit(y_lag2.values.reshape(-1, 1), y_lag1[1:].values)
                    y_lag1_fitted = first_stage.predict(y_lag2.values.reshape(-1, 1))
                    
                    # Second stage: regress current on fitted lag1
                    second_stage = LinearRegression()
                    second_stage.fit(y_lag1_fitted.reshape(-1, 1), y_current.values)
                    
                    # Generate forecasts
                    predictions = []
                    last_value = y[-1]
                    second_last = y[-2] if len(y) > 1 else y[-1]
                    
                    for i in range(forecast_periods):
                        # Predict next lag1 value
                        next_lag1 = first_stage.predict([[second_last]])[0]
                        # Predict next value
                        next_pred = second_stage.predict([[next_lag1]])[0]
                        predictions.append(next_pred)
                        second_last = last_value
                        last_value = next_pred
                    
                    predictions = np.array(predictions)
                    
                    # Calculate confidence intervals
                    residuals = y_current.values - second_stage.predict(y_lag1_fitted.reshape(-1, 1))
                    std_error = np.std(residuals)
                    confidence_lower = predictions - 1.96 * std_error
                    confidence_upper = predictions + 1.96 * std_error
                    
                except Exception:
                    # Fallback to linear regression
                    model = LinearRegression()
                    model.fit(X, y)
                    predictions = model.predict(X_forecast)
                    residuals = y - model.predict(X)
                    mse = np.mean(residuals ** 2)
                    std_error = np.sqrt(mse)
                    confidence_lower = predictions - 1.96 * std_error
                    confidence_upper = predictions + 1.96 * std_error
            else:
                # Fallback for insufficient data
                model = LinearRegression()
                model.fit(X, y)
                predictions = model.predict(X_forecast)
                residuals = y - model.predict(X)
                mse = np.mean(residuals ** 2)
                std_error = np.sqrt(mse)
                confidence_lower = predictions - 1.96 * std_error
                confidence_upper = predictions + 1.96 * std_error
                
        elif method == 'cge_model':
            # Computable General Equilibrium (simplified approach)
            # Using elasticity-based projections
            if len(y) > 5:
                # Calculate historical elasticity (responsiveness to changes)
                y_changes = np.diff(y)
                y_levels = y[:-1]
                
                # Calculate average elasticity
                elasticities = []
                for i in range(len(y_changes)):
                    if y_levels[i] != 0:
                        elasticity = (y_changes[i] / y_levels[i])
                        elasticities.append(elasticity)
                
                if elasticities:
                    avg_elasticity = np.mean(elasticities)
                    elasticity_std = np.std(elasticities)
                else:
                    avg_elasticity = 0.02  # Default 2% growth
                    elasticity_std = 0.01
                
                # Generate forecasts using elasticity
                predictions = []
                last_value = y[-1]
                
                for i in range(forecast_periods):
                    # Apply elasticity with some random variation
                    growth_rate = avg_elasticity + np.random.normal(0, elasticity_std/2)
                    next_value = last_value * (1 + growth_rate)
                    predictions.append(next_value)
                    last_value = next_value
                
                predictions = np.array(predictions)
                
                # Calculate confidence intervals based on historical volatility
                volatility = np.std(y_changes) if len(y_changes) > 0 else np.std(y) * 0.1
                confidence_lower = predictions - 1.96 * volatility
                confidence_upper = predictions + 1.96 * volatility
            else:
                # Fallback to linear regression for small datasets
                model = LinearRegression()
                model.fit(X, y)
                predictions = model.predict(X_forecast)
                residuals = y - model.predict(X)
                mse = np.mean(residuals ** 2)
                std_error = np.sqrt(mse)
                confidence_lower = predictions - 1.96 * std_error
                confidence_upper = predictions + 1.96 * std_error
            
        else:
            # Default to linear regression
            model = LinearRegression()
            model.fit(X, y)
            predictions = model.predict(X_forecast)
            
            residuals = y - model.predict(X)
            mse = np.mean(residuals ** 2)
            std_error = np.sqrt(mse)
            
            confidence_lower = predictions - 1.96 * std_error
            confidence_upper = predictions + 1.96 * std_error
        
        return predictions, confidence_lower, confidence_upper
    
    def calculate_forecast_accuracy(self, df, method='linear_regression', test_periods=3):
        """
        Calculate forecast accuracy using backtesting
        
        Args:
            df: Analysis dataframe
            method: Forecasting method to test
            test_periods: Number of periods to use for testing
            
        Returns:
            dict: Accuracy metrics for each component
        """
        accuracy_results = {}
        
        for column in df.columns:
            if df[column].notna().sum() > test_periods + 5:
                try:
                    # Split data into train and test
                    series = df[column].dropna()
                    train_data = series[:-test_periods]
                    test_data = series[-test_periods:]
                    
                    # Generate forecasts
                    predictions, _, _ = self._apply_forecast_method(train_data, method, test_periods)
                    
                    # Calculate accuracy metrics
                    mae = mean_absolute_error(test_data.values, predictions)
                    mse = mean_squared_error(test_data.values, predictions)
                    rmse = np.sqrt(mse)
                    
                    # Calculate percentage errors
                    mape = np.mean(np.abs((test_data.values - predictions) / test_data.values)) * 100
                    
                    accuracy_results[column] = {
                        'mae': mae,
                        'mse': mse,
                        'rmse': rmse,
                        'mape': mape,
                        'method': method
                    }
                    
                except Exception as e:
                    continue
        
        return accuracy_results
    
    def _analyze_trends(self, df, growth_rates):
        """
        Analyze trends in the data
        
        Args:
            df: Analysis dataframe
            growth_rates: Growth rate calculations
            
        Returns:
            str: Trend analysis summary
        """
        trend_summary = []
        
        for column, gr_data in growth_rates.items():
            if 'yoy_growth' in gr_data:
                growth_series = gr_data['yoy_growth'].dropna()
                
                if len(growth_series) > 2:
                    # Check for trend using simple linear regression
                    x = np.arange(len(growth_series))
                    y = growth_series.values
                    
                    if len(x) > 1 and not np.all(np.isnan(y)):
                        slope, _, _, p_value, _ = stats.linregress(x, y)
                        
                        category_name = self.expenditure_categories.get(column, column.title())
                        
                        if p_value < 0.05:
                            if slope > 0:
                                trend_summary.append(f"{category_name}: Positive growth trend (slope: {slope:.3f})")
                            else:
                                trend_summary.append(f"{category_name}: Negative growth trend (slope: {slope:.3f})")
                        else:
                            trend_summary.append(f"{category_name}: No significant trend detected")
        
        return " | ".join(trend_summary) if trend_summary else "Trend analysis not available"
    
    def _identify_driver_log_log_regression(self, df):
        """
        Identify primary growth driver using log-log regression analysis
        
        Args:
            df: Analysis dataframe
            
        Returns:
            str: Primary driver identified through log-log regression
        """
        if 'gdp' not in df.columns:
            return "Insufficient data"
        
        import numpy as np
        from sklearn.linear_model import LinearRegression
        
        # Convert to log values (add small constant to avoid log(0))
        log_gdp = np.log(df['gdp'] + 1e-6)
        
        elasticities = {}
        for column in df.columns:
            if column != 'gdp' and df[column].dtype in ['float64', 'int64']:
                try:
                    log_component = np.log(df[column] + 1e-6)
                    
                    # Remove any infinite or NaN values
                    valid_mask = np.isfinite(log_gdp) & np.isfinite(log_component)
                    if valid_mask.sum() < 3:  # Need at least 3 points
                        continue
                    
                    X = log_component[valid_mask].values.reshape(-1, 1)
                    y = log_gdp[valid_mask].values
                    
                    model = LinearRegression()
                    model.fit(X, y)
                    
                    # Elasticity is the coefficient in log-log regression
                    elasticities[column] = abs(model.coef_[0])
                except Exception:
                    continue
        
        if elasticities:
            primary_driver = max(elasticities.keys(), key=lambda k: elasticities[k])
            return self.expenditure_categories.get(primary_driver, primary_driver)
        else:
            return "Insufficient data"
    
    def _identify_driver_average_elasticity(self, df):
        """
        Identify primary growth driver using average elasticity method
        
        Args:
            df: Analysis dataframe
            
        Returns:
            str: Primary driver identified through average elasticity
        """
        if 'gdp' not in df.columns:
            return "Insufficient data"
        
        import numpy as np
        
        # Calculate elasticities using percentage changes
        gdp_pct_change = df['gdp'].pct_change()
        
        average_elasticities = {}
        for column in df.columns:
            if column != 'gdp' and df[column].dtype in ['float64', 'int64']:
                try:
                    component_pct_change = df[column].pct_change()
                    
                    # Calculate elasticity = % change in GDP / % change in component
                    # Remove infinite and zero values
                    valid_mask = (
                        np.isfinite(gdp_pct_change) & 
                        np.isfinite(component_pct_change) & 
                        (component_pct_change != 0) &
                        (abs(component_pct_change) > 1e-6)
                    )
                    
                    if valid_mask.sum() < 2:
                        continue
                    
                    elasticities = gdp_pct_change[valid_mask] / component_pct_change[valid_mask]
                    # Take absolute value and average
                    average_elasticities[column] = np.mean(np.abs(elasticities))
                    
                except Exception:
                    continue
        
        if average_elasticities:
            primary_driver = max(average_elasticities.keys(), key=lambda k: average_elasticities[k])
            return self.expenditure_categories.get(primary_driver, primary_driver)
        else:
            return "Insufficient data"
    
    def get_var_analysis_results(self, df):
        """
        Get VAR-specific analysis results including impulse response and variance decomposition
        
        Args:
            df: Analysis dataframe
            
        Returns:
            dict: VAR analysis results
        """
        try:
            from statsmodels.tsa.vector_ar.var_model import VAR
            import numpy as np
            
            # Prepare data for VAR analysis
            var_data = df.dropna()
            if len(var_data) < 10:  # Need sufficient data points
                return None
            
            # Fit VAR model
            model = VAR(var_data)
            # Use information criteria to select optimal lag
            lag_order = model.select_order(maxlags=min(4, len(var_data)//4))
            optimal_lags = lag_order.aic if hasattr(lag_order, 'aic') else 2
            
            var_fitted = model.fit(optimal_lags)
            
            # Calculate impulse response functions
            irf = var_fitted.irf(periods=10)
            
            # Calculate variance decomposition
            variance_decomp = var_fitted.fevd(periods=10)
            
            return {
                'model': var_fitted,
                'impulse_response': irf,
                'variance_decomposition': variance_decomp,
                'lag_order': optimal_lags
            }
            
        except Exception as e:
            return None
    
    def _calculate_log_log_elasticities(self, df):
        """
        Calculate detailed log-log regression elasticities for all components
        
        Args:
            df: Analysis dataframe
            
        Returns:
            dict: Detailed elasticity results for each component
        """
        if 'gdp' not in df.columns:
            return {}
        
        import numpy as np
        from sklearn.linear_model import LinearRegression
        from scipy import stats
        
        # Convert to log values (add small constant to avoid log(0))
        log_gdp = np.log(df['gdp'] + 1e-6)
        
        elasticity_results = {}
        for column in df.columns:
            if column != 'gdp' and df[column].dtype in ['float64', 'int64']:
                try:
                    log_component = np.log(df[column] + 1e-6)
                    
                    # Remove any infinite or NaN values
                    valid_mask = np.isfinite(log_gdp) & np.isfinite(log_component)
                    if valid_mask.sum() < 3:  # Need at least 3 points
                        continue
                    
                    X = log_component[valid_mask].values.reshape(-1, 1)
                    y = log_gdp[valid_mask].values
                    
                    model = LinearRegression()
                    model.fit(X, y)
                    
                    # Calculate R-squared
                    y_pred = model.predict(X)
                    r_squared = 1 - (np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2))
                    
                    # Calculate p-value using t-test
                    residuals = y - y_pred
                    mse = np.mean(residuals ** 2)
                    se = np.sqrt(mse * np.sum((X.flatten() - np.mean(X)) ** 2))
                    if se > 0:
                        t_stat = model.coef_[0] / se
                        p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), len(y) - 2))
                    else:
                        p_value = 1.0
                    
                    elasticity_results[column] = {
                        'elasticity': model.coef_[0],
                        'r_squared': r_squared,
                        'p_value': p_value,
                        'intercept': model.intercept_,
                        'significant': p_value < 0.05
                    }
                except Exception:
                    continue
        
        return elasticity_results
    
    def _calculate_average_elasticities(self, df):
        """
        Calculate detailed average elasticities for all components
        
        Args:
            df: Analysis dataframe
            
        Returns:
            dict: Detailed elasticity results for each component
        """
        if 'gdp' not in df.columns:
            return {}
        
        import numpy as np
        
        # Calculate elasticities using percentage changes
        gdp_pct_change = df['gdp'].pct_change()
        
        elasticity_results = {}
        for column in df.columns:
            if column != 'gdp' and df[column].dtype in ['float64', 'int64']:
                try:
                    component_pct_change = df[column].pct_change()
                    
                    # Calculate elasticity = % change in GDP / % change in component
                    # Remove infinite and zero values
                    valid_mask = (
                        np.isfinite(gdp_pct_change) & 
                        np.isfinite(component_pct_change) & 
                        (component_pct_change != 0) &
                        (abs(component_pct_change) > 1e-6)
                    )
                    
                    if valid_mask.sum() < 2:
                        continue
                    
                    elasticities = gdp_pct_change[valid_mask] / component_pct_change[valid_mask]
                    
                    # Calculate statistics
                    avg_elasticity = np.mean(elasticities)
                    std_elasticity = np.std(elasticities)
                    abs_avg_elasticity = np.mean(np.abs(elasticities))
                    
                    # Calculate correlation for reference
                    correlation = np.corrcoef(gdp_pct_change[valid_mask], component_pct_change[valid_mask])[0, 1]
                    
                    elasticity_results[column] = {
                        'average_elasticity': avg_elasticity,
                        'absolute_average_elasticity': abs_avg_elasticity,
                        'standard_deviation': std_elasticity,
                        'correlation': correlation,
                        'observations': valid_mask.sum(),
                        'elasticity_range': {
                            'min': np.min(elasticities),
                            'max': np.max(elasticities),
                            'median': np.median(elasticities)
                        }
                    }
                    
                except Exception:
                    continue
        
        return elasticity_results
