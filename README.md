# Economic Growth Analysis Tool

A sophisticated Streamlit-based economic analysis platform that enables comprehensive regional GDP growth analysis with advanced visualization and reporting capabilities.

## Features

### Core Analysis
- **Multiple Analysis Methods**: Correlation & Contribution, Log-Log Regression, Average Elasticity
- **Advanced Forecasting**: Linear Regression, Polynomial, Random Forest, SVM, VAR, ARIMA, GMM, CGE
- **Interactive Visualizations**: Time series plots, correlation heatmaps, growth contribution charts
- **Statistical Analysis**: Regression tables with color-coded p-values, elasticity calculations

### Reporting & Export
- **Comprehensive Reports**: PDF and DOCX formats with all analysis data
- **Multilingual Support**: English and Indonesian language options
- **Regional Customization**: Personalized reports for specific countries/regions
- **Professional Formatting**: Clean tables, charts, and statistical summaries

### Advanced Features
- **Custom Growth Scenarios**: Set custom GDP growth rates with automatic component adjustments
- **Forecast Accuracy**: Backtesting with MAPE, RMSE, and MAE metrics
- **VAR Analysis**: Impulse Response Functions and Variance Decomposition
- **Data Processing**: Automatic date detection, missing value handling, frequency detection

## Installation

1. Install Python 3.11 or higher
2. Install dependencies:
   ```bash
   pip install streamlit pandas numpy matplotlib plotly scikit-learn scipy statsmodels python-docx reportlab
   ```

## Usage

1. Run the application:
   ```bash
   streamlit run app.py --server.port 5000
   ```

2. Upload your CSV file with economic data including:
   - Date column (automatically detected)
   - GDP data
   - Expenditure components (Consumption, Investment, Government, Exports, Imports)

3. Map your data columns to the appropriate expenditure categories

4. Choose your analysis method and run the analysis

5. Explore results through interactive tabs:
   - **Overview**: Primary growth driver and key metrics
   - **Visualizations**: Interactive charts and plots
   - **Analysis**: Detailed statistical results
   - **Forecasting**: Future projections and accuracy
   - **Export**: Generate comprehensive reports

## File Structure

```
economic_analysis_tool/
├── app.py                      # Main Streamlit application
├── utils/
│   ├── data_processor.py      # Data loading and validation
│   ├── economic_analyzer.py   # Economic analysis algorithms
│   └── visualizer.py          # Chart and plot generation
├── .streamlit/
│   └── config.toml            # Streamlit configuration
├── pyproject.toml             # Python dependencies
└── uv.lock                    # Dependency lock file
```

## Key Components

### DataProcessor
- CSV file validation and loading
- Automatic date column detection
- Missing value handling
- Data frequency detection

### EconomicAnalyzer
- Multiple growth driver identification methods
- Advanced forecasting algorithms
- Statistical analysis and regression
- VAR model implementation

### Visualizer
- Interactive time series plots
- Correlation heatmaps
- Forecast visualizations with confidence intervals
- Growth contribution charts

## Analysis Methods

### 1. Correlation & Contribution (Default)
- Pearson correlation analysis
- Growth contribution calculations
- Linear regression with statistical significance

### 2. Log-Log Regression
- Elasticity coefficient estimation
- Statistical significance testing
- R-squared model fit evaluation

### 3. Average Elasticity
- Dynamic elasticity calculations
- Comprehensive elasticity statistics
- Correlation and variability analysis

## Forecasting Methods

- **Linear Regression**: Simple trend extrapolation
- **Polynomial**: Non-linear trend fitting
- **Random Forest**: Machine learning ensemble method
- **SVM**: Support Vector Machine regression
- **VAR**: Vector Autoregression for multivariate analysis
- **ARIMA**: Auto-regressive integrated moving average
- **GMM**: Generalized Method of Moments
- **CGE**: Computable General Equilibrium (simplified)

## Export Features

### Report Formats
- **PDF**: Professional reports with tables and charts
- **DOCX**: Editable Word documents with complete analysis

### Report Content
- Executive summary with key findings
- Data overview and methodology
- Method-specific analysis results
- Correlation and contribution tables
- Forecast results and accuracy metrics
- Conclusions and recommendations

### Multilingual Support
- English: Complete analysis in English
- Indonesian: Full Indonesian language support

## Technical Requirements

- Python 3.11+
- Streamlit 1.28+
- Pandas, NumPy for data processing
- Plotly for interactive visualizations
- Scikit-learn for machine learning
- SciPy, Statsmodels for statistical analysis
- ReportLab for PDF generation
- Python-docx for Word document creation

## Data Requirements

Your CSV file should include:
- **Date column**: Any recognizable date format
- **GDP column**: Gross Domestic Product values
- **Expenditure components**: At least one of:
  - Consumption
  - Investment
  - Government spending
  - Exports
  - Imports

## Getting Started

1. Prepare your economic data in CSV format
2. Launch the application
3. Upload your data file
4. Map columns to expenditure categories
5. Select analysis method and forecasting options
6. Run analysis and explore results
7. Generate comprehensive reports

## Support

For technical support or questions about the economic analysis methods, refer to the built-in methodology section in the application.

---

**Developed for comprehensive economic growth analysis and policy research.**
