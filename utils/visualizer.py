import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import streamlit as st

class Visualizer:
    """Creates interactive visualizations for economic analysis results"""
    
    def __init__(self):
        self.color_palette = {
            'consumption': '#2E86AB',
            'investment': '#A23B72', 
            'government': '#F18F01',
            'exports': '#C73E1D',
            'imports': '#592E83',
            'net_exports': '#4CAF50',
            'gdp': '#1f77b4'
        }
        
        self.expenditure_labels = {
            'consumption': 'Consumption',
            'investment': 'Investment',
            'government': 'Government',
            'exports': 'Exports', 
            'imports': 'Imports',
            'net_exports': 'Net Exports',
            'gdp': 'GDP'
        }
    
    def create_time_series_plot(self, df):
        """
        Create interactive time series plot of all expenditure components
        
        Args:
            df: Analysis dataframe
            
        Returns:
            plotly.graph_objects.Figure: Time series plot
        """
        fig = go.Figure()
        
        for column in df.columns:
            if df[column].notna().sum() > 1:
                color = self.color_palette.get(column, '#636EFA')
                label = self.expenditure_labels.get(column, column.title())
                
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df[column],
                    mode='lines+markers',
                    name=label,
                    line=dict(color=color, width=2),
                    marker=dict(size=4),
                    hovertemplate=f'<b>{label}</b><br>Date: %{{x}}<br>Value: %{{y:,.0f}}<extra></extra>'
                ))
        
        fig.update_layout(
            title="Expenditure Components Over Time",
            xaxis_title="Time Period",
            yaxis_title="Value",
            hovermode='x unified',
            showlegend=True,
            height=500,
            template='plotly_white'
        )
        
        return fig
    
    def create_growth_rates_plot(self, growth_rates_data):
        """
        Create growth rates visualization
        
        Args:
            growth_rates_data: Dictionary of growth rate calculations
            
        Returns:
            plotly.graph_objects.Figure: Growth rates plot
        """
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Year-over-Year Growth Rates', 'Average Growth Rates Comparison'),
            specs=[[{"secondary_y": False}], [{"secondary_y": False}]],
            vertical_spacing=0.15
        )
        
        # Plot 1: Time series of growth rates
        for component, data in growth_rates_data.items():
            if 'yoy_growth' in data and data['yoy_growth'].notna().sum() > 1:
                color = self.color_palette.get(component, '#636EFA')
                label = self.expenditure_labels.get(component, component.title())
                
                fig.add_trace(
                    go.Scatter(
                        x=data['yoy_growth'].index,
                        y=data['yoy_growth'],
                        mode='lines+markers',
                        name=f'{label} Growth',
                        line=dict(color=color, width=2),
                        marker=dict(size=3),
                        hovertemplate=f'<b>{label}</b><br>Date: %{{x}}<br>Growth: %{{y:.2f}}%<extra></extra>'
                    ),
                    row=1, col=1
                )
        
        # Plot 2: Average growth rates comparison
        components = []
        avg_growth = []
        colors = []
        
        for component, data in growth_rates_data.items():
            if not np.isnan(data.get('avg_growth', np.nan)):
                components.append(self.expenditure_labels.get(component, component.title()))
                avg_growth.append(data['avg_growth'])
                colors.append(self.color_palette.get(component, '#636EFA'))
        
        if components:
            fig.add_trace(
                go.Bar(
                    x=components,
                    y=avg_growth,
                    name='Average Growth',
                    marker_color=colors,
                    hovertemplate='<b>%{x}</b><br>Avg Growth: %{y:.2f}%<extra></extra>'
                ),
                row=2, col=1
            )
        
        fig.update_layout(
            height=700,
            showlegend=True,
            template='plotly_white'
        )
        
        fig.update_xaxes(title_text="Time Period", row=1, col=1)
        fig.update_yaxes(title_text="Growth Rate (%)", row=1, col=1)
        fig.update_xaxes(title_text="Expenditure Type", row=2, col=1)
        fig.update_yaxes(title_text="Average Growth Rate (%)", row=2, col=1)
        
        return fig
    
    def create_contribution_chart(self, contributions_data):
        """
        Create contribution analysis visualization
        
        Args:
            contributions_data: Dictionary of contribution values
            
        Returns:
            plotly.graph_objects.Figure: Contribution chart
        """
        # Prepare data for visualization
        components = []
        contributions = []
        colors = []
        
        for component, contribution in contributions_data.items():
            if not np.isnan(contribution):
                components.append(self.expenditure_labels.get(component, component.title()))
                contributions.append(contribution)
                colors.append(self.color_palette.get(component, '#636EFA'))
        
        if not components:
            # Return empty figure if no data
            fig = go.Figure()
            fig.add_annotation(text="No contribution data available", 
                             xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig
        
        # Create subplot with pie chart and bar chart
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{"type": "domain"}, {"type": "xy"}]],
            subplot_titles=('Contribution Distribution', 'Contribution Values'),
            column_widths=[0.4, 0.6]
        )
        
        # Pie chart
        fig.add_trace(
            go.Pie(
                labels=components,
                values=[abs(c) for c in contributions],  # Use absolute values for pie chart
                marker_colors=colors,
                hovertemplate='<b>%{label}</b><br>Contribution: %{value:.2f}%<extra></extra>',
                textinfo='label+percent'
            ),
            row=1, col=1
        )
        
        # Bar chart
        fig.add_trace(
            go.Bar(
                x=components,
                y=contributions,
                marker_color=colors,
                name='Contribution to GDP Growth',
                hovertemplate='<b>%{x}</b><br>Contribution: %{y:.2f}%<extra></extra>'
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title="Growth Contribution Analysis",
            height=500,
            showlegend=False,
            template='plotly_white'
        )
        
        fig.update_xaxes(title_text="Expenditure Type", row=1, col=2)
        fig.update_yaxes(title_text="Contribution to GDP Growth (%)", row=1, col=2)
        
        return fig
    
    def create_correlation_heatmap(self, correlations_data):
        """
        Create correlation heatmap
        
        Args:
            correlations_data: Dictionary of correlation values with GDP
            
        Returns:
            plotly.graph_objects.Figure: Correlation heatmap
        """
        # Prepare data
        components = []
        correlations = []
        
        for component, correlation in correlations_data.items():
            if not np.isnan(correlation):
                components.append(self.expenditure_labels.get(component, component.title()))
                correlations.append(correlation)
        
        if not components:
            fig = go.Figure()
            fig.add_annotation(text="No correlation data available", 
                             xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig
        
        # Create correlation matrix (single row since we're correlating with GDP)
        correlation_matrix = np.array([correlations])
        
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix,
            x=components,
            y=['GDP Growth'],
            colorscale='RdBu',
            zmid=0,
            zmin=-1,
            zmax=1,
            hovertemplate='<b>%{x}</b> vs %{y}<br>Correlation: %{z:.3f}<extra></extra>',
            colorbar=dict(title="Correlation Coefficient")
        ))
        
        # Add correlation values as text
        for i, (component, corr) in enumerate(zip(components, correlations)):
            fig.add_annotation(
                x=i,
                y=0,
                text=f"{corr:.3f}",
                showarrow=False,
                font=dict(color="white" if abs(corr) > 0.5 else "black", size=12)
            )
        
        fig.update_layout(
            title="Correlation with GDP Growth",
            height=200,
            template='plotly_white'
        )
        
        return fig
    
    def create_regression_plot(self, df, x_column, y_column='gdp'):
        """
        Create regression scatter plot
        
        Args:
            df: Analysis dataframe
            x_column: Independent variable column
            y_column: Dependent variable column (default: gdp)
            
        Returns:
            plotly.graph_objects.Figure: Regression plot
        """
        if x_column not in df.columns or y_column not in df.columns:
            fig = go.Figure()
            fig.add_annotation(text="Data not available for regression plot", 
                             xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig
        
        # Calculate growth rates
        x_growth = df[x_column].pct_change().dropna() * 100
        y_growth = df[y_column].pct_change().dropna() * 100
        
        # Align series
        aligned_x, aligned_y = x_growth.align(y_growth, join='inner')
        
        if len(aligned_x) < 3:
            fig = go.Figure()
            fig.add_annotation(text="Insufficient data for regression plot", 
                             xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig
        
        # Create scatter plot
        x_label = self.expenditure_labels.get(x_column, x_column.title())
        y_label = self.expenditure_labels.get(y_column, y_column.title())
        
        fig = px.scatter(
            x=aligned_x,
            y=aligned_y,
            trendline="ols",
            labels={
                'x': f'{x_label} Growth Rate (%)',
                'y': f'{y_label} Growth Rate (%)'
            },
            title=f'{x_label} vs {y_label} Growth Rates'
        )
        
        fig.update_traces(
            marker=dict(size=8, color=self.color_palette.get(x_column, '#636EFA')),
            hovertemplate=f'<b>{x_label}</b>: %{{x:.2f}}%<br><b>{y_label}</b>: %{{y:.2f}}%<extra></extra>'
        )
        
        fig.update_layout(
            height=400,
            template='plotly_white'
        )
        
        return fig
    
    def create_dashboard_summary(self, results):
        """
        Create a summary dashboard with key metrics
        
        Args:
            results: Analysis results dictionary
            
        Returns:
            plotly.graph_objects.Figure: Dashboard summary
        """
        fig = make_subplots(
            rows=2, cols=2,
            specs=[
                [{"type": "indicator"}, {"type": "indicator"}],
                [{"type": "bar", "colspan": 2}, None]
            ],
            subplot_titles=('Primary Driver', 'GDP Growth Rate', 'Growth Contributions'),
            vertical_spacing=0.3
        )
        
        # Primary driver indicator
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=1,  # Placeholder value
                title={'text': results.get('primary_driver', 'N/A')},
                gauge={
                    'axis': {'range': [None, 1]},
                    'bar': {'color': "darkblue"},
                    'steps': [{'range': [0, 1], 'color': "lightgray"}],
                }
            ),
            row=1, col=1
        )
        
        # GDP growth rate indicator
        gdp_growth = results.get('gdp_growth_rate', 0)
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=gdp_growth,
                number={'suffix': "%"},
                delta={'reference': 3, 'relative': True},  # Assuming 3% as reference
                title={'text': "GDP Growth Rate"},
                gauge={
                    'axis': {'range': [-10, 10]},
                    'bar': {'color': "darkgreen" if gdp_growth > 0 else "darkred"},
                    'steps': [
                        {'range': [-10, 0], 'color': "lightcoral"},
                        {'range': [0, 10], 'color': "lightgreen"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 3
                    }
                }
            ),
            row=1, col=2
        )
        
        # Growth contributions bar chart
        if 'growth_contributions' in results:
            components = []
            contributions = []
            colors = []
            
            for component, contribution in results['growth_contributions'].items():
                if not np.isnan(contribution):
                    components.append(self.expenditure_labels.get(component, component.title()))
                    contributions.append(contribution)
                    colors.append(self.color_palette.get(component, '#636EFA'))
            
            if components:
                fig.add_trace(
                    go.Bar(
                        x=components,
                        y=contributions,
                        marker_color=colors,
                        name='Contributions',
                        hovertemplate='<b>%{x}</b><br>Contribution: %{y:.2f}%<extra></extra>'
                    ),
                    row=2, col=1
                )
        
        fig.update_layout(
            height=600,
            showlegend=False,
            template='plotly_white',
            title_text="Economic Analysis Dashboard"
        )
        
        return fig
    
    def create_forecast_plot(self, historical_data, forecast_results, component):
        """
        Create forecast visualization with historical data and predictions
        
        Args:
            historical_data: Historical time series data
            forecast_results: Forecast results dictionary
            component: Component name to plot
            
        Returns:
            plotly.graph_objects.Figure: Forecast plot
        """
        if component not in forecast_results:
            fig = go.Figure()
            fig.add_annotation(text=f"No forecast data available for {component}", 
                             xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig
        
        fig = go.Figure()
        
        # Historical data
        label = self.expenditure_labels.get(component, component.title())
        color = self.color_palette.get(component, '#636EFA')
        
        fig.add_trace(go.Scatter(
            x=historical_data.index,
            y=historical_data[component] if component in historical_data.columns else [],
            mode='lines+markers',
            name=f'Historical {label}',
            line=dict(color=color, width=2),
            marker=dict(size=4)
        ))
        
        # Forecast predictions
        forecast_data = forecast_results[component]
        fig.add_trace(go.Scatter(
            x=forecast_data['forecast_dates'],
            y=forecast_data['predictions'],
            mode='lines+markers',
            name=f'Forecast {label}',
            line=dict(color=color, width=2, dash='dash'),
            marker=dict(size=4, symbol='diamond')
        ))
        
        # Confidence intervals
        fig.add_trace(go.Scatter(
            x=forecast_data['forecast_dates'],
            y=forecast_data['confidence_upper'],
            mode='lines',
            line=dict(color=color, width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig.add_trace(go.Scatter(
            x=forecast_data['forecast_dates'],
            y=forecast_data['confidence_lower'],
            mode='lines',
            line=dict(color=color, width=0),
            fill='tonexty',
            fillcolor=f'rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.2)',
            name='Confidence Interval',
            hoverinfo='skip'
        ))
        
        fig.update_layout(
            title=f'{label} Forecast with Confidence Intervals',
            xaxis_title='Time Period',
            yaxis_title='Value',
            hovermode='x unified',
            height=500,
            template='plotly_white'
        )
        
        return fig
    
    def create_forecast_comparison(self, forecast_results):
        """
        Create comparison chart of forecasted growth rates for all components
        
        Args:
            forecast_results: Dictionary of forecast results
            
        Returns:
            plotly.graph_objects.Figure: Comparison chart
        """
        fig = go.Figure()
        
        components = []
        avg_forecast_growth = []
        colors = []
        
        for component, data in forecast_results.items():
            if 'predictions' in data and len(data['predictions']) > 0:
                # Calculate average forecasted growth rate
                predictions = data['predictions']
                if len(predictions) > 1:
                    growth_rates = [(predictions[i+1] - predictions[i]) / predictions[i] * 100 
                                   for i in range(len(predictions)-1) if predictions[i] != 0]
                    avg_growth = np.mean(growth_rates) if growth_rates else 0
                else:
                    avg_growth = 0
                
                components.append(self.expenditure_labels.get(component, component.title()))
                avg_forecast_growth.append(avg_growth)
                colors.append(self.color_palette.get(component, '#636EFA'))
        
        if components:
            fig.add_trace(go.Bar(
                x=components,
                y=avg_forecast_growth,
                marker_color=colors,
                name='Forecasted Growth Rate',
                hovertemplate='<b>%{x}</b><br>Avg Forecast Growth: %{y:.2f}%<extra></extra>'
            ))
        
        fig.update_layout(
            title='Forecasted Average Growth Rates by Component',
            xaxis_title='Expenditure Type',
            yaxis_title='Average Forecasted Growth Rate (%)',
            height=400,
            template='plotly_white'
        )
        
        return fig
    
    def create_accuracy_comparison(self, accuracy_results):
        """
        Create forecast accuracy comparison chart
        
        Args:
            accuracy_results: Dictionary of accuracy metrics
            
        Returns:
            plotly.graph_objects.Figure: Accuracy comparison chart
        """
        if not accuracy_results:
            fig = go.Figure()
            fig.add_annotation(text="No accuracy data available", 
                             xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig
        
        components = []
        mape_values = []
        rmse_values = []
        colors = []
        
        for component, metrics in accuracy_results.items():
            components.append(self.expenditure_labels.get(component, component.title()))
            mape_values.append(metrics['mape'])
            rmse_values.append(metrics['rmse'])
            colors.append(self.color_palette.get(component, '#636EFA'))
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Mean Absolute Percentage Error (MAPE)', 'Root Mean Square Error (RMSE)'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # MAPE chart
        fig.add_trace(
            go.Bar(
                x=components,
                y=mape_values,
                marker_color=colors,
                name='MAPE (%)',
                hovertemplate='<b>%{x}</b><br>MAPE: %{y:.2f}%<extra></extra>'
            ),
            row=1, col=1
        )
        
        # RMSE chart
        fig.add_trace(
            go.Bar(
                x=components,
                y=rmse_values,
                marker_color=colors,
                name='RMSE',
                showlegend=False,
                hovertemplate='<b>%{x}</b><br>RMSE: %{y:.2f}<extra></extra>'
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title='Forecast Accuracy Metrics',
            height=400,
            template='plotly_white'
        )
        
        fig.update_xaxes(title_text="Components", row=1, col=1)
        fig.update_xaxes(title_text="Components", row=1, col=2)
        fig.update_yaxes(title_text="MAPE (%)", row=1, col=1)
        fig.update_yaxes(title_text="RMSE", row=1, col=2)
        
        return fig
    
    def create_impulse_response_plot(self, irf_data, shock_variable, response_variable):
        """
        Create impulse response function plot
        
        Args:
            irf_data: Impulse response function data
            shock_variable: Variable that receives the shock
            response_variable: Variable whose response is measured
            
        Returns:
            plotly.graph_objects.Figure: Impulse response plot
        """
        try:
            import plotly.graph_objects as go
            import numpy as np
            
            # Get the impulse response values
            periods = range(len(irf_data))
            responses = irf_data
            
            fig = go.Figure()
            
            # Add the impulse response line
            fig.add_trace(go.Scatter(
                x=list(periods),
                y=responses,
                mode='lines+markers',
                name=f'Response of {response_variable} to {shock_variable}',
                line=dict(width=3)
            ))
            
            # Add zero line
            fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
            
            fig.update_layout(
                title=f'Impulse Response: {response_variable} response to {shock_variable} shock',
                xaxis_title='Periods',
                yaxis_title='Response',
                template='plotly_white',
                height=400
            )
            
            return fig
            
        except Exception as e:
            # Fallback empty plot
            fig = go.Figure()
            fig.add_annotation(
                text="Unable to generate impulse response plot",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
    
    def create_variance_decomposition_plot(self, variance_decomp_data, variable_names):
        """
        Create variance decomposition plot
        
        Args:
            variance_decomp_data: Variance decomposition data
            variable_names: List of variable names
            
        Returns:
            plotly.graph_objects.Figure: Variance decomposition plot
        """
        try:
            import plotly.graph_objects as go
            import numpy as np
            
            fig = go.Figure()
            
            # Create stacked bar chart for variance decomposition
            periods = range(len(variance_decomp_data))
            
            for i, var_name in enumerate(variable_names):
                if i < variance_decomp_data.shape[1]:
                    fig.add_trace(go.Bar(
                        x=list(periods),
                        y=variance_decomp_data[:, i],
                        name=var_name.title(),
                        text=[f'{val:.1f}%' for val in variance_decomp_data[:, i] * 100],
                        textposition='inside'
                    ))
            
            fig.update_layout(
                title='Variance Decomposition',
                xaxis_title='Periods',
                yaxis_title='Percentage of Variance',
                barmode='stack',
                template='plotly_white',
                height=500,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            return fig
            
        except Exception as e:
            # Fallback empty plot
            fig = go.Figure()
            fig.add_annotation(
                text="Unable to generate variance decomposition plot",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
