import dash
from dash import dcc, html, Input, Output, State, callback, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
import numpy as np
import requests
import json

# Function to get a fresh database connection
def get_db_connection():
    return sqlite3.connect('depos.db')

# Load data
def load_data():
    conn = get_db_connection()
    query = "SELECT * FROM lending_rates"
    df = pd.read_sql_query(query, conn)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    conn.close()
    return df

# Global variable to store BTCZAR price data cache
btczar_price_cache = {}

# Function to fetch BCTZAR price data from VALR API
def fetch_bctzar_price_data(start_time, end_time):
    # Create a cache key based on the date range
    cache_key = f"{start_time}_{end_time}"
    
    # Check if we already have this data in the cache
    if cache_key in btczar_price_cache:
        return btczar_price_cache[cache_key]
    
    try:
        # VALR API endpoint for historical market data
        url = "https://api.valr.com/v1/public/marketsummary"
        response = requests.get(url)
        
        if response.status_code == 200:
            all_market_data = response.json()
            # Extract BTCZAR data
            btczar_data = next((item for item in all_market_data if item.get('currencyPair') == 'BTCZAR'), None)
            
            if btczar_data:
                current_price = float(btczar_data.get('lastTradedPrice', 0))
                
                # For historical data, we would need to use a different endpoint with authentication
                # Since we don't have API keys, we'll simulate historical data based on current price
                # This is just for demonstration - in a real app, you would use the authenticated API
                
                # Create a dataframe with timestamps from our lending_rates data
                conn = get_db_connection()
                query = f"SELECT DISTINCT timestamp FROM lending_rates WHERE timestamp BETWEEN '{start_time}' AND '{end_time}' ORDER BY timestamp"
                timestamps_df = pd.read_sql_query(query, conn)
                conn.close()
                timestamps_df['timestamp'] = pd.to_datetime(timestamps_df['timestamp'])
                
                # Generate simulated prices (in a real app, fetch from VALR historical API)
                # Using a random walk starting from current price
                np.random.seed(42)  # For reproducibility
                price_variations = np.random.normal(0, current_price * 0.01, len(timestamps_df))
                cumulative_variations = np.cumsum(price_variations)
                
                timestamps_df['btczar_price'] = current_price + cumulative_variations
                
                # Store in cache
                btczar_price_cache[cache_key] = timestamps_df
                
                return timestamps_df
            else:
                print("BTCZAR pair not found in market data")
                return pd.DataFrame()
        else:
            print(f"API request failed with status code: {response.status_code}")
            return pd.DataFrame()
    except Exception as e:
        print(f"Error fetching BTCZAR price data: {str(e)}")
        return pd.DataFrame()

# Initialize the app with a dark theme
app = dash.Dash(__name__, 
                external_stylesheets=[dbc.themes.DARKLY],
                meta_tags=[{'name': 'viewport', 'content': 'width=device-width, initial-scale=1'}],
                suppress_callback_exceptions=True)

# App title
app.title = "ZAR & USDT Lending Rates Dashboard"

# Define colors for the dark theme
colors = {
    'background': '#111111',
    'text': '#7FDBFF',
    'grid': '#333333',
    'paper': '#222222'
}

# Custom plotly template for dark theme
dark_template = go.layout.Template(
    layout=dict(
        paper_bgcolor=colors['paper'],
        plot_bgcolor=colors['background'],
        font=dict(color='white'),
        title=dict(font=dict(color='white')),
        xaxis=dict(
            gridcolor=colors['grid'],
            zerolinecolor=colors['grid'],
            tickfont=dict(color='white')
        ),
        yaxis=dict(
            gridcolor=colors['grid'],
            zerolinecolor=colors['grid'],
            tickfont=dict(color='white')
        ),
        legend=dict(font=dict(color='white'))
    )
)

# Load initial data
df = load_data()
available_currencies = sorted(df['currency'].unique())
default_currencies = ['ZAR', 'USDT']

# Create tabs for the dashboard
tabs_styles = {
    'height': '44px'
}
tab_style = {
    'borderBottom': '1px solid #d6d6d6',
    'padding': '6px',
    'fontWeight': 'bold',
    'backgroundColor': colors['paper'],
    'color': 'white'
}
tab_selected_style = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': '#119DFF',
    'color': 'white',
    'padding': '6px'
}

# App layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("ZAR & USDT Lending Rates Dashboard", 
                    className="text-center text-primary mb-4",
                    style={'margin-top': '20px'})
        ], width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            dcc.Tabs(id='tabs', value='tab-charts', children=[
                dcc.Tab(label='Charts', value='tab-charts', style=tab_style, selected_style=tab_selected_style),
                dcc.Tab(label='Hourly Data', value='tab-hourly-data', style=tab_style, selected_style=tab_selected_style),
                dcc.Tab(label='BTCZAR Correlation', value='tab-btczar', style=tab_style, selected_style=tab_selected_style),
                dcc.Tab(label='Interest to BTC', value='tab-interest-btc', style=tab_style, selected_style=tab_selected_style),
            ], style=tabs_styles)
        ], width=12)
    ]),
    
    html.Div(id='tabs-content'),
    
    dbc.Row([
        dbc.Col([
            html.P("Data last updated: " + df['timestamp'].max().strftime('%Y-%m-%d %H:%M:%S'),
                   className="text-center text-muted mb-4")
        ], width=12)
    ])
], fluid=True, style={"background-color": colors['background'], "min-height": "100vh", "padding": "20px"})

# Callback to render the selected tab content
@app.callback(
    Output('tabs-content', 'children'),
    [Input('tabs', 'value')]
)
def render_tab_content(tab):
    if tab == 'tab-charts':
        return charts_layout()
    elif tab == 'tab-hourly-data':
        return hourly_data_layout()
    elif tab == 'tab-btczar':
        return btczar_correlation_layout()
    elif tab == 'tab-interest-btc':
        return interest_btc_layout()

# Define the charts layout (original dashboard content)
def charts_layout():
    return html.Div([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Filters", className="text-primary"),
                    dbc.CardBody([
                        html.P("Select Date Range:"),
                        dcc.DatePickerRange(
                            id='date-picker-range',
                            min_date_allowed=df['timestamp'].min().date(),
                            max_date_allowed=df['timestamp'].max().date(),
                            start_date=df['timestamp'].min().date(), #(df['timestamp'].max() - timedelta(days=7)).date(),
                            end_date=df['timestamp'].max().date(),
                            style={'color': colors['text']}
                        ),
                        html.P("Select Currencies:", className="mt-3"),
                        dcc.Dropdown(
                            id='currency-dropdown',
                            options=[{'label': curr, 'value': curr} for curr in available_currencies],
                            value=default_currencies,
                            multi=True,
                            style={'color': 'black'}
                        ),
                        html.Div([
                            dbc.Button("Update Dashboard", id="update-button", color="primary", className="mt-2")
                        ], className="d-grid gap-2")
                    ])
                ], className="mb-4", style={"background-color": colors['paper']})
            ], width=12, lg=3),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Rate Trends Over Time", className="text-primary"),
                    dbc.CardBody([
                        dcc.Graph(id='time-series-chart', style={'height': '400px'})
                    ])
                ], className="mb-4", style={"background-color": colors['paper']})
            ], width=12, lg=9)
        ]),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Rate Comparison", className="text-primary"),
                    dbc.CardBody([
                        dcc.Graph(id='rate-comparison', style={'height': '400px'})
                    ])
                ], className="mb-4", style={"background-color": colors['paper']})
            ], width=12, lg=6),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Rate Distribution", className="text-primary"),
                    dbc.CardBody([
                        dcc.Graph(id='rate-distribution', style={'height': '400px'})
                    ])
                ], className="mb-4", style={"background-color": colors['paper']})
            ], width=12, lg=6)
        ]),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Recent Rate Changes", className="text-primary"),
                    dbc.CardBody([
                        html.Div(id='recent-changes-table')
                    ])
                ], className="mb-4", style={"background-color": colors['paper']})
            ], width=12)
        ])
    ])

# Define the hourly data layout
def hourly_data_layout():
    return html.Div([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Hourly Data Filters", className="text-primary"),
                    dbc.CardBody([
                        html.P("Select Date:"),
                        dcc.DatePickerSingle(
                            id='hourly-date-picker',
                            min_date_allowed=df['timestamp'].min().date(),
                            max_date_allowed=df['timestamp'].max().date(),
                            date=df['timestamp'].max().date(),
                            style={'color': colors['text']}
                        ),
                        html.P("Select Currency:", className="mt-3"),
                        dcc.Dropdown(
                            id='hourly-currency-dropdown',
                            options=[{'label': curr, 'value': curr} for curr in ['ZAR', 'USDT']],
                            value='ZAR',
                            style={'color': 'black'}
                        ),
                        html.Div([
                            dbc.Button("Update Table", id="update-hourly-button", color="primary", className="mt-2")
                        ], className="d-grid gap-2")
                    ])
                ], className="mb-4", style={"background-color": colors['paper']})
            ], width=12)
        ]),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Hourly Lending Rates", className="text-primary"),
                    dbc.CardBody([
                        html.Div(id='hourly-data-table')
                    ])
                ], className="mb-4", style={"background-color": colors['paper']})
            ], width=12)
        ])
    ])

# Define the BTCZAR correlation layout
def btczar_correlation_layout():
    return html.Div([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("BTCZAR Correlation Filters", className="text-primary"),
                    dbc.CardBody([
                        html.P("Select Date Range:"),
                        dcc.DatePickerRange(
                            id='btczar-date-picker-range',
                            min_date_allowed=df['timestamp'].min().date(),
                            max_date_allowed=df['timestamp'].max().date(),
                            start_date=df['timestamp'].min().date(), # (df['timestamp'].max() - timedelta(days=7)).date(),
                            end_date=df['timestamp'].max().date(),
                            style={'color': colors['text']}
                        ),
                        html.P("Select Currency:", className="mt-3"),
                        dcc.Dropdown(
                            id='btczar-currency-dropdown',
                            options=[{'label': curr, 'value': curr} for curr in ['ZAR', 'USDT']],
                            value='ZAR',
                            style={'color': 'black'}
                        ),
                        html.Div([
                            dbc.Button("Update Correlation", id="update-btczar-button", color="primary", className="mt-2")
                        ], className="d-grid gap-2")
                    ])
                ], className="mb-4", style={"background-color": colors['paper']})
            ], width=12)
        ]),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("BTCZAR Price vs Lending Rate", className="text-primary"),
                    dbc.CardBody([
                        dcc.Graph(id='btczar-correlation-chart', style={'height': '400px'})
                    ])
                ], className="mb-4", style={"background-color": colors['paper']})
            ], width=12)
        ]),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("BTCZAR and Lending Rate Data", className="text-primary"),
                    dbc.CardBody([
                        html.Div(id='btczar-data-table')
                    ])
                ], className="mb-4", style={"background-color": colors['paper']})
            ], width=12)
        ])
    ])

# Callback to update charts in the first tab
@app.callback(
    [Output('time-series-chart', 'figure'),
     Output('rate-comparison', 'figure'),
     Output('rate-distribution', 'figure'),
     Output('recent-changes-table', 'children')],
    [Input('update-button', 'n_clicks')],
    [State('date-picker-range', 'start_date'),
     State('date-picker-range', 'end_date'),
     State('currency-dropdown', 'value')]
)
def update_charts(n_clicks, start_date, end_date, selected_currencies):
    # Get fresh data for the charts
    conn = get_db_connection()
    currencies_str = "', '".join(selected_currencies)
    query = f"""
    SELECT * FROM lending_rates 
    WHERE timestamp BETWEEN '{start_date}' AND '{end_date} 23:59:59'
    AND currency IN ('{currencies_str}')
    """
    filtered_df = pd.read_sql_query(query, conn)
    filtered_df['timestamp'] = pd.to_datetime(filtered_df['timestamp'])
    conn.close()
    
    # If no data is available after filtering
    if filtered_df.empty:
        filtered_df = df[
            (df['timestamp'] >= start_date) & 
            (df['timestamp'] <= end_date) & 
            (df['currency'].isin(selected_currencies))
        ]
    
    # If no data is available after filtering
    if filtered_df.empty:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            template=dark_template,
            title="No data available for the selected filters",
            xaxis=dict(title=""),
            yaxis=dict(title="")
        )
        empty_table = html.Div("No data available for the selected filters")
        return empty_fig, empty_fig, empty_fig, empty_table
    
    # Focus only on next_annual field
    rate_cols = ['next_annual']
    rate_title = 'Annual Rates (%)'
    
    # 1. Time Series Chart
    time_series_fig = go.Figure()
    
    for currency in selected_currencies:
        currency_df = filtered_df[filtered_df['currency'] == currency]
        if not currency_df.empty:
            time_series_fig.add_trace(go.Scatter(
                x=currency_df['timestamp'],
                y=currency_df[rate_cols[0]],  # next_annual
                mode='lines',
                name=currency
            ))
    
    time_series_fig.update_layout(
        template=dark_template,
        title=f"Lending {rate_title} Over Time",
        xaxis_title="Date",
        yaxis_title=rate_title,
        legend_title="Currency",
        hovermode="x unified"
    )
    
    # 2. Rate Comparison Chart (Bar Chart)
    # Get the latest rates for each currency
    latest_rates = filtered_df.sort_values('timestamp').groupby('currency').last().reset_index()
    
    comparison_fig = go.Figure()
    
    comparison_fig.add_trace(go.Bar(
        x=latest_rates['currency'],
        y=latest_rates['next_annual'],
        name='Next Annual Rate'
    ))
    
    comparison_fig.update_layout(
        template=dark_template,
        title=f"Latest {rate_title} Comparison",
        xaxis_title="Currency",
        yaxis_title=rate_title,
        barmode='group',
        legend_title="Rate Type"
    )
    
    # 3. Rate Distribution (Box Plot)
    distribution_fig = go.Figure()
    
    for currency in selected_currencies:
        currency_df = filtered_df[filtered_df['currency'] == currency]
        if not currency_df.empty:
            distribution_fig.add_trace(go.Box(
                y=currency_df[rate_cols[0]],  # next_annual
                name=currency,
                boxpoints='all',
                jitter=0.3,
                pointpos=-1.8
            ))
    
    distribution_fig.update_layout(
        template=dark_template,
        title=f"{rate_title} Distribution",
        xaxis_title="Currency",
        yaxis_title=rate_title
    )
    
    # 4. Recent Changes Table
    # Get the most recent data points and calculate the change
    recent_df = filtered_df.sort_values('timestamp').groupby('currency').tail(2)
    
    # Calculate rate changes
    change_data = []
    for currency in selected_currencies:
        currency_recent = recent_df[recent_df['currency'] == currency]
        if len(currency_recent) >= 2:
            old_rate = currency_recent.iloc[0][rate_cols[0]]
            new_rate = currency_recent.iloc[1][rate_cols[0]]
            change = new_rate - old_rate
            pct_change = (change / old_rate * 100) if old_rate != 0 else float('inf')
            
            change_data.append({
                'Currency': currency,
                'Previous Rate': f"{old_rate:.2f}%",
                'Current Rate': f"{new_rate:.2f}%",
                'Change': f"{change:.2f}%",
                'Percent Change': f"{pct_change:.2f}%",
                'Last Updated': currency_recent.iloc[1]['timestamp'].strftime('%Y-%m-%d %H:%M')
            })
    
    # Create the table
    if change_data:
        table_header = [
            html.Thead(html.Tr([
                html.Th("Currency"),
                html.Th("Previous Rate"),
                html.Th("Current Rate"),
                html.Th("Change"),
                html.Th("% Change"),
                html.Th("Last Updated")
            ]))
        ]
        
        rows = []
        for data in change_data:
            # Determine cell color based on change direction
            if data['Percent Change'] != 'inf%':
                pct_change_value = float(data['Percent Change'].replace('%', ''))
                if pct_change_value > 0:
                    style = {'color': '#77dd77'}  # green for positive
                elif pct_change_value < 0:
                    style = {'color': '#ff6961'}  # red for negative
                else:
                    style = {}
            else:
                style = {}
                
            rows.append(html.Tr([
                html.Td(data['Currency']),
                html.Td(data['Previous Rate']),
                html.Td(data['Current Rate']),
                html.Td(data['Change']),
                html.Td(data['Percent Change'], style=style),
                html.Td(data['Last Updated'])
            ]))
        
        table_body = [html.Tbody(rows)]
        table = dbc.Table(table_header + table_body, striped=True, bordered=True, hover=True, responsive=True)
    else:
        table = html.Div("No recent changes data available for the selected filters")
    
    return time_series_fig, comparison_fig, distribution_fig, table

# Callback to update hourly data table
@app.callback(
    Output('hourly-data-table', 'children'),
    [Input('update-hourly-button', 'n_clicks')],
    [State('hourly-date-picker', 'date'),
     State('hourly-currency-dropdown', 'value')]
)
def update_hourly_data(n_clicks, selected_date, selected_currency):
    if not selected_date or not selected_currency:
        return html.Div("Please select a date and currency")
    
    # Query to get hourly data for the selected date and currency
    query = f"""
    SELECT 
        timestamp,
        currency,
        prev_rate,
        next_rate,
        borrow_rate,
        prev_annual,
        next_annual,
        borrow_annual
    FROM 
        lending_rates
    WHERE 
        date(timestamp) = '{selected_date}'
        AND currency = '{selected_currency}'
    ORDER BY 
        timestamp
    """
    
    conn = get_db_connection()
    hourly_df = pd.read_sql_query(query, conn)
    conn.close()
    
    if hourly_df.empty:
        return html.Div("No data available for the selected date and currency")
    
    # Format the timestamp for better readability
    hourly_df['timestamp'] = pd.to_datetime(hourly_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Format the numeric columns
    for col in ['prev_rate', 'next_rate', 'borrow_rate']:
        hourly_df[col] = hourly_df[col].apply(lambda x: f"{x:.6f}")
    
    for col in ['prev_annual', 'next_annual', 'borrow_annual']:
        hourly_df[col] = hourly_df[col].apply(lambda x: f"{x:.2f}%")
    
    # Create the data table
    table = dash_table.DataTable(
        id='hourly-table',
        columns=[
            {'name': 'Timestamp', 'id': 'timestamp'},
            {'name': 'Currency', 'id': 'currency'},
            {'name': 'Previous Rate', 'id': 'prev_rate'},
            {'name': 'Next Rate', 'id': 'next_rate'},
            {'name': 'Borrow Rate', 'id': 'borrow_rate'},
            {'name': 'Previous Annual', 'id': 'prev_annual'},
            {'name': 'Next Annual', 'id': 'next_annual'},
            {'name': 'Borrow Annual', 'id': 'borrow_annual'}
        ],
        data=hourly_df.to_dict('records'),
        style_header={
            'backgroundColor': colors['grid'],
            'color': 'white',
            'fontWeight': 'bold'
        },
        style_cell={
            'backgroundColor': colors['paper'],
            'color': 'white',
            'textAlign': 'left',
            'padding': '10px',
            'whiteSpace': 'normal',
            'height': 'auto',
        },
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': colors['background']
            }
        ],
        page_size=24,  # Show all hours in a day
        sort_action='native',
        filter_action='native',
        style_table={'overflowX': 'auto'}
    )
    
    return table

# Callback to update BTCZAR correlation
@app.callback(
    [Output('btczar-correlation-chart', 'figure'),
     Output('btczar-data-table', 'children')],
    [Input('update-btczar-button', 'n_clicks')],
    [State('btczar-date-picker-range', 'start_date'),
     State('btczar-date-picker-range', 'end_date'),
     State('btczar-currency-dropdown', 'value')]
)
def update_btczar_correlation(n_clicks, start_date, end_date, selected_currency):
    if not start_date or not end_date or not selected_currency:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            template=dark_template,
            title="Please select date range and currency"
        )
        return empty_fig, html.Div("Please select date range and currency")
    
    # Get lending rates data for the selected period and currency
    query = f"""
    SELECT 
        timestamp,
        currency,
        next_annual
    FROM 
        lending_rates
    WHERE 
        timestamp BETWEEN '{start_date}' AND '{end_date} 23:59:59'
        AND currency = '{selected_currency}'
    ORDER BY 
        timestamp
    """
    
    conn = get_db_connection()
    lending_df = pd.read_sql_query(query, conn)
    conn.close()
    lending_df['timestamp'] = pd.to_datetime(lending_df['timestamp'])
    
    if lending_df.empty:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            template=dark_template,
            title="No lending data available for the selected period"
        )
        return empty_fig, html.Div("No lending data available for the selected period")
    
    # Fetch BTCZAR price data for the same period
    btczar_df = fetch_bctzar_price_data(start_date, end_date)
    
    if btczar_df.empty:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            template=dark_template,
            title="Could not fetch BTCZAR price data"
        )
        return empty_fig, html.Div("Could not fetch BTCZAR price data")
    
    # Merge the datasets on timestamp
    merged_df = pd.merge_asof(
        lending_df.sort_values('timestamp'),
        btczar_df.sort_values('timestamp'),
        on='timestamp',
        direction='nearest'
    )
    
    # Create the correlation chart
    fig = go.Figure()
    
    # Add lending rate line
    fig.add_trace(go.Scatter(
        x=merged_df['timestamp'],
        y=merged_df['next_annual'],
        mode='lines',
        name=f'{selected_currency} Lending Rate (%)',
        yaxis='y'
    ))
    
    # Add BTCZAR price line on secondary y-axis
    fig.add_trace(go.Scatter(
        x=merged_df['timestamp'],
        y=merged_df['btczar_price'],
        mode='lines',
        name='BTCZAR Price',
        yaxis='y2'
    ))
    
    # Set up the layout with two y-axes
    fig.update_layout(
        template=dark_template,
        title=f'BTCZAR Price vs {selected_currency} Lending Rate',
        xaxis=dict(title='Date'),
        yaxis=dict(
            title=f'{selected_currency} Lending Rate (%)',
            side='left'
        ),
        yaxis2=dict(
            title='BTCZAR Price (ZAR)',
            side='right',
            overlaying='y',
            showgrid=False
        ),
        legend=dict(x=0.01, y=0.99),
        hovermode='x unified'
    )
    
    # Calculate correlation coefficient
    correlation = merged_df['next_annual'].corr(merged_df['btczar_price'])
    
    # Add correlation annotation
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.5, y=0.01,
        text=f"Correlation: {correlation:.2f}",
        showarrow=False,
        font=dict(color="white", size=14),
        bgcolor="rgba(0,0,0,0.5)",
        bordercolor="white",
        borderwidth=1,
        borderpad=4
    )
    
    # Create the data table
    # Format the data for display
    table_df = merged_df.copy()
    table_df['timestamp'] = table_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    table_df['next_annual'] = table_df['next_annual'].apply(lambda x: f"{x:.2f}%")
    table_df['btczar_price'] = table_df['btczar_price'].apply(lambda x: f"R {x:.2f}")
    table_df = table_df.rename(columns={
        'timestamp': 'Timestamp',
        'currency': 'Currency',
        'next_annual': 'Lending Rate',
        'btczar_price': 'BTCZAR Price'
    })
    
    table = dash_table.DataTable(
        id='btczar-table',
        columns=[
            {'name': col, 'id': col} for col in table_df.columns
        ],
        data=table_df.to_dict('records'),
        style_header={
            'backgroundColor': colors['grid'],
            'color': 'white',
            'fontWeight': 'bold'
        },
        style_cell={
            'backgroundColor': colors['paper'],
            'color': 'white',
            'textAlign': 'left',
            'padding': '10px',
            'whiteSpace': 'normal',
            'height': 'auto',
        },
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': colors['background']
            }
        ],
        page_size=10,
        sort_action='native',
        filter_action='native',
        style_table={'overflowX': 'auto'}
    )
    
    return fig, table

# Define the Interest to BTC layout
def interest_btc_layout():
    return html.Div([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Interest to BTC Calculator", className="text-primary"),
                    dbc.CardBody([
                        html.P("Select Date Range:"),
                        dcc.DatePickerRange(
                            id='interest-btc-date-picker-range',
                            min_date_allowed=df['timestamp'].min().date(),
                            max_date_allowed=df['timestamp'].max().date(),
                            start_date=df['timestamp'].min().date(), #(df['timestamp'].max() - timedelta(days=7)).date(),
                            end_date=df['timestamp'].max().date(),
                            style={'color': colors['text']}
                        ),
                        html.P("Enter ZAR Amount:", className="mt-3"),
                        dbc.Input(
                            id='zar-amount-input',
                            type='number',
                            value=1000000,
                            min=1,
                            step=1000,
                            style={'color': 'black'}
                        ),
                        html.Div([
                            dbc.Button("Calculate Interest to BTC", id="calculate-interest-btc-button", color="primary", className="mt-2")
                        ], className="d-grid gap-2")
                    ])
                ], className="mb-4", style={"background-color": colors['paper']})
            ], width=12)
        ]),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Hourly Interest Earned and BTC Equivalent", className="text-primary"),
                    dbc.CardBody([
                        html.Div(id='interest-btc-table')
                    ])
                ], className="mb-4", style={"background-color": colors['paper']})
            ], width=12)
        ]),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Interest to BTC Summary", className="text-primary"),
                    dbc.CardBody([
                        html.Div(id='interest-btc-summary')
                    ])
                ], className="mb-4", style={"background-color": colors['paper']})
            ], width=12)
        ])
    ])

# Callback to update interest to BTC calculations
@app.callback(
    [Output('interest-btc-table', 'children'),
     Output('interest-btc-summary', 'children')],
    [Input('calculate-interest-btc-button', 'n_clicks')],
    [State('interest-btc-date-picker-range', 'start_date'),
     State('interest-btc-date-picker-range', 'end_date'),
     State('zar-amount-input', 'value')]
)
def update_interest_btc_calculations(n_clicks, start_date, end_date, zar_amount):
    if not start_date or not end_date or not zar_amount:
        return html.Div("Please select date range and enter ZAR amount"), html.Div("")
    
    # Get ZAR lending rates for the selected period
    conn = get_db_connection()
    query = f"""
    SELECT 
        timestamp,
        next_annual
    FROM 
        lending_rates
    WHERE 
        timestamp BETWEEN '{start_date}' AND '{end_date} 23:59:59'
        AND currency = 'ZAR'
    ORDER BY 
        timestamp
    """
    
    zar_rates_df = pd.read_sql_query(query, conn)
    conn.close()
    
    if zar_rates_df.empty:
        return html.Div("No ZAR lending rate data available for the selected period"), html.Div("")
    
    # Fetch BTCZAR price data for the same period
    btczar_df = fetch_bctzar_price_data(start_date, end_date)
    
    if btczar_df.empty:
        return html.Div("Could not fetch BTCZAR price data"), html.Div("")
    
    # Merge the datasets on timestamp
    zar_rates_df['timestamp'] = pd.to_datetime(zar_rates_df['timestamp'])
    merged_df = pd.merge_asof(
        zar_rates_df.sort_values('timestamp'),
        btczar_df.sort_values('timestamp'),
        on='timestamp',
        direction='nearest'
    )
    
    # Calculate hourly interest and BTC equivalent
    merged_df['hourly_interest_rate'] = merged_df['next_annual'] / 100 / 365 / 24  # Convert annual % to hourly rate
    merged_df['hourly_interest_zar'] = zar_amount * merged_df['hourly_interest_rate']
    merged_df['hourly_interest_btc'] = merged_df['hourly_interest_zar'] / merged_df['btczar_price']
    
    # Calculate cumulative values
    merged_df['cumulative_interest_zar'] = merged_df['hourly_interest_zar'].cumsum()
    merged_df['cumulative_interest_btc'] = merged_df['hourly_interest_btc'].cumsum()
    
    # Format the data for display
    table_df = merged_df.copy()
    table_df['timestamp'] = table_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    table_df['next_annual'] = table_df['next_annual'].apply(lambda x: f"{x:.2f}%")
    table_df['btczar_price'] = table_df['btczar_price'].apply(lambda x: f"R {x:,.2f}")
    table_df['hourly_interest_zar'] = table_df['hourly_interest_zar'].apply(lambda x: f"R {x:,.2f}")
    table_df['hourly_interest_btc'] = table_df['hourly_interest_btc'].apply(lambda x: f"{x:.8f} BTC")
    table_df['cumulative_interest_zar'] = table_df['cumulative_interest_zar'].apply(lambda x: f"R {x:,.2f}")
    table_df['cumulative_interest_btc'] = table_df['cumulative_interest_btc'].apply(lambda x: f"{x:.8f} BTC")
    
    # Rename columns for display
    display_df = table_df[['timestamp', 'next_annual', 'btczar_price', 'hourly_interest_zar', 'hourly_interest_btc', 
                          'cumulative_interest_zar', 'cumulative_interest_btc']].rename(columns={
        'timestamp': 'Timestamp',
        'next_annual': 'Annual Rate',
        'btczar_price': 'BTCZAR Price',
        'hourly_interest_zar': 'Hourly Interest (ZAR)',
        'hourly_interest_btc': 'Hourly Interest (BTC)',
        'cumulative_interest_zar': 'Cumulative Interest (ZAR)',
        'cumulative_interest_btc': 'Cumulative Interest (BTC)'
    })
    
    # Create the data table
    table = dash_table.DataTable(
        id='interest-btc-table-data',
        columns=[{'name': col, 'id': col} for col in display_df.columns],
        data=display_df.to_dict('records'),
        style_header={
            'backgroundColor': colors['grid'],
            'color': 'white',
            'fontWeight': 'bold'
        },
        style_cell={
            'backgroundColor': colors['paper'],
            'color': 'white',
            'textAlign': 'left',
            'padding': '10px',
            'whiteSpace': 'normal',
            'height': 'auto',
        },
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': colors['background']
            }
        ],
        page_size=10,
        sort_action='native',
        filter_action='native',
        style_table={'overflowX': 'auto'}
    )
    
    # Create summary statistics
    total_hours = len(merged_df)
    total_interest_zar = merged_df['hourly_interest_zar'].sum()
    total_interest_btc = merged_df['hourly_interest_btc'].sum()
    avg_hourly_interest_zar = total_interest_zar / total_hours if total_hours > 0 else 0
    avg_hourly_interest_btc = total_interest_btc / total_hours if total_hours > 0 else 0
    avg_btczar_price = merged_df['btczar_price'].mean()
    avg_annual_rate = merged_df['next_annual'].mean()
    
    summary_cards = dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Total Period", className="text-center"),
                dbc.CardBody([
                    html.H4(f"{total_hours} hours", className="text-center")
                ])
            ], color="primary", inverse=True)
        ], width=12, lg=3),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Total Interest Earned", className="text-center"),
                dbc.CardBody([
                    html.H4(f"R {total_interest_zar:,.2f}", className="text-center"),
                    html.H6(f"{total_interest_btc:.8f} BTC", className="text-center text-muted")
                ])
            ], color="success", inverse=True)
        ], width=12, lg=3),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Average Hourly Interest", className="text-center"),
                dbc.CardBody([
                    html.H4(f"R {avg_hourly_interest_zar:,.2f}", className="text-center"),
                    html.H6(f"{avg_hourly_interest_btc:.8f} BTC", className="text-center text-muted")
                ])
            ], color="info", inverse=True)
        ], width=12, lg=3),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Average Values", className="text-center"),
                dbc.CardBody([
                    html.H5(f"Rate: {avg_annual_rate:.2f}%", className="text-center"),
                    html.H5(f"BTCZAR: R {avg_btczar_price:,.2f}", className="text-center")
                ])
            ], color="warning", inverse=True)
        ], width=12, lg=3),
    ], className="mb-4")
    
    return table, summary_cards

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
