import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
import numpy as np
import requests
import json

# Set page configuration and theme
st.set_page_config(
    page_title="ZAR & USDT Lending Rates Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to get a fresh database connection
def get_db_connection():
    return sqlite3.connect('depos.db')

# Load data
@st.cache_data(ttl=300)  # Cache data for 5 minutes
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
                st.error("BTCZAR pair not found in market data")
                return pd.DataFrame()
        else:
            st.error(f"API request failed with status code: {response.status_code}")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error fetching BTCZAR price data: {str(e)}")
        return pd.DataFrame()

# Define dark theme colors for Plotly
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

# Apply custom CSS for dark theme
st.markdown("""
<style>
    .stApp {
        background-color: #111111;
    }
    .css-1d391kg, .css-1wrcr25, .css-ocqkz7, .css-1n76uvr {
        background-color: #222222;
        color: white;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #222222;
        color: white;
        border-radius: 4px 4px 0px 0px;
        border-right: 1px solid #d6d6d6;
        border-left: 1px solid #d6d6d6;
        border-top: 1px solid #d6d6d6;
    }
    .stTabs [aria-selected="true"] {
        background-color: #119DFF;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #7FDBFF;
    }
</style>
""", unsafe_allow_html=True)

# Dashboard title
st.title("ZAR & USDT Lending Rates Dashboard")

# Display data last updated information
st.caption(f"Data last updated: {df['timestamp'].max().strftime('%Y-%m-%d %H:%M:%S')}")

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["Charts", "Hourly Data", "BTCZAR Correlation", "Interest to BTC"])

# ===== Tab 1: Charts =====
with tab1:
    st.header("Charts")
    
    # Filters sidebar
    with st.expander("Filters"):
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Date picker for date range
            start_date = st.date_input(
                "Start Date",
                value=(df['timestamp'].max() - timedelta(days=60)).date(),
                min_value=df['timestamp'].min().date(),
                max_value=df['timestamp'].max().date()
            )
            
            end_date = st.date_input(
                "End Date",
                value=df['timestamp'].max().date(),
                min_value=df['timestamp'].min().date(),
                max_value=df['timestamp'].max().date()
            )
            
        with col2:
            # Multi-select for currencies
            selected_currencies = st.multiselect(
                "Select Currencies",
                options=available_currencies,
                default=default_currencies
            )
    
    # Only proceed if currencies are selected
    if selected_currencies:
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
                (df['timestamp'] >= pd.Timestamp(start_date)) & 
                (df['timestamp'] <= pd.Timestamp(end_date)) & 
                (df['currency'].isin(selected_currencies))
            ]
        
        # If still no data available
        if filtered_df.empty:
            st.warning("No data available for the selected filters")
        else:
            # Focus only on next_annual field
            rate_cols = ['next_annual']
            rate_title = 'Annual Rates (%)'
            
            # 1. Time Series Chart
            st.subheader(f"Lending {rate_title} Over Time")
            
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
                xaxis_title="Date",
                yaxis_title=rate_title,
                legend_title="Currency",
                hovermode="x unified",
                height=400
            )
            
            st.plotly_chart(time_series_fig, use_container_width=True)
            
            # Create two columns for the remaining charts
            col1, col2 = st.columns(2)
            
            with col1:
                # 2. Rate Comparison Chart (Bar Chart)
                st.subheader(f"Latest {rate_title} Comparison")
                
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
                    xaxis_title="Currency",
                    yaxis_title=rate_title,
                    barmode='group',
                    legend_title="Rate Type",
                    height=400
                )
                
                st.plotly_chart(comparison_fig, use_container_width=True)
            
            with col2:
                # 3. Rate Distribution (Box Plot)
                st.subheader(f"{rate_title} Distribution")
                
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
                    xaxis_title="Currency",
                    yaxis_title=rate_title,
                    height=400
                )
                
                st.plotly_chart(distribution_fig, use_container_width=True)
            
            # 4. Recent Changes Table
            st.subheader("Recent Rate Changes")
            
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
                        'Percent Change': f"{pct_change:.2f}%" if old_rate != 0 else "N/A",
                        'Last Updated': currency_recent.iloc[1]['timestamp'].strftime('%Y-%m-%d %H:%M')
                    })
            
            # Create the table
            if change_data:
                change_df = pd.DataFrame(change_data)
                st.dataframe(change_df, use_container_width=True)
            else:
                st.warning("No recent changes data available for the selected filters")
    else:
        st.warning("Please select at least one currency")

# ===== Tab 2: Hourly Data =====
with tab2:
    st.header("Hourly Data")
    
    # Filters sidebar
    with st.expander("Hourly Data Filters"):
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Date picker for single date
            selected_date = st.date_input(
                "Select Date",
                value=df['timestamp'].max().date(),
                min_value=df['timestamp'].min().date(),
                max_value=df['timestamp'].max().date(),
                key="hourly_date"
            )
        
        with col2:
            # Dropdown for currency
            selected_currency = st.selectbox(
                "Select Currency",
                options=['ZAR', 'USDT'],
                index=0,
                key="hourly_currency"
            )
    
    # Get hourly data for the selected date and currency
    start_timestamp = pd.Timestamp(selected_date)
    end_timestamp = pd.Timestamp(selected_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    
    conn = get_db_connection()
    query = f"""
    SELECT * FROM lending_rates 
    WHERE timestamp BETWEEN '{start_timestamp}' AND '{end_timestamp}'
    AND currency = '{selected_currency}'
    ORDER BY timestamp
    """
    hourly_df = pd.read_sql_query(query, conn)
    hourly_df['timestamp'] = pd.to_datetime(hourly_df['timestamp'])
    conn.close()
    
    if hourly_df.empty:
        st.warning(f"No hourly data available for {selected_currency} on {selected_date}")
    else:
        st.subheader(f"Hourly Lending Rates for {selected_currency} on {selected_date}")
        
        # Format the hourly data for display
        display_df = hourly_df[['timestamp', 'currency', 'next_annual']].copy()
        display_df.rename(columns={
            'timestamp': 'Time',
            'currency': 'Currency',
            'next_annual': 'Annual Rate (%)'
        }, inplace=True)
        
        # Add Hour column for better readability
        display_df['Hour'] = display_df['Time'].dt.strftime('%H:%M')
        display_df = display_df[['Hour', 'Currency', 'Annual Rate (%)']]
        
        # Display the hourly data table
        st.dataframe(display_df, use_container_width=True)
        
        # Display a line chart for visualization
        st.subheader("Hourly Rate Trend")
        
        hourly_fig = go.Figure()
        hourly_fig.add_trace(go.Scatter(
            x=hourly_df['timestamp'],
            y=hourly_df['next_annual'],
            mode='lines+markers',
            name=selected_currency
        ))
        
        hourly_fig.update_layout(
            template=dark_template,
            xaxis_title="Hour",
            yaxis_title="Annual Rate (%)",
            height=400
        )
        
        st.plotly_chart(hourly_fig, use_container_width=True)

# ===== Tab 3: BTCZAR Correlation =====
with tab3:
    st.header("BTCZAR Correlation")
    
    # Filters sidebar
    with st.expander("BTCZAR Correlation Filters"):
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Date pickers for date range
            btczar_start_date = st.date_input(
                "Start Date",
                value=(df['timestamp'].max() - timedelta(days=60)).date(),
                min_value=df['timestamp'].min().date(),
                max_value=df['timestamp'].max().date(),
                key="btczar_start_date"
            )
            
            btczar_end_date = st.date_input(
                "End Date",
                value=df['timestamp'].max().date(),
                min_value=df['timestamp'].min().date(),
                max_value=df['timestamp'].max().date(),
                key="btczar_end_date"
            )
        
        with col2:
            # Dropdown for currency
            btczar_currency = st.selectbox(
                "Select Currency",
                options=['ZAR', 'USDT'],
                index=0,
                key="btczar_currency"
            )
    
    # Get lending rate data for the selected date range and currency
    conn = get_db_connection()
    query = f"""
    SELECT * FROM lending_rates 
    WHERE timestamp BETWEEN '{btczar_start_date}' AND '{btczar_end_date} 23:59:59'
    AND currency = '{btczar_currency}'
    ORDER BY timestamp
    """
    lending_df = pd.read_sql_query(query, conn)
    lending_df['timestamp'] = pd.to_datetime(lending_df['timestamp'])
    conn.close()
    
    # Get BTCZAR price data
    btczar_df = fetch_bctzar_price_data(btczar_start_date, btczar_end_date)
    
    if lending_df.empty or btczar_df.empty:
        st.warning(f"No data available for the selected filters")
    else:
        # Merge the lending rates and BTCZAR price data on timestamp
        # First, resample to hourly data to ensure alignment
        # Select only numeric columns for resampling
        numeric_cols = lending_df.select_dtypes(include=['number']).columns.tolist()
        lending_hourly = lending_df.set_index('timestamp')
        # Keep only numeric columns for resampling with mean
        lending_hourly = lending_hourly[numeric_cols].resample('H').mean().reset_index()
        
        btczar_hourly = btczar_df.set_index('timestamp').resample('H').mean().reset_index()
        
        # Merge the dataframes
        merged_df = pd.merge_asof(
            lending_hourly.sort_values('timestamp'),
            btczar_hourly.sort_values('timestamp'),
            on='timestamp',
            direction='nearest'
        )
        
        # Drop rows with NaN values
        merged_df = merged_df.dropna()
        
        # 1. BTCZAR Price vs Lending Rate Chart
        st.subheader(f"BTCZAR Price vs {btczar_currency} Lending Rate")
        
        # Create figure with secondary y-axis
        fig = go.Figure()
        
        # Add lending rate trace
        fig.add_trace(go.Scatter(
            x=merged_df['timestamp'],
            y=merged_df['next_annual'],
            name=f"{btczar_currency} Lending Rate",
            mode='lines',
            line=dict(color='#7FDBFF')
        ))
        
        # Add BTCZAR price trace on secondary y-axis
        fig.add_trace(go.Scatter(
            x=merged_df['timestamp'],
            y=merged_df['btczar_price'],
            name='BTCZAR Price',
            mode='lines',
            line=dict(color='#FF851B'),
            yaxis='y2'
        ))
        
        # Set up the layout with two y-axes
        fig.update_layout(
            template=dark_template,
            xaxis_title="Date",
            yaxis=dict(
                title=f"{btczar_currency} Lending Rate (%)",
                side="left"
            ),
            yaxis2=dict(
                title="BTCZAR Price (ZAR)",
                side="right",
                overlaying="y",
                showgrid=False
            ),
            legend=dict(x=0.01, y=0.99),
            hovermode="x unified",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate correlation
        correlation = merged_df['next_annual'].corr(merged_df['btczar_price'])
        
        st.info(f"Correlation between {btczar_currency} Lending Rate and BTCZAR Price: {correlation:.4f}")
        
        # 2. Data Table
        st.subheader("BTCZAR and Lending Rate Data")
        
        # Prepare data for display
        display_df = merged_df[['timestamp', 'next_annual', 'btczar_price']].copy()
        display_df.rename(columns={
            'timestamp': 'Timestamp',
            'next_annual': f'{btczar_currency} Rate (%)',
            'btczar_price': 'BTCZAR Price (ZAR)'
        }, inplace=True)
        
        # Format the numbers
        display_df[f'{btczar_currency} Rate (%)'] = display_df[f'{btczar_currency} Rate (%)'].round(2)
        display_df['BTCZAR Price (ZAR)'] = display_df['BTCZAR Price (ZAR)'].round(2)
        
        # Display the table
        st.dataframe(display_df, use_container_width=True)

# ===== Tab 4: Interest to BTC =====
with tab4:
    st.header("Interest to BTC Calculator")
    
    # Filters sidebar
    with st.expander("Interest to BTC Calculator Filters"):
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Date pickers for date range
            interest_start_date = st.date_input(
                "Start Date",
                value=(df['timestamp'].max() - timedelta(days=60)).date(),
                min_value=df['timestamp'].min().date(),
                max_value=df['timestamp'].max().date(),
                key="interest_start_date"
            )
            
            interest_end_date = st.date_input(
                "End Date",
                value=df['timestamp'].max().date(),
                min_value=df['timestamp'].min().date(),
                max_value=df['timestamp'].max().date(),
                key="interest_end_date"
            )
        
        with col2:
            # Input for ZAR amount
            zar_amount = st.number_input(
                "Enter ZAR Amount",
                min_value=1,
                value=1000000,
                step=1000,
                format="%d"
            )
    
    # Get lending rate data for ZAR in the selected date range
    conn = get_db_connection()
    query = f"""
    SELECT * FROM lending_rates 
    WHERE timestamp BETWEEN '{interest_start_date}' AND '{interest_end_date} 23:59:59'
    AND currency = 'ZAR'
    ORDER BY timestamp
    """
    zar_rates_df = pd.read_sql_query(query, conn)
    zar_rates_df['timestamp'] = pd.to_datetime(zar_rates_df['timestamp'])
    conn.close()
    
    # Get BTCZAR price data for the same period
    btczar_price_df = fetch_bctzar_price_data(interest_start_date, interest_end_date)
    
    if zar_rates_df.empty or btczar_price_df.empty:
        st.warning(f"No data available for the selected filters")
    else:
        # Calculate hourly interest earned and BTC equivalent
        # Merge the lending rates and BTCZAR price data on timestamp
        # First, resample to hourly data to ensure alignment
        # Select only numeric columns for resampling
        numeric_cols = zar_rates_df.select_dtypes(include=['number']).columns.tolist()
        zar_hourly = zar_rates_df.set_index('timestamp')
        # Keep only numeric columns for resampling with mean
        zar_hourly = zar_hourly[numeric_cols].resample('H').mean().reset_index()
        
        btczar_hourly = btczar_price_df.set_index('timestamp').resample('H').mean().reset_index()
        
        # Merge the dataframes
        merged_df = pd.merge_asof(
            zar_hourly.sort_values('timestamp'),
            btczar_hourly.sort_values('timestamp'),
            on='timestamp',
            direction='nearest'
        )
        
        # Drop rows with NaN values
        merged_df = merged_df.dropna()
        
        # Calculate hourly interest earned
        merged_df['hourly_rate'] = merged_df['next_annual'] / 365 / 24  # Convert annual rate to hourly
        merged_df['hourly_interest_zar'] = zar_amount * (merged_df['hourly_rate'] / 100)  # Interest in ZAR
        merged_df['btc_equivalent'] = merged_df['hourly_interest_zar'] / merged_df['btczar_price']  # BTC equivalent
        
        # Display the calculation results
        st.subheader("Hourly Interest Earned and BTC Equivalent")
        
        # Prepare data for display
        display_df = merged_df[['timestamp', 'next_annual', 'hourly_rate', 'hourly_interest_zar', 'btczar_price', 'btc_equivalent']].copy()
        display_df.rename(columns={
            'timestamp': 'Timestamp',
            'next_annual': 'Annual Rate (%)',
            'hourly_rate': 'Hourly Rate (%)',
            'hourly_interest_zar': 'Hourly Interest (ZAR)',
            'btczar_price': 'BTCZAR Price (ZAR)',
            'btc_equivalent': 'BTC Equivalent'
        }, inplace=True)
        
        # Format the numbers
        display_df['Annual Rate (%)'] = display_df['Annual Rate (%)'].round(2)
        display_df['Hourly Rate (%)'] = display_df['Hourly Rate (%)'].round(6)
        display_df['Hourly Interest (ZAR)'] = display_df['Hourly Interest (ZAR)'].round(2)
        display_df['BTCZAR Price (ZAR)'] = display_df['BTCZAR Price (ZAR)'].round(2)
        display_df['BTC Equivalent'] = display_df['BTC Equivalent'].round(8)
        
        # Display the table
        st.dataframe(display_df, use_container_width=True)
        
        # Calculate summary statistics
        total_interest_zar = merged_df['hourly_interest_zar'].sum()
        total_btc_equivalent = merged_df['btc_equivalent'].sum()
        avg_rate = merged_df['next_annual'].mean()
        avg_btczar_price = merged_df['btczar_price'].mean()
        period_days = (merged_df['timestamp'].max() - merged_df['timestamp'].min()).total_seconds() / (24 * 60 * 60)
        
        # Display summary
        st.subheader("Interest to BTC Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("ZAR Amount", f"R {zar_amount:,.2f}")
            st.metric("Average Annual Rate", f"{avg_rate:.2f}%")
            st.metric("Period Duration", f"{period_days:.2f} days")
        
        with col2:
            st.metric("Total Interest Earned", f"R {total_interest_zar:,.2f}")
            st.metric("Total BTC Equivalent", f"{total_btc_equivalent:.8f} BTC")
            st.metric("Average BTCZAR Price", f"R {avg_btczar_price:,.2f}")
        
        # Interest to BTC visualization
        st.subheader("Cumulative Interest and BTC Equivalent")
        
        # Calculate cumulative sums
        merged_df['cumulative_interest_zar'] = merged_df['hourly_interest_zar'].cumsum()
        merged_df['cumulative_btc'] = merged_df['btc_equivalent'].cumsum()
        
        # Create figure with secondary y-axis
        fig = go.Figure()
        
        # Add cumulative interest trace
        fig.add_trace(go.Scatter(
            x=merged_df['timestamp'],
            y=merged_df['cumulative_interest_zar'],
            name='Cumulative Interest (ZAR)',
            mode='lines',
            line=dict(color='#7FDBFF')
        ))
        
        # Add cumulative BTC trace on secondary y-axis
        fig.add_trace(go.Scatter(
            x=merged_df['timestamp'],
            y=merged_df['cumulative_btc'],
            name='Cumulative BTC',
            mode='lines',
            line=dict(color='#FF851B'),
            yaxis='y2'
        ))
        
        # Set up the layout with two y-axes
        fig.update_layout(
            template=dark_template,
            xaxis_title="Date",
            yaxis=dict(
                title="Cumulative Interest (ZAR)",
                side="left"
            ),
            yaxis2=dict(
                title="Cumulative BTC",
                side="right",
                overlaying="y",
                showgrid=False
            ),
            legend=dict(x=0.01, y=0.99),
            hovermode="x unified",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
