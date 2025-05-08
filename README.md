# Crypto Lending Rates Dashboard

A dark-themed Dash Plotly dashboard for analyzing cryptocurrency lending rates from the depos.db database.

## Features

- Interactive time series visualization of lending rates
- Rate comparison across different currencies
- Rate distribution analysis
- Recent rate changes table with color-coded indicators
- Filtering by date range, currencies, and rate type (daily/annual)

## Installation

1. Install the required dependencies:

```
pip install -r requirements.txt
```

2. Make sure the `depos.db` file is in the same directory as the dashboard application.

## Usage

1. Run the dashboard:

```
python depo_dashboard.py
```

2. Open your web browser and navigate to http://127.0.0.1:8050/

3. Use the filters on the left side to:
   - Select a date range
   - Choose currencies to display
   - Toggle between daily and annual rates
   - Click "Update Dashboard" to refresh the visualizations

## Dashboard Components

- **Rate Trends Over Time**: Line chart showing how rates change over the selected time period
- **Rate Comparison**: Bar chart comparing the latest rates for selected currencies
- **Rate Distribution**: Box plot showing the statistical distribution of rates
- **Recent Rate Changes**: Table displaying the most recent rate changes with color-coded indicators
