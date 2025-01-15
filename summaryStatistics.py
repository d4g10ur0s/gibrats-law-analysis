import pandas as pd
import numpy as np
from scipy import stats

def load_and_clean_data(file_path):
    """Load and perform initial data cleaning"""
    df = pd.read_csv(file_path)
    # Remove any rows where all numeric values are null
    df = df.dropna(how='all', subset=['Size', 'Growth_sales', 'Total_Assets', 'Salesn'])
    return df

def create_summary_statistics(df, columns=None, additional_stats=True, percentiles=None):
    """
    Create comprehensive summary statistics for a DataFrame

    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame
    columns : list, optional
        List of column names to analyze. If None, uses all numeric columns
    additional_stats : bool, default True
        Whether to include additional statistics (skewness, kurtosis)
    percentiles : list, optional
        Custom percentiles to include in description (e.g., [0.05, 0.25, 0.5, 0.75, 0.95])

    Returns:
    --------
    pandas.DataFrame
        Summary statistics
    """
    # Automatically detect numeric columns if none specified
    if columns is None:
        columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Ensure all specified columns exist and are numeric
    valid_columns = [col for col in columns if col in df.columns
                    and df[col].dtype in ['int64', 'float64']]

    if not valid_columns:
        raise ValueError("No valid numeric columns found")

    # Create basic statistics with optional custom percentiles
    if percentiles:
        summary = df[valid_columns].describe(percentiles=percentiles)
    else:
        summary = df[valid_columns].describe()

    # Add additional statistics if requested
    if additional_stats:
        for col in valid_columns:
            clean_data = df[col].dropna()
            try:
                summary.loc['skew', col] = stats.skew(clean_data)
                summary.loc['kurtosis', col] = stats.kurtosis(clean_data)
                summary.loc['missing_count', col] = df[col].isna().sum()
                summary.loc['missing_pct', col] = (df[col].isna().sum() / len(df)) * 100
            except Exception as e:
                print(f"Warning: Could not calculate some statistics for {col}: {str(e)}")

    return summary


def analyze_categorical_variables(df, columns=None, metrics=None, top_n=None, sort_by='count',
                               include_pct=True, min_freq=None):
    """
    Analyze categorical variables with customizable metrics and filtering

    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame
    columns : list or None
        List of columns to analyze. If None, automatically detects categorical columns
    metrics : list or None
        List of metrics to calculate. Options: ['count', 'percentage', 'cumulative']
    top_n : int or None
        Limit results to top N categories
    sort_by : str
        How to sort results ('count', 'alphabetical', 'index')
    include_pct : bool
        Whether to include percentages in the output
    min_freq : int or float
        Minimum frequency/percentage threshold for including categories
        If float < 1, treated as percentage threshold
        If int >= 1, treated as count threshold

    Returns:
    --------
    dict : Dictionary of DataFrames with categorical analysis
    """
    # Automatically detect categorical columns if none specified
    if columns is None:
        columns = df.select_dtypes(include=['object', 'category', 'string' ,'int64']).columns.tolist()

    # Validate columns exist in DataFrame
    valid_columns = [col for col in columns if col in df.columns]
    if not valid_columns:
        raise ValueError("No valid categorical columns found")

    # Default metrics if none specified
    if metrics is None:
        metrics = ['count', 'percentage']

    cat_summary = {}

    for col in valid_columns:
        # Calculate value counts
        counts = df[col].value_counts()
        total = len(df)

        # Create summary DataFrame
        summary = pd.DataFrame()

        if 'count' in metrics:
            summary['Count'] = counts

        if 'percentage' in metrics:
            summary['Percentage'] = counts / total * 100

        if 'cumulative' in metrics:
            if 'percentage' in metrics:
                summary['Cumulative %'] = summary['Percentage'].cumsum()
            else:
                summary['Cumulative Count'] = counts.cumsum()

        # Apply minimum frequency filter
        if min_freq is not None:
            if min_freq < 1:  # Percentage threshold
                summary = summary[summary['Percentage'] >= min_freq * 100]
            else:  # Count threshold
                summary = summary[summary['Count'] >= min_freq]

        # Sort results
        if sort_by == 'count':
            summary = summary.sort_values('Count', ascending=False)
        elif sort_by == 'alphabetical':
            summary = summary.sort_index()
        # 'index' sorting is default pandas behavior

        # Limit to top N categories
        if top_n is not None:
            summary = summary.head(top_n)

        # Add "Other" category if needed
        if top_n is not None or min_freq is not None:
            remaining_count = total - summary['Count'].sum()
            if remaining_count > 0:
                other_row = pd.DataFrame({
                    'Count': [remaining_count],
                    'Percentage': [remaining_count / total * 100]
                }, index=['Other'])
                summary = pd.concat([summary, other_row])

        cat_summary[col] = summary

    return cat_summary

def analyze_size_distributions(df):
    """Analyze size distributions and create size classes"""
    # Create size quintiles
    df['size_quintile'] = pd.qcut(df['Size'], q=5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])

    # Calculate mean growth rates by size quintile
    size_growth = df.groupby('size_quintile')['Growth_sales'].agg(['mean', 'std', 'count'])

    return size_growth

def create_correlation_matrix(df):
    """Create correlation matrix for key variables"""
    corr_vars = ['Size', 'Growth_sales', 'Tobins_Q_', 'ROA_Net_income',
                 'Leverage', 'Salesn']
    correlation_matrix = df[corr_vars].corr()

    return correlation_matrix

def analyze_by_country(df):
    """Create summary statistics by country"""
    key_metrics = ['Size', 'Growth_sales', 'Tobins_Q_', 'ROA_Net_income']
    country_stats = df.groupby('CountryISOcode')[key_metrics].agg(['mean', 'median', 'std'])

    return country_stats

def main():
    # Load data
    df = load_and_clean_data('firms.csv')

    print("1. Basic Summary Statistics")
    print("-" * 50)
    print(create_summary_statistics(df))
    print("\n")

    print("2. Categorical Variables Analysis")
    print("-" * 50)
    cat_results = analyze_categorical_variables(df)
    for key, value in cat_results.items():
        print(f"\n{key}:")
        print(value)
    print("\n")

    print("3. Size-Growth Analysis")
    print("-" * 50)
    print(analyze_size_distributions(df))
    print("\n")

    print("4. Correlation Matrix")
    print("-" * 50)
    print(create_correlation_matrix(df))
    print("\n")

    print("5. Country Analysis")
    print("-" * 50)
    print(analyze_by_country(df))
    print(list(df.keys()))

if __name__ == "__main__":
    main()
