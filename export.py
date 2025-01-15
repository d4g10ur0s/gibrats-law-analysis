import pandas as pd

def format_coefficient_with_stars(coef, p_value):
    """Format coefficient with STATA-style significance stars"""
    coef_str = f"{coef:.3f}"
    if p_value < 0.01:
        coef_str += "$^{***}$"
    elif p_value < 0.05:
        coef_str += "$^{**}$"
    elif p_value < 0.1:
        coef_str += "$^{*}$"
    return coef_str

def create_latex_firm_table(size_results, sales_results):
    """Create LaTeX table for firm-level analysis"""
    latex = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Gibrat's Law Test Results - Firm Level}",
        "\\label{tab:gibrat_firm}",
        "\\begin{tabular}{lcc}",
        "\\hline\\hline",
        "& Size & Sales \\\\",
        "\\hline"
    ]

    # Add coefficient row with stars
    size_coef = format_coefficient_with_stars(size_results['coefficient'], size_results['p_value'])
    sales_coef = format_coefficient_with_stars(sales_results['coefficient'], sales_results['p_value'])
    latex.append(f"Coefficient ($\\beta$) & {size_coef} & {sales_coef} \\\\")

    # Add standard errors in parentheses
    latex.append(f"& ({size_results['std_error']:.3f}) & ({sales_results['std_error']:.3f}) \\\\")

    # Add other statistics
    latex.extend([
        f"R-squared & {size_results['r_squared']:.3f} & {sales_results['r_squared']:.3f} \\\\",
        f"Observations & {size_results['n_obs']} & {sales_results['n_obs']} \\\\",
        f"Number of Firms & {size_results['n_firms']} & {sales_results['n_firms']} \\\\",
        "\\hline\\hline",
        "\\end{tabular}",
        "\\begin{tablenotes}",
        "\\small",
        "\\item Notes: Standard errors in parentheses. $^{***}$ p<0.01, $^{**}$ p<0.05, $^{*}$ p<0.1",
        "\\item Gibrat's Law holds if coefficient equals 1.",
        "\\end{tablenotes}",
        "\\end{table}"
    ])

    return '\n'.join(latex)

def create_latex_sector_table(sector_results, measure_type):
    """Create LaTeX table for sector-level analysis"""
    latex = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{Gibrat's Law Test Results by Sector - {measure_type}}}",
        f"\\label{{tab:gibrat_sector_{measure_type.lower()}}}",
        "\\begin{tabular}{lccccc}",
        "\\hline\\hline",
        "Sector & Coefficient & R-squared & Obs & Firms & Growth Mean \\\\",
        "\\hline"
    ]

    # Add rows for each sector
    for sector, result in sector_results.items():
        if 'error' in result:
            continue

        coef = format_coefficient_with_stars(result['coefficient'], result['p_value'])
        latex.append(
            f"{sector} & {coef} & {result['r_squared']:.3f} & "
            f"{result['n_obs']} & {result['n_firms']} & {result['growth_rate_mean']:.3f} \\\\"
        )

    latex.extend([
        "\\hline\\hline",
        "\\end{tabular}",
        "\\begin{tablenotes}",
        "\\small",
        "\\item Notes: $^{***}$ p<0.01, $^{**}$ p<0.05, $^{*}$ p<0.1",
        "\\item Gibrat's Law holds if coefficient equals 1.",
        "\\end{tablenotes}",
        "\\end{table}"
    ])

    return '\n'.join(latex)

def summary_to_latex(summary_df, caption=None, label=None, float_format=lambda x: '{:,.3f}'.format(x)):
    """
    Convert summary statistics DataFrame to a LaTeX table

    Parameters:
    -----------
    summary_df : pandas.DataFrame
        Summary statistics DataFrame
    caption : str, optional
        Table caption
    label : str, optional
        Table label for referencing
    float_format : callable, optional
        Function to format float numbers

    Returns:
    --------
    str : LaTeX table code
    """
    # Create a copy to avoid modifying the original
    df_latex = summary_df.copy()

    # Rename the index for better readability
    index_mapping = {
        'count': 'N',
        'mean': 'Mean',
        'std': 'Std Dev',
        'min': 'Min',
        'max': 'Max',
        'skew': 'Skewness',
        'kurtosis': 'Kurtosis',
        'missing_count': 'Missing (N)',
        'missing_pct': 'Missing (\%)',
        '5%': 'P5',
        '25%': 'P25',
        '50%': 'P50',
        '75%': 'P75',
        '95%': 'P95'
    }
    df_latex.index = df_latex.index.map(lambda x: index_mapping.get(x, x))

    # Convert to LaTeX
    latex_table = df_latex.to_latex(
        float_format=float_format,
        escape=False,
        caption=caption,
        label=label,
        column_format='l' + 'r' * len(df_latex.columns)  # left align first column, right align others
    )

    # Add some common LaTeX table improvements
    latex_table = latex_table.replace('table', 'table*')  # Make table span multiple columns
    latex_table = latex_table.replace('\\begin{table*}',
                                    '\\begin{table*}[htbp]\n\\centering\n\\small')

    return latex_table

def save_latex_table(latex_table, output_path, standalone=False):
    """
    Save LaTeX table to a file

    Parameters:
    -----------
    latex_table : str
        LaTeX table code
    output_path : str
        Path where to save the file (e.g., 'tables/summary_stats.tex')
    standalone : bool, default False
        If True, adds LaTeX document wrapper for standalone compilation
    """
    # Create directories if they don't exist
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if standalone:
        # Add document wrapper for standalone compilation
        content = (
            "\\documentclass{article}\n"
            "\\usepackage{booktabs}\n"
            "\\usepackage{siunitx}\n"
            "\\usepackage[margin=1in]{geometry}\n"
            "\\begin{document}\n\n"
            f"{latex_table}\n\n"
            "\\end{document}"
        )
    else:
        content = latex_table

    # Save to file
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"LaTeX table saved successfully to {output_path}")
    except Exception as e:
        print(f"Error saving LaTeX table: {str(e)}")

# Example usage combining both functions:
def create_and_save_summary_latex(df, output_path, caption=None, label=None, standalone=False):
    """
    Create summary statistics and save as LaTeX table

    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame
    output_path : str
        Path where to save the LaTeX file
    caption : str, optional
        Table caption
    label : str, optional
        Table label for referencing
    standalone : bool, default False
        If True, creates a standalone LaTeX document
    """
    # Create summary statistics
    summary_stats = create_summary_statistics(df, percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])

    # Convert to LaTeX
    latex_table = summary_to_latex(
        summary_stats,
        caption=caption,
        label=label
    )

    # Save to file
    save_latex_table(latex_table, output_path, standalone=standalone)

def categorical_to_latex(cat_summary, caption_prefix="Distribution of", float_format=lambda x: '{:,.2f}'.format(x)):
    """
    Convert categorical analysis results to LaTeX tables

    Parameters:
    -----------
    cat_summary : dict
        Dictionary output from analyze_categorical_variables
    caption_prefix : str
        Prefix for table captions
    float_format : callable
        Function to format float numbers

    Returns:
    --------
    dict : Dictionary of LaTeX tables
    """
    latex_tables = {}

    for var_name, df in cat_summary.items():
        # Create a copy to avoid modifying the original
        df_latex = df.copy()

        # Format column names
        column_mapping = {
            'Count': 'N',
            'Percentage': '\% of Total',
            'Cumulative %': 'Cumulative \%'
        }
        df_latex.columns = [column_mapping.get(col, col) for col in df_latex.columns]

        # Create caption and label
        clean_name = var_name.replace('_', ' ').title()
        caption = f"{caption_prefix} {clean_name}"
        label = f"tab:dist_{var_name.lower().replace(' ', '_')}"

        # Convert to LaTeX
        latex_table = df_latex.to_latex(
            float_format=float_format,
            escape=False,
            caption=caption,
            label=label,
            column_format='l' + 'r' * len(df_latex.columns)
        )

        # Add LaTeX improvements
        latex_table = latex_table.replace('table', 'table*')
        latex_table = latex_table.replace('\\begin{table*}',
                                        '\\begin{table*}[htbp]\n\\centering\n\\small')

        latex_tables[var_name] = latex_table

    return latex_tables

def save_categorical_latex(cat_latex_tables, output_dir, standalone=False):
    """
    Save categorical LaTeX tables to files

    Parameters:
    -----------
    cat_latex_tables : dict
        Dictionary of LaTeX tables
    output_dir : str
        Directory to save the files
    standalone : bool
        Whether to create standalone LaTeX documents
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    for var_name, latex_table in cat_latex_tables.items():
        # Create filename
        filename = f"distribution_{var_name.lower().replace(' ', '_')}.tex"
        output_path = os.path.join(output_dir, filename)

        # Save table
        save_latex_table(latex_table, output_path, standalone)

# Combined function for convenience
def create_and_save_categorical_latex(df, output_dir, standalone=False, **cat_analysis_kwargs):
    """
    Create and save categorical analysis LaTeX tables

    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame
    output_dir : str
        Directory to save the files
    standalone : bool
        Whether to create standalone LaTeX documents
    **cat_analysis_kwargs :
        Additional arguments for analyze_categorical_variables
    """
    # Create categorical analysis
    cat_summary = analyze_categorical_variables(df, **cat_analysis_kwargs)

    # Convert to LaTeX
    latex_tables = categorical_to_latex(cat_summary)

    # Save tables
    save_categorical_latex(latex_tables, output_dir, standalone)
