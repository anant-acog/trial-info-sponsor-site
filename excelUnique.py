import pandas as pd

def get_unique_column_c(file_path):
    """
    Reads column C from the 'Pipeline_Data' sheet of the specified Excel file
    and returns a list of unique values.
    
    Args:
        file_path (str): Path to the Excel file
    
    Returns:
        list: List of unique values from column C
    """
    # Read the specific sheet and column C only for efficiency
    df = pd.read_excel(file_path, sheet_name='Pipeline_Data', usecols='C')
    
    # Get unique values from column C and convert to list
    unique_values = df.iloc[:, 0].dropna().unique().tolist()
    
    return unique_values

# Example usage:
file_path = "GIPR-Target-Pipeline-Updated.xlsx"
unique_list = get_unique_column_c(file_path)
print(unique_list)
print(len(unique_list))
