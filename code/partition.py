import pandas as pd


# Define functions to classify method names
def classify_method_name(method_name):
    method_name_lower = method_name.lower()
    if method_name_lower.startswith(('is','are','do','does','check', 'verify', 'validate', 'ensure', 'confirm', 'assure', 'corroborate', 'ratify', 'authenticate', 'certify', 'test')):
        return 'is'
    elif method_name_lower.startswith(('get', 'extract', 'retrieve', 'acquire', 'obtain', 'receive', 'collect', 'pick', 'capture', 'trace', 'fetch')):
        return 'get'
    elif method_name_lower.startswith(('set', 'assign', 'allcate', 'allot', 'give')):
        return 'set'
    else:
        return 'other'


def partition_methods(file):
    # Create a dictionary of DataFrames for each method type
    df = pd.read_csv(f"../{file}_data.csv")
    method_data = {}
    for method_type in ['is', 'get', 'set',]:
        method_data[method_type] = df[df['method_name'].apply(classify_method_name) == method_type]

    # Write data to respective CSV files
    for method_type, data in method_data.items():
        filename = f'{file}_{method_type}.csv'
        data.to_csv(filename, index=False)

    return method_data



if __name__ == "__main__":
    file = input("Enter the file name: ")
    partition_methods(file)
