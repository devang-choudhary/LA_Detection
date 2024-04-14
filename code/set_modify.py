import pandas as pd
def modify_csv_v3(file_path, output_path):
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Iterate over the rows
    for index, row in df.iterrows():
        if row['label'] == 'negative':
            # Change return_type to int and add 'return 0;' to returns
            df.at[index, 'return_type'] = 'int'
            df.at[index, 'returns'] = 'return 0;'

            # Check if content is not NaN and is a string
            if pd.notna(row['content']) and isinstance(row['content'], str):
                # Modify content field
                content = row['content']
                first_brace = content.find('{')
                last_brace = content.rfind('}')
                if first_brace != -1 and last_brace != -1:
                    # Extract content between the first '{' and the last '}'
                    modified_content = content[first_brace:last_brace]
                    # Add 'return 0;' before the last '}'
                    modified_content += ' return 0;' + content[last_brace:]
                    df.at[index, 'content'] = modified_content
                df['label'] = 'positive'    
    # Save the modified DataFrame to a new CSV file
    df.to_csv(output_path, index=False)

# Example usage (commented out)

def set_modify(file):
    modify_csv_v3(f'../{file}_set_labeled.csv', f'../{file}_set_labeled_output.csv')


if __name__ == "__main__":
    file = input("Enter the file name: ")
    set_modify(file)
