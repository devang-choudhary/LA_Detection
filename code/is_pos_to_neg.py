import pandas as pd

def modify_csv(file_path, output_path):
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Iterate over the rows
    for index, row in df.iterrows():
        if not row['content'].strip():
            continue
        
        if row['label'] == 'positive' and row['return_type']=='boolean' and 'return ' in row['returns']:
            # Replace 'return something;' with 'return something == 1;'
            returns = row['returns']
            returns = returns.strip()
            if returns.endswith(';'):
                returns.strip()
                returns = returns[:-1]  # Remove the trailing semicolon
                df.at[index, 'returns'] = returns.strip() + ' == 1;'
                df.at[index, 'label'] = 'negative'
        
        elif row['label'] == 'positive' and row['return_type']=='void':
                print(row['return_type'])
                # print(row)
                df.at[index, 'return_type'] = 'int'
                df.at[index, 'returns'] = 'return 0;'
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
                df.at[index, 'label'] = 'negative'
                print(df.at[index, 'return_type'])
    
    print(df['label'].value_counts())
    df=df[df['label'] == 'negative']
    print(df['label'].value_counts())
    # Save the modified DataFrame to a new CSV file
    df.to_csv(output_path, index=False)


def is_modified(file):
    modify_csv(f'../{file}_is_labeled.csv', f'../data/{file}_is_labeled_temp.csv')


if __name__ == "__main__":
    file = input("Enter the file name: ")
    is_modified(file)
