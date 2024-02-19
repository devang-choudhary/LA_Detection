import pandas as pd

def modify_csv(file_path, output_path):
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Iterate over the rows
    for index, row in df.iterrows():
        if row['label'] == 'negative':
            # Change return_type to int
            df.at[index, 'return_type'] = 'int'

            # Replace 'true' with 1 and 'false' with 0 in returns column
            df.at[index, 'returns'] = str(row['returns']).replace('true', '1').replace('false', '0')

            # Modify content field
            content = row['content']
            first_brace = content.find('{')
            last_brace = content.rfind('}')
            if first_brace != -1 and last_brace != -1:
                modified_content = content[first_brace:last_brace+1]
                modified_content = modified_content.replace('return true;', 'return 1;').replace('return false;', 'return 0;')
                df.at[index, 'content'] = modified_content
            row['label'] = 'positive'
    # Save the modified DataFrame to a new CSV file
    df.to_csv(output_path, index=False)


def is_modified(file):
    modify_csv(f'../{file}_is_labeled.csv', f'../{file}_is_labeled_output.csv')


if __name__ == "__main__":
    file = input("Enter the file name: ")
    is_modified(file)