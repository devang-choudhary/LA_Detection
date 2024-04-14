import csv
import re
import sys

def process_csv(input_file, output_file):
    # Increase the maximum field size limit
    max_field_size = sys.maxsize
    while True:
        try:
            csv.field_size_limit(max_field_size)
            break
        except OverflowError:
            max_field_size = int(max_field_size/10)

    # Define the pattern to match specific return statements
    pattern = re.compile(r'return\s+(true;|false;)\s*')

    with open(input_file, 'r', newline='') as infile, open(output_file, 'w', newline='') as outfile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames + ['label']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)

        writer.writeheader()

        for row in reader:
            # Ignore rows where content is ''
            if row['content'] == '':
                continue

            # Default label is positive
            label = 'positive'

            # Check if the return type is Boolean
            if row['return_type'].lower() == 'boolean':
                # Check if the returns field matches the pattern
                if pattern.fullmatch(row['returns'].strip()):
                    label = 'negative'

            # Write the row with the new label
            row['label'] = label
            writer.writerow(row)

# The function calls are commented out to prevent execution
# Uncomment these lines to execute the function with the appropriate file paths
# process_csv('../pytorch-test/is.csv', '../pytorch-test/is_labeled_data.csv')

def is_labeled_data(file):
    process_csv(f'../{file}_is.csv', f'../{file}_is_labeled.csv')


if __name__ == "__main__":  
    file = input("Enter the file name: ")
    is_labeled_data(file)  
