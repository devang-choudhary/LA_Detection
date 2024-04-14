import csv
import sys

def process_csv_updated(input_file, output_file):
    # Increase the maximum size of fields
    max_int = sys.maxsize
    while True:
        # Decrease the max_int value by half each time to prevent OverflowError
        try:
            csv.field_size_limit(max_int)
            break
        except OverflowError:
            max_int = int(max_int/2)

    with open(input_file, 'r', newline='') as infile, open(output_file, 'w', newline='') as outfile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames + ['label']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)

        writer.writeheader()
        
        for row in reader:
            # Ignore rows where content is ''
            if row['content'] == '':
                continue
            # Check if the return type is not void and there are return statements
            if row['return_type'].lower() != 'void' and 'return' in row['returns']:
                label = 'negative'
            else:
                label = 'positive'

            # Write the row with the new label
            row['label'] = label
            writer.writerow(row)

# Uncomment these lines to execute the function with the appropriate file paths
def get_labeled(file):
    process_csv_updated(f'../{file}_get.csv', f'../{file}_get_labeled.csv')
    
if __name__ == "__main__":
    file = input("Enter the file name: ")
    get_labeled(file)
