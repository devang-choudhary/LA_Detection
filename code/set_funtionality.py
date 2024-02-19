import csv
def process_csv_new(input_file, output_file):
    with open(input_file, 'r', newline='') as infile, open(output_file, 'w', newline='') as outfile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames + ['label']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)

        writer.writeheader()

        for row in reader:
            # Ignore rows where content is ''
            if row['content'] == '':
                continue
            # Check if the return type is null and there is no return statement
            if row['return_type'].lower() == 'void' and 'return' not in row['returns']:
                label = 'negative'
            else:
                label = 'positive'

            # Write the row with the new label
            row['label'] = label
            writer.writerow(row)

# The function calls are commented out to prevent execution
# Uncomment these lines to execute the function with the appropriate file paths
def set_labeled(file):
    process_csv_new(f'../{file}_set.csv', f'../{file}_set_labeled.csv')

if __name__ == "__main__":
    file = input("Enter the file name: ")
    set_labeled(file)
