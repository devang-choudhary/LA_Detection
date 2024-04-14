import csv
import re
import sys

def process_csv(input_file, output_file):
    # Increase the maximum size of fields
    csv.field_size_limit(sys.maxsize)

    with open(input_file, 'r', newline='') as infile, open(output_file, 'w', newline='') as outfile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)

        writer.writeheader()
        
        for row in reader:
            if row['label'] == 'negative':
                # Change return_type to int
                row['return_type'] = 'void'

                # Set returns to empty string
                row['returns'] = ''

                # Remove 'return <expression>;' from content
                # row['content'] = re.sub(r'return [^;]*;', '', row['content'])
            row["label"] = "positive" 
            # Write the modified row
            writer.writerow(row)

# Uncomment these lines to execute the function with the appropriate file paths


def get_modify(file):
    process_csv(f'../{file}_get_labeled.csv', f'../{file}_get_labeled_output.csv')

if __name__ == "__main__":
    file = input("Enter the file name: ")
    get_modify(file)
