import re
import csv
import os

def extract_java_methods(java_file_path, csv_output_path, include_csv_fields=True):
    method_regex = r'(public|protected|private|static|\s)?\s*((?:[\w\<\>\[\]]+\s*)+)\s+(\w+)\s*\(([^)]*)\)\s*\{?'
    class_regex = r'class\s+\w+'
    return_regex = r'return [^;]+;'
    comment_regex = r'//.*|/\*.*\*/|/\*\*[\s\S]*?\*/'

    methods = []
    class_count = 0
    loc = 0  # Lines of code

    try:
        with open(java_file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
    except Exception as e:
        print(f"Error reading file {java_file_path}: {e}")
        return {}, None

    current_method = None
    brace_count = 0
    method_content = []

    for line in lines:
        loc += 1

        if re.search(class_regex, line):
            class_count += 1

        comment_match = re.search(comment_regex, line)
        comments = comment_match.group() if comment_match else ''

        if current_method is None:
            method_match = re.search(method_regex, line)
            if method_match:
                access_modifier = method_match.group(1).strip() if method_match.group(1) else ''
                method_name = method_match.group(3).strip()
                if not (method_name and access_modifier):
                    continue
                current_method = {
                    'access_modifier': method_match.group(1).strip() if method_match.group(1) else '',
                    'return_type': method_match.group(2).strip(),
                    'method_name': method_match.group(3).strip(),
                    'parameters': method_match.group(4).strip(),
                    'returns': '',
                    'comments': comments,
                    'content': '',
                    'path': java_file_path
                }
                if '{' in line:
                    brace_count = 1
                    method_content = [line.strip()]
                continue

        if current_method is not None:
            method_content.append(line.strip())

            if re.search(return_regex, line):
                current_method['returns'] += line.strip() + ' '

            brace_count += line.count('{') - line.count('}')
            if brace_count == 0:
                current_method['content'] = ' '.join(method_content)
                methods.append(current_method)
                current_method = None
                method_content = []

    if include_csv_fields:
        file_exists = os.path.isfile(csv_output_path)
        with open(csv_output_path, 'a', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['access_modifier', 'return_type', 'method_name', 'parameters', 'returns', 'comments', 'content','path']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()
            for method in methods:
                writer.writerow(method)

    print_info = {
        'Loc': loc,
        'Classes': class_count,
        'Methods': len(methods)
    }

    return print_info, csv_output_path if include_csv_fields else None

# The function calls are commented out to prevent execution in the Python Code Interpreter
# Uncomment and execute with valid Java file paths and CSV file path to test
# java_file_paths = ["path/to/java_file1.java", "path/to/java_file2.java"]
# files_info = []
# for i in java_file_paths:
#     print_info, csv_file_path = extract_java_methods(i, "output_methods.csv", include_csv_fields=True)
#     files_info.append({"file_name":i, "info":print_info})
# print(files_info)
