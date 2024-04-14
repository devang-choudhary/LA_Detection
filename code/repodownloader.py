import os
import git
from method_extractor import extract_java_methods
def download_github_project_and_extract_java_files(repo_url, output_file):
    if os.path.exists(output_file):
        print("Project already exists")
    else:
        git.Repo.clone_from(repo_url, output_file)
    java_files = []
    for root, dirs, files in os.walk(output_file):
        for file in files:
            if file.endswith(".java"):
                java_files.append(os.path.join(root, file))
                
    print(f"Total Java files found: {len(java_files)}")
    return java_files

def extract_java_methods_from_project(repo_url, output_file):
    # print(java_file_paths)
    java_file_paths = download_github_project_and_extract_java_files(repo_url, output_file)
    files_output = open(output_file+"_info.txt","w")
    count = 0
    print(f"iterating over the files: {len(java_file_paths)}")
    for i in java_file_paths:
        count += 1
        if count % 100 == 0:
            print(f"Processed {count} files")
        print_info, csv_file_path = extract_java_methods(i, output_file+"_data.csv", include_csv_fields=True)
        files_output.write(f"File Name: {i}    Info:{print_info}\n")
# Example usage

if __name__ == "__main__":
    repo_url = input("Enter the repo url: ")
    output_file = input("Enter the output file name: ")
    extract_java_methods_from_project(repo_url, output_file)




# files_output = open(output_file+"_info.txt","w")
# for i in files_info:

