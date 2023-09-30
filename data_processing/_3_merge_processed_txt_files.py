import os

def merge_txt_files(directory_path, output_filename):
    """
    Traverse all files ending with '_processed.txt' in the specified directory,
    and merge their content into a new txt file.
    
    Parameters:
    - directory_path (str): The path to the directory to be traversed.
    - output_filename (str): The name of the output file.
    """
    
    # Get all files ending with '_processed.txt' in the directory
    processed_files = [f for f in os.listdir(directory_path) if f.endswith('_processed.txt')]
    
    # Initialize an empty string list to hold the content of the files
    texts = []
    
    # Traverse the files, read and store their content
    for txt_file in processed_files:
        file_path = os.path.join(directory_path, txt_file)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            texts.append(content)
    
    # Merge the content of all files and save it to a new txt file
    merged_content = "\n".join(texts)
    output_path = os.path.join(directory_path, output_filename)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(merged_content)

# This program is designed to merge the content of all text files ending with '_processed.txt'
# located in a specified directory and save the merged content to a new text file within the same directory.

# Call the function
directory_path = '<Your Data Directory>' # replace with the actual directory path
output_filename = 'res.txt'
merge_txt_files(directory_path, output_filename)

print(f"All content from files ending with '_processed.txt' has been merged into {output_filename}.")
