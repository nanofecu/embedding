import pdfplumber
import os
from tqdm import tqdm

def pdf_to_text_with_pdfplumber(pdf_file_path, txt_file_path):
    """
    Extracts text from a PDF file and saves it to a txt file using pdfplumber.
    
    :param pdf_file_path: The file path to the source PDF file. 
    :type pdf_file_path: str
    :param txt_file_path: The file path where the extracted text will be saved.
    :type txt_file_path: str
    """
    with pdfplumber.open(pdf_file_path) as pdf:
        # Concatenating text from all pages
        text = '\n'.join(page.extract_text() for page in pdf.pages if page.extract_text())
        
    with open(txt_file_path, 'w', encoding='utf-8') as txt_file:
        txt_file.write(text)

# Getting all pdf files in the "data" directory
pdf_files = [f for f in os.listdir('<Your Data Directory>') if f.endswith('.pdf')]

# This program is designed to extract text from all PDF files located 
# in a specified directory and save the extracted text to corresponding 
# txt files within the same directory. The overall progress of processing 
# each file is displayed through a progress bar.

for pdf_file in tqdm(pdf_files, desc="Processing", ncols=100, unit="file"):
    # Printing the name of the file currently being processed
    print(f"\nProcessing: {pdf_file}")
    
    # Defining the full path of each pdf file
    pdf_file_path = os.path.join('<Your Data Directory>', pdf_file)
    
    # Defining the name of the corresponding txt file using the name of the pdf file (without the extension)
    txt_file_name = os.path.splitext(pdf_file)[0] + '.txt'
    txt_file_path = os.path.join('<Your Data Directory>', txt_file_name)

    try:
        pdf_to_text_with_pdfplumber(pdf_file_path, txt_file_path)
    except Exception as e:
        # Printing the error message and informing that this file has been skipped
        print(f"Error processing {pdf_file}: {e}. This file has been skipped.")

print("\nText from all PDF files has been successfully extracted and saved to corresponding TXT files!")