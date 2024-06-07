import os
import asyncio
from dotenv import load_dotenv
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
from textwrap import dedent
from PyPDF2 import PdfReader

# Load environment variables
load_dotenv()

# Directory where the PDFs are stored
pdf_dir = './PDF'

# Directory where the MD files will be stored
md_dir = './MD'

# Create the MD directory if it doesn't exist
if not os.path.exists(md_dir):
    os.makedirs(md_dir)

parsing_instruction = dedent("""
    You are an advanced language model tasked with converting academic PDF books and research papers into markdown format. Ensure all text, figures, tables, equations, references, citations, and footnotes are accurately converted, preserving the original content, order, and formatting. Your goal is to produce markdown files that faithfully reflect the structure and data of the original PDFs, making them easy to read and reference.
""").strip()

# Set up parser
parser = LlamaParse(
    parsing_instruction=parsing_instruction,
    result_type="markdown"  # "markdown" and "text" are available
)

def validate_page_count(directory, max_pages=750):
    """
    This function validates the total number of pages in all PDF files in a given directory.

    Parameters:
    directory (str): The path to the directory containing the PDF files.
    max_pages (int, optional): The maximum allowed total number of pages in all PDF files. Default is 750.

    Returns:
    bool: True if the total number of pages in all PDF files is less than or equal to max_pages. False otherwise.

    The function works by looping through all files in the specified directory. If a file ends with '.pdf',
    it opens the file using the PdfReader class from the PyPDF2 library and counts the number of pages in the PDF.
    It then adds this number to a running total. After checking all PDF files, the function prints the total number
    of pages and returns a boolean value indicating whether this total is less than or equal to max_pages.
    """
    total_pages = 0

    # Loop through all files in the PDF directory
    for filename in os.listdir(directory):
        if filename.endswith('.pdf'):
            input_filepath = os.path.join(directory, filename)

            # Count the number of pages in the PDF
            reader = PdfReader(input_filepath)
            num_pages = len(reader.pages)
            total_pages += num_pages

    return total_pages <= max_pages


async def process_pdfs():
    """
    This function is used to process PDF files in a directory. It converts each PDF file into a markdown file.
    If a markdown file for a PDF already exists, the function skips the conversion for that PDF.

    The function follows these steps:
    1. Loops through all files in the PDF directory.
    2. Checks if the file is a PDF.
    3. Constructs the output filepath for the markdown file.
    4. Checks if the markdown file already exists. If it does, the function skips the conversion for that PDF.
    5. Loads the PDF document asynchronously.
    6. Extracts the markdown content from the document.
    7. Writes the markdown content to a file.
    """
    # Loop through all files in the PDF directory
    for index, filename in enumerate(os.listdir(pdf_dir)):
        if filename.endswith('.pdf'):
            print(f"{index} : {filename}")
            input_filepath = os.path.join(pdf_dir, filename)

            # Construct the output filepath
            output_filename = f"{os.path.splitext(filename)[0]}.md"
            output_filepath = os.path.join(md_dir, output_filename)

            # Check if the markdown file already exists
            if os.path.exists(output_filepath):
                print(f"Markdown file for {filename} already exists. Skipping conversion.")
                continue

            # Load documents asynchronously
            documents = await parser.aload_data(input_filepath)

            # Extract the markdown content
            markdown_content = documents[0].text  # Assuming documents is a list and the text property holds the markdown content

            # Write the markdown content to a file
            with open(output_filepath, 'w', encoding='utf-8') as md_file:
                md_file.write(markdown_content)

            print(f"Markdown content saved to {output_filepath}")


# Validate the page count and run the asynchronous function if valid
if validate_page_count(pdf_dir):
    # Run the asynchronous function
    asyncio.run(process_pdfs())
else:
    print("The total number of pages in all PDF files is greater than the max_pages")
