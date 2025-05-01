import pdfplumber
import os


def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF datasheet."""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = "".join(page.extract_text() or "" for page in pdf.pages)
        return text
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        return ""


def process_directory(directory):
    """Process all PDFs in a directory and return a list of (filename, text) tuples."""
    results = []
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            filepath = os.path.join(directory, filename)
            text = extract_text_from_pdf(filepath)
            results.append((filename, text))
    return results


if __name__ == "__main__":
    # Example usage
    datasheets = process_directory("data/datasheets")
    for filename, text in datasheets:
        print(f"Processed {filename}: {text[:100]}...")
