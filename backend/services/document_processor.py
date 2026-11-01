
import pymupdf


def extract_text_from_pdf(file_path: str):

    """
    Extract text from every page of a PDF.
    """

    document = pymupdf.open(file_path)

    extracted_pages = []

    for page_number, page in enumerate(document):

        text = page.get_text("text")

        if text.strip():

            extracted_pages.append({
                "page_number": page_number + 1,
                "text": text
            })

    document.close()

    return extracted_pages


def chunk_text(text: str, chunk_size=500, overlap=100):

    """
    Divide text into smaller overlapping chunks.
    """

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk.strip())

        start += chunk_size - overlap

    return chunks


def process_pdf(file_path: str):

    """
    Extract PDF text and generate chunks
    while preserving page information.
    """

    pages = extract_text_from_pdf(file_path)

    processed_chunks = []

    for page in pages:

        chunks = chunk_text(page["text"])

        for chunk in chunks:

            processed_chunks.append({
                "text": chunk,
                "page_number": page["page_number"]
            })

    return processed_chunks
