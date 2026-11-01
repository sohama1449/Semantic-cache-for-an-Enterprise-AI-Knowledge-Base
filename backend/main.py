
from fastapi import FastAPI, UploadFile, File, HTTPException
from pathlib import Path
import shutil

from backend.services.document_processor import process_pdf


app = FastAPI(title="KnowCache API")

UPLOAD_DIR = Path("sample_documents")

UPLOAD_DIR.mkdir(exist_ok=True)


@app.get("/")
def home():

    return {
        "message": "Welcome to KnowCache API",
        "status": "Running"
    }


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )

    file_path = UPLOAD_DIR / file.filename

    try:

        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        chunks = process_pdf(str(file_path))

        return {
            "filename": file.filename,
            "total_chunks": len(chunks),
            "chunks": chunks
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )

    finally:
        await file.close()
