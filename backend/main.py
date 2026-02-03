"""
FastAPI Backend for Worksheet Splitter
YOLOv26 Custom Model for Question Detection
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import tempfile
import shutil
import zipfile
import os
from pathlib import Path
import io
import traceback
import fitz

from pocketbase import PocketBase
from datetime import datetime
from contextlib import asynccontextmanager

from split_pdf import YOLOQuestionSplitter
from dotenv import load_dotenv
load_dotenv()

# 1. Detect environment
IS_RAILWAY = os.environ.get("RAILWAY_ENVIRONMENT_NAME") is not None

# 2. Set URL based on environment
if IS_RAILWAY:
    # Production: Fast internal link
    POCKETBASE_URL = "http://pocketbase.railway.internal:8080"
else:
    # Local: Public link for your MacBook
    POCKETBASE_URL = "https://pocketbase-production-4854.up.railway.app"

# 3. Get Credentials from Environment (No hardcoded strings!)
# The second argument is None, which forces the app to look at the system
POCKETBASE_EMAIL = os.environ.get("POCKETBASE_EMAIL")
POCKETBASE_PASSWORD = os.environ.get("POCKETBASE_PASSWORD")

pb = PocketBase(POCKETBASE_URL)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # This runs ON STARTUP
    try:
        pb.admins.auth_with_password(POCKETBASE_EMAIL, POCKETBASE_PASSWORD)
        print("✅ Successfully authenticated with PocketBase")
    except Exception as e:
        print(f"❌ PocketBase Auth Failed: {e}")
    
    yield  # The app runs while this is held
    
    # This runs ON SHUTDOWN (optional)
    print("Shutting down...")


app = FastAPI(
    title="Worksheet Splitter - YOLOv11 Custom",
    description="AI-powered question splitting using custom-trained YOLOv11",
    version="11.0.0",
    lifespan=lifespan
)

# CORS for frontend
# Define which domains are allowed to talk to this API
origins = [
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://localhost",
    "http://127.0.0.1",
    "https://examcrop.com",
    "https://www.examcrop.com",
    "https://pdf-splitter-production-9d84.up.railway.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Questions-Count", "Content-Disposition"],
)


@app.get("/api")
def read_root():
    model_status = "trained" if os.path.exists("best.pt") else "not_trained"
    
    return {
        "status": "ok",
        "service": "yolov11-question-splitter",
        "version": "11.0.0",
        "model": "YOLOv11 Custom Trained",
        "model_status": model_status,
    }


@app.post("/split")
@app.post("/api/split")
async def split_worksheet(
    file: UploadFile = File(...),
    dpi: int = 300,
    debug: bool = False,
    conf_threshold: float = 0.15
):
    """
    Split worksheets using custom-trained YOLOv11 model.
    
    Args:
        file: PDF, JPG, JPEG, or PNG
        dpi: Processing resolution (300-600 recommended)
        debug: Save intermediate visualization images
        conf_threshold: YOLO confidence threshold (0.05-0.95)
    
    Returns:
        ZIP file with individual question PDFs
    """
    
    # Check if model exists
    if not os.path.exists("best.pt"):
        raise HTTPException(
            status_code=503,
            detail="Service Unavailable"
        )
    
    MAX_SIZE = 20 * 1024 * 1024
    MAX_PAGES = 20  # Free tier limit
    
    contents = await file.read()
    file_size_mb = len(contents) / (1024 * 1024)
    
    if len(contents) > MAX_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({file_size_mb:.1f}MB). Maximum file size is 20MB during our testing phase. For larger files, please wait for our Pro plan launch!"
        )
    
    allowed_extensions = ['.jpg', '.jpeg', '.png', '.pdf']
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"
        )
    
    if not (100 <= dpi <= 600):
        raise HTTPException(
            status_code=400,
            detail="DPI must be between 100 and 600"
        )
    
    if not (0.05 <= conf_threshold <= 0.95):
        raise HTTPException(
            status_code=400,
            detail="Confidence threshold must be between 0.05 and 0.95"
        )
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Save uploaded file
        input_path = os.path.join(temp_dir, file.filename)
        with open(input_path, 'wb') as f:
            f.write(contents)
        
        # Check page count for PDFs BEFORE processing
        if file_ext == '.pdf':
            try:
                doc = fitz.open(input_path)
                page_count = len(doc)
                doc.close()
                
                if page_count > MAX_PAGES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Your PDF has {page_count} pages. During our testing phase, we support up to {MAX_PAGES} pages per document. We're working hard to bring you unlimited pages with our Pro plan soon! For now, please split your document into smaller sections. Thank you for your understanding! 🙏"
                    )
                
                print(f"\nProcessing: {file.filename} ({file_size_mb:.1f}MB, {page_count} pages)")
            except HTTPException:
                raise
            except Exception as e:
                print(f"Warning: Could not check page count: {e}")
                print(f"\nProcessing: {file.filename} ({file_size_mb:.1f}MB)")
        else:
            # For images, assume 1 page
            print(f"\nProcessing: {file.filename} ({file_size_mb:.1f}MB, 1 page)")
        
        print(f"  DPI: {dpi}, Confidence: {conf_threshold}")
        
        # Create output directory
        output_dir = os.path.join(temp_dir, 'output')
        os.makedirs(output_dir, exist_ok=True)
        
        # Run YOLOv11 splitter
        splitter = YOLOQuestionSplitter(debug=debug, model_path="best.pt")
        
        try:
            splitter.split_worksheet(
                input_path=input_path,
                output_dir=output_dir,
                dpi=dpi,
                cleanup_temp=True,
                conf_threshold=conf_threshold
            )
        except SystemExit:
            raise HTTPException(
                status_code=422,
                detail="No questions detected"
            )
        
        # Check output
        output_files = list(Path(output_dir).glob('*.pdf'))
        
        if not output_files:
            raise HTTPException(
                status_code=422,
                detail="No questions detected"
            )
        
        print(f"✓ Successfully split into {len(output_files)} questions")
        
        # Create combined PDF with all questions
        combined_pdf = fitz.open()
        for pdf_file in sorted(output_files):
            src_pdf = fitz.open(pdf_file)
            combined_pdf.insert_pdf(src_pdf)
            src_pdf.close()
        combined_path = os.path.join(output_dir, 'all_questions_combined.pdf')
        combined_pdf.save(combined_path, garbage=4, deflate=True, clean=True, pretty=False,)
        combined_pdf.close()
        
        # Create ZIP in memory
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Add combined PDF
            zip_file.write(combined_path, 'all_questions_combined.pdf')
            
            # Add question PDFs
            for pdf_file in sorted(output_files):
                zip_file.write(pdf_file, pdf_file.name)
            
            # Add debug images if enabled
            if debug:
                for debug_file in Path(temp_dir).glob('debug_*.png'):
                    zip_file.write(debug_file, f"debug/{debug_file.name}")
        
        zip_buffer.seek(0)
        
        base_name = Path(file.filename).stem
        zip_filename = f"{base_name}_questions.zip"
        
        print(f"✓ Created ZIP: {zip_filename} ({len(zip_buffer.getvalue()) / 1024 / 1024:.2f}MB)")
        
        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={
                "Content-Disposition": f"attachment; filename={zip_filename}",
                "X-Questions-Count": str(len(output_files)),
                "X-Method": "YOLOv11-Custom",
            }
        )
    
    except HTTPException:
        raise
    
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"\n❌ ERROR: {error_trace}")
        
        raise HTTPException(
            status_code=500,
            detail=f"Processing failed: {str(e)}"
        )
    
    finally:
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"Cleanup warning: {e}")


@app.get("/api/health")
def health_check():
    model_exists = os.path.exists("best.pt")
    
    return {
        "status": "healthy" if model_exists else "model_missing",
        "method": "YOLOv11 Custom Trained",
        "model_ready": model_exists,
    }


@app.get("/api/info")
def get_info():
    return {
        "service": "YOLOv11 Question Splitter",
        "version": "11.0.0",
        "description": "Custom-trained YOLOv11 for worksheet question detection",
        "supported_formats": ["PDF", "JPG", "JPEG", "PNG"],
        "max_file_size": "20MB",
        "max_pages": "20 pages (testing phase)",
        "recommended_dpi": 300
    }


@app.post("/api/feedback")
async def collect_feedback(request: dict):
    """Collect user feedback and save to PocketBase"""
    try:
        email = request.get('email', '')
        comment = request.get('comment', '')
        timestamp = request.get('timestamp', '')
        
        # Validate
        if not email and not comment:
            return {"status": "success", "message": "No data provided"}
        
        # Save to PocketBase
        data = {
            "email": email or "",
            "feedback": comment or "",
            "timestamp": timestamp or ""
        }
        
        record = pb.collection('leads').create(data)
        
        print(f"✓ Saved lead: {email}")
        
        return {
            "status": "success",
            "message": "Thank you for your feedback!",
            "id": record.id
        }
        
    except Exception as e:
        print(f"❌ PocketBase save error: {e}")
        # Don't fail - just log the error
        return {"status": "error", "message": str(e)}


@app.get("/")
@app.get("/{path_name:path}")
async def serve_frontend(path_name: str = None):
    # 1. Define where the frontend files might live
    # Docker usually puts them in /app/frontend, but local is ../frontend
    possible_folders = [Path("frontend"), Path("../frontend"), Path(".")]
    
    # 2. If the browser is asking for a specific file (like script.js or styles.css)
    if path_name and "." in path_name:
        for folder in possible_folders:
            file_path = folder / path_name
            if file_path.exists():
                return FileResponse(file_path)

    # 3. Otherwise, serve the main index.html
    for folder in possible_folders:
        index_path = folder / "index.html"
        if index_path.exists():
            return FileResponse(index_path)

    # 4. If we get here, the files are missing
    raise HTTPException(
        status_code=404, 
        detail="Frontend files not found. Check your Docker COPY commands."
    )

if __name__ == "__main__":
    import uvicorn
    import os
    
    # This allows Railway to tell your app which port to use
    port = int(os.environ.get("PORT", 8000))
    
    print("="*70)
    print(f"Running on port: {port}")
    print("="*70)
    
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")