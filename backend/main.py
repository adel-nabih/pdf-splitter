"""
FastAPI Backend for Worksheet Splitter
YOLOv11 Custom Model for Question Detection
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
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

from split_pdf import YOLOQuestionSplitter

app = FastAPI(
    title="Worksheet Splitter - YOLOv11 Custom",
    description="AI-powered question splitting using custom-trained YOLOv11",
    version="11.0.0"
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
    conf_threshold: float = 0.25
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
    contents = await file.read()
    file_size_mb = len(contents) / (1024 * 1024)
    
    if len(contents) > MAX_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({file_size_mb:.1f}MB). Max: 20MB"
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
        
        print(f"\nProcessing: {file.filename} ({file_size_mb:.1f}MB)")
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
        combined_pdf.save(combined_path)
        combined_pdf.close()
        
        # create the zip file in memory
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # add the combined PDF
            zip_file.write(combined_path, 'all_questions_combined.pdf')
            
            # add question pdfs
            for pdf_file in sorted(output_files):
                zip_file.write(pdf_file, pdf_file.name)
            
            
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
        "recommended_dpi": 300
    }


current_dir = Path(__file__).parent
frontend_path = current_dir / "frontend"

if frontend_path.exists() and frontend_path.is_dir():
    from fastapi.responses import FileResponse
    
    app.mount("/static", StaticFiles(directory=frontend_path), name="static")
    
    @app.get("/")
    async def serve_frontend():
        index_file = frontend_path / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        raise HTTPException(status_code=404, detail="index.html not found")
else:
    print(f"⚠ WARNING: Frontend folder not found at {frontend_path}")

if __name__ == "__main__":
    import uvicorn
    
    print("="*70)
    print("YOLOv11 Custom Question Splitter v11.0.0")
    print("="*70)
    
    if os.path.exists("best.pt"):
        print("✓ Model found: best.pt")
    else:
        print("⚠ Model not found!")
        print("  Run: python train_yolo.py")
    
    if frontend_path.exists():
        print(f"✓ Frontend found: {frontend_path}")
    else:
        print("⚠ Frontend folder not found")
    
    print("="*70)
    print(f"\n🌐 Open in browser: http://localhost:8000\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")