# 📄 Worksheet Splitter

AI-powered worksheet question splitter using custom-trained YOLOv26.

Split exam worksheets into individual questions automatically using computer vision.

## ✨ Features

- 🤖 Custom YOLOv26 model for question detection
- 📦 Outputs combined PDF + individual questions
- 🚀 Fast processing (GPU-accelerated)
- 🔒 Rate limiting (10 requests/hour per IP)
- 📏 20MB file size limit
- 🎯 92.9% detection accuracy (mAP50)

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Tesseract OCR installed

**macOS:**
```bash
brew install tesseract
```

**Ubuntu/Debian:**
```bash
sudo apt-get install tesseract-ocr
```

**Windows:**
Download from: https://github.com/UB-Mannheim/tesseract/wiki

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/worksheet-splitter.git
cd worksheet-splitter

# Install dependencies
pip install -r requirements.txt

# Download trained model (you'll need to add this)
# Place best.pt in the backend/ folder
```

### Running Locally

```bash
# Start backend
cd backend
python main.py

# Open browser
open http://localhost:8000
```

## 📁 Project Structure

```
worksheet-splitter/
├── backend/
│   ├── main.py              # FastAPI server
│   ├── split_pdf.py         # Core splitting logic
│   ├── best.pt              # Trained YOLOv26 model (not in git)
│   └── requirements.txt     # Python dependencies
├── frontend/
│   └── index.html          # Web interface
├── .gitignore
└── README.md
```

## 🎯 Training Your Own Model

See [TRAINING.md](TRAINING.md) for detailed instructions on:
- Annotating your worksheets
- Training YOLOv26 on Google Colab
- Deploying your custom model

## 🚀 Deployment

### Option 1: Render.com (Recommended - Free Tier)

**Backend:**
1. Create account on [render.com](https://render.com)
2. New Web Service
3. Connect GitHub repo
4. Configure:
   - **Build Command:** `pip install -r backend/requirements.txt`
   - **Start Command:** `cd backend && uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Root Directory:** `backend`
5. Add environment variable: `PORT=8000`
6. Deploy!

**Frontend:**
1. New Static Site
2. Connect same GitHub repo
3. Configure:
   - **Build Command:** (leave empty)
   - **Publish Directory:** `frontend`
4. Update `index.html` with your backend URL

### Option 2: Railway.app

Similar to Render, free tier available.

### Option 3: DigitalOcean App Platform

$5/month, more reliable than free tiers.

## ⚙️ Configuration

Edit `backend/main.py`:

```python
MAX_REQUESTS_PER_HOUR = 10    # Rate limit
MAX_FILE_SIZE_MB = 20         # File size limit
```

## 🔒 Security Notes

- ✅ No API keys needed (YOLO runs locally)
- ✅ Rate limiting enabled
- ✅ File size limits enforced
- ✅ No data stored after processing
- ⚠️ Don't commit `best.pt` to git (too large)
- ⚠️ Don't commit training data

## 📊 Model Performance

- **mAP50:** 92.9%
- **Precision:** 85.7%
- **Recall:** 88.3%
- **Training:** 100 epochs on custom dataset

## 🐛 Troubleshooting

**"Model not found" error:**
- Make sure `best.pt` is in `backend/` folder
- Re-train or download the model

**Rate limit errors:**
- Wait 1 hour or increase `MAX_REQUESTS_PER_HOUR`

**Large files failing:**
- Increase `MAX_FILE_SIZE_MB` if you have resources
- Or split your PDF into smaller chunks

## 📝 License

MIT License - feel free to use for personal or commercial projects.

## 🙏 Acknowledgments

- Built with [Ultralytics YOLOv26](https://github.com/ultralytics/ultralytics)
- PDF processing: [PyMuPDF](https://pymupdf.readthedocs.io/)
- OCR: [Tesseract](https://github.com/tesseract-ocr/tesseract)