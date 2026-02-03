#!/usr/bin/env python3
"""
YOLOv26-Based Question Splitter with Custom Training
FIXED: Handles PDF rotation properly
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Optional
import re
import fitz
from PIL import Image
import cv2
import numpy as np
import pandas as pd
import pytesseract
import tempfile

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("ERROR: ultralytics not installed. Run: pip install ultralytics")

QUESTION_PATTERNS = [
    r'^\s*\d+[\.\)]',  # "1." or "1)"
    r'^\s*Q\s*\d+',     # "Q1" or "Q 1"
    r'^\s*\(\d+\)',     # "(1)"
]


class YOLOQuestionSplitter:
    """YOLOv26-based worksheet question splitter with rotation handling"""
    
    def __init__(self, debug: bool = False, model_path: str = None):
        self.debug = debug
        self.model_path = model_path or 'best.pt'
        
        if not YOLO_AVAILABLE:
            raise ImportError("ultralytics not available. Run: pip install ultralytics")
        
        if os.path.exists(self.model_path):
            print(f"Loading custom trained model: {self.model_path}")
            self.model = YOLO(self.model_path)
            self.use_custom_model = True
        else:
            print(f"ERROR: Model '{self.model_path}' not found.")
            raise FileNotFoundError(f"Model not found: {self.model_path}")
    
    def convert_to_pdf(self, input_path: str) -> str:
        """Convert image to PDF if needed"""
        file_ext = Path(input_path).suffix.lower()
        if file_ext == '.pdf':
            return input_path
        
        print("  Converting image to PDF...")
        img = Image.open(input_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        temp_pdf_path = temp_pdf.name
        temp_pdf.close()
        
        img.save(temp_pdf_path, "PDF", resolution=300.0, quality=95)
        return temp_pdf_path
    
    def pdf_to_images(self, pdf_path: str, dpi: int = 300) -> List[Dict]:
        """Convert PDF pages to OpenCV images at high quality"""
        doc = fitz.open(pdf_path)
        page_data = []
        
        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)
        
        print(f"  Converting {len(doc)} page(s) to images at {dpi} DPI...")
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Just render - don't mess with rotation
            pix = page.get_pixmap(matrix=mat, alpha=False)
            
            img_data = pix.samples
            img = np.frombuffer(img_data, dtype=np.uint8)
            img = img.reshape(pix.height, pix.width, pix.n)
            
            if pix.n == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            elif pix.n == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            
            page_data.append({
                'image': img,
                'width': pix.width,
                'height': pix.height
            })
            
            print(f"    Page {page_num+1}: {pix.width}x{pix.height} pixels")
        
        doc.close()
        return page_data
    
    def detect_questions_with_yolo(self, image: np.ndarray, conf_threshold: float = 0.25) -> List[Dict]:
        """Use trained YOLOv26 model to detect question blocks"""
        print(f"  Running YOLOv26 detection (conf={conf_threshold})...")
        
        results = self.model(image, conf=conf_threshold, verbose=False)
        
        blocks = []
        if len(results) > 0:
            result = results[0]
            
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                
                for i, (box, conf, cls) in enumerate(zip(boxes, confidences, classes)):
                    if cls != 0:  # class 0 is 'question'
                        continue
                    x1, y1, x2, y2 = box
                    w = x2 - x1
                    h = y2 - y1
                    
                    blocks.append({
                        'x': int(x1),
                        'y': int(y1),
                        'w': int(w),
                        'h': int(h),
                        'confidence': float(conf),
                        'class': int(cls)
                    })
        
        print(f"    Found {len(blocks)} question blocks")
        return blocks
    
    def ocr_block(self, image: np.ndarray, block: Dict) -> str:
        """Extract text from a block using OCR"""
        x, y, w, h = block['x'], block['y'], block['w'], block['h']
        
        h_img, w_img = image.shape[:2]
        x = max(0, min(x, w_img - 1))
        y = max(0, min(y, h_img - 1))
        x2 = max(0, min(x + w, w_img))
        y2 = max(0, min(y + h, h_img))
        
        roi = image[y:y2, x:x2]
        if roi.size == 0:
            return ""
        
        pil_roi = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        return pytesseract.image_to_string(pil_roi).strip()
    
    def find_question_number(self, text: str) -> Optional[int]:
        """Extract question number from OCR text"""
        if not text:
            return None
        
        lines = text.split('\n')
        for line in lines[:10]:
            line = line.strip()
            for pattern in QUESTION_PATTERNS:
                if re.search(pattern, line):
                    numbers = re.findall(r'\d+', line)
                    if numbers:
                        return int(numbers[0])
        return None
    
    def assign_question_numbers(self, blocks: List[Dict], image: np.ndarray, 
                               page_num: int, page_offset: int = 0) -> pd.DataFrame:
        """
        Assign question numbers based on POSITION ONLY (top-to-bottom, left-to-right).
        No OCR needed - just use spatial ordering.
        
        Args:
            blocks: List of detected bounding boxes
            image: Source image (not used, kept for compatibility)
            page_num: Current page number
            page_offset: Starting question number offset
        """
        print("  Assigning question numbers by position...")
        
        if len(blocks) == 0:
            return pd.DataFrame()
        
        # Remove duplicate detections (overlapping boxes)
        # Two boxes overlap if IoU > 0.5
        def calc_iou(box1, box2):
            x1 = max(box1['x'], box2['x'])
            y1 = max(box1['y'], box2['y'])
            x2 = min(box1['x'] + box1['w'], box2['x'] + box2['w'])
            y2 = min(box1['y'] + box1['h'], box2['y'] + box2['h'])
            
            if x2 < x1 or y2 < y1:
                return 0.0
            
            intersection = (x2 - x1) * (y2 - y1)
            area1 = box1['w'] * box1['h']
            area2 = box2['w'] * box2['h']
            union = area1 + area2 - intersection
            
            return intersection / union if union > 0 else 0
        
        # Remove duplicates using Non-Maximum Suppression
        unique_blocks = []
        sorted_by_conf = sorted(blocks, key=lambda b: b.get('confidence', 1.0), reverse=True)
        
        for block in sorted_by_conf:
            is_duplicate = False
            for unique in unique_blocks:
                if calc_iou(block, unique) > 0.5:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_blocks.append(block)
        
        if self.debug:
            print(f"    Removed {len(blocks) - len(unique_blocks)} duplicate detections")
        
        # Sort by position: top to bottom, then left to right
        # Use vertical position primarily (same row if y difference < 50 pixels)
        def sort_key(b):
            # Round y to nearest 50 pixels to group items in same row
            row = round(b['y'] / 50)
            return (row, b['x'])
        
        sorted_blocks = sorted(unique_blocks, key=sort_key)
        
        # Assign sequential numbers based on position
        questions = []
        for i, block in enumerate(sorted_blocks):
            q_num = page_offset + i + 1
            
            questions.append({
                'question_num': q_num,
                'page_num': page_num,
                'x': block['x'],
                'y': block['y'],
                'w': block['w'],
                'h': block['h'],
                'confidence': block.get('confidence', 1.0)
            })
            
            if self.debug:
                print(f"    Q{q_num}: position ({block['x']}, {block['y']}), conf={block.get('confidence', 1.0):.2f}")
        
        df = pd.DataFrame(questions)
        print(f"    Assigned {len(df)} unique questions")
        return df
    
    def crop_from_pdf(self, pdf_path: str, df: pd.DataFrame, page_num: int, 
                     dpi: int, output_dir: str):
        """
        Crop questions from PDF using bounding boxes.
        Since we removed rotation before detection, coordinates are already correct.
        """
        doc = fitz.open(pdf_path)
        page = doc[page_num]
        
        # Remove rotation if present (same as during detection)
        if page.rotation != 0:
            page.set_rotation(0)
        
        scale = 72 / dpi
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        original_rect = page.rect
        
        for _, row in df.iterrows():
            q_num = int(row['question_num'])
            
            # Convert pixel coordinates to PDF points
            x_pts = row['x'] * scale
            y_pts = row['y'] * scale
            w_pts = row['w'] * scale
            h_pts = row['h'] * scale
            
            # Add margin
            margin = 5
            x_pts = max(0, x_pts - margin)
            y_pts = max(0, y_pts - margin)
            w_pts += 2 * margin
            h_pts += 2 * margin
            
            # Ensure we don't go out of bounds
            x2_pts = min(original_rect.width, x_pts + w_pts)
            y2_pts = min(original_rect.height, y_pts + h_pts)
            
            rect = fitz.Rect(x_pts, y_pts, x2_pts, y2_pts)
            
            # Create new PDF with cropped question
            out_pdf = fitz.open()
            out_page = out_pdf.new_page(width=rect.width, height=rect.height)
            
            # Clip the region from source page
            out_page.show_pdf_page(out_page.rect, doc, page_num, clip=rect)
            
            # CRITICAL FIX: Force output page rotation to 0
            # This ensures output PDFs are always right-side-up
            out_page.set_rotation(0)
            
            filepath = Path(output_dir) / f"question_{q_num:02d}.pdf"
            out_pdf.save(filepath, garbage=4, deflate=True, clean=True, pretty=False,)
            out_pdf.close()
            
            if self.debug:
                print(f"  Saved: question_{q_num:02d}.pdf (rotation corrected)")
        
        doc.close()
    
    def visualize(self, image: np.ndarray, df: pd.DataFrame, output_path: str):
        """Save visualization of detected questions"""
        vis = image.copy()
        for _, row in df.iterrows():
            x, y, w, h = int(row['x']), int(row['y']), int(row['w']), int(row['h'])
            q = int(row['question_num'])
            conf = row.get('confidence', 1.0)
            
            cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 3)
            label = f"Q{q} ({conf:.2f})"
            cv2.putText(vis, label, (x+5, y+30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imwrite(output_path, vis)
        print(f"  Saved visualization: {output_path}")
    
    def split_worksheet(self, input_path: str, output_dir: str, 
                       dpi: int = 300, cleanup_temp: bool = True,
                       conf_threshold: float = 0.25):
        """
        Main pipeline to split worksheet into questions.
        FIXED: Properly handles PDF rotation.
        """
        print(f"Processing: {input_path}\n")
        
        temp_pdf = None
        pdf_path = self.convert_to_pdf(input_path)
        if pdf_path != input_path:
            temp_pdf = pdf_path
        
        try:
            # Convert all pages to images (with rotation info)
            page_data_list = self.pdf_to_images(pdf_path, dpi)
            
            all_questions_found = False
            total_questions = 0
            question_offset = 0
            
            for page_num, page_data in enumerate(page_data_list):
                image = page_data['image']
                
                print(f"\n{'='*60}")
                print(f"Page {page_num + 1} of {len(page_data_list)}")
                print('='*60)
                
                # Detect questions with YOLO
                blocks = self.detect_questions_with_yolo(image, conf_threshold)
                
                if not blocks:
                    print("  No questions detected on this page")
                    continue
                
                # Assign question numbers (by position, not OCR)
                df = self.assign_question_numbers(blocks, image, page_num, question_offset)
                
                if len(df) == 0:
                    print("  No valid questions after processing")
                    continue
                
                all_questions_found = True
                page_question_count = len(df)
                total_questions += page_question_count
                
                max_q_num = df['question_num'].max()
                question_offset = max_q_num
                
                # Save visualization if debug mode
                if self.debug:
                    debug_path = f"debug_yolo_page_{page_num+1}.png"
                    self.visualize(image, df, debug_path)
                
                # Crop and save questions
                self.crop_from_pdf(pdf_path, df, page_num, dpi, output_dir)
                
                print(f"  ✓ Extracted {page_question_count} questions from page {page_num+1}")
            
            if not all_questions_found:
                print("\n❌ No questions detected!")
                sys.exit(1)
            else:
                print("\n" + "="*60)
                print(f"✓ SUCCESS!")
                print(f"  Total pages processed: {len(page_data_list)}")
                print(f"  Total questions extracted: {total_questions}")
                print(f"  Output directory: {output_dir}")
                print("="*60)
        
        finally:
            if temp_pdf and cleanup_temp and os.path.exists(temp_pdf):
                os.unlink(temp_pdf)


# Keep the original class name for backward compatibility
YOLOQuestionSplitter = YOLOQuestionSplitter


def main():
    if len(sys.argv) < 3:
        print("Usage: python split_pdf.py <input> <output_dir> [--debug] [--model path] [--conf 0.25] [--dpi 300]")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_dir = sys.argv[2]
    
    model_path = 'best.pt'
    conf_threshold = 0.25
    dpi = 300
    debug = '--debug' in sys.argv
    
    if '--model' in sys.argv:
        idx = sys.argv.index('--model')
        if idx + 1 < len(sys.argv):
            model_path = sys.argv[idx + 1]
    
    if '--conf' in sys.argv:
        idx = sys.argv.index('--conf')
        if idx + 1 < len(sys.argv):
            conf_threshold = float(sys.argv[idx + 1])
    
    if '--dpi' in sys.argv:
        idx = sys.argv.index('--dpi')
        if idx + 1 < len(sys.argv):
            dpi = int(sys.argv[idx + 1])
    
    splitter = YOLOQuestionSplitter(debug=debug, model_path=model_path)
    splitter.split_worksheet(input_path, output_dir, dpi=dpi, conf_threshold=conf_threshold)


if __name__ == "__main__":
    main()