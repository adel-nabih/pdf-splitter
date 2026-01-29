#!/usr/bin/env python3
"""
Convert PDF pages to high-quality PNGs for annotation in makesense.ai

Uses PyMuPDF (fitz) to render at high DPI without quality loss.
"""

import fitz
import sys
from pathlib import Path


def pdf_to_pngs(pdf_path: str, output_dir: str = "train/images", dpi: int = 300):
    """
    Convert each PDF page to a high-quality PNG.
    
    Args:
        pdf_path: Path to input PDF
        output_dir: Where to save PNGs (default: train/images)
        dpi: Resolution (300+ recommended, 600 for best quality)
    """
    
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not pdf_path.exists():
        print(f"ERROR: File not found: {pdf_path}")
        return
    
    print(f"Converting: {pdf_path.name}")
    print(f"Output: {output_dir}")
    print(f"DPI: {dpi}")
    print("-" * 60)
    
    # Open PDF
    doc = fitz.open(pdf_path)
    
    # Base filename (remove .pdf extension)
    base_name = pdf_path.stem
    
    # Get page count before conversion
    total_pages = len(doc)
    
    # Convert each page
    for page_num in range(total_pages):
        page = doc[page_num]
        
        # Calculate zoom for desired DPI
        # PDF default is 72 DPI, so zoom = target_dpi / 72
        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)
        
        # Render page at high resolution
        pix = page.get_pixmap(matrix=mat, alpha=False)
        
        # Generate filename: worksheet_01_page_001.png
        output_filename = f"{base_name}_page_{page_num+1:03d}.png"
        output_path = output_dir / output_filename
        
        # Save as PNG
        pix.save(output_path)
        
        print(f"✓ Page {page_num+1}/{total_pages}: {output_filename}")
        print(f"  Size: {pix.width}x{pix.height} pixels")
    
    doc.close()
    
    print("-" * 60)
    print(f"✓ Converted {total_pages} pages")
    print(f"\nNext steps:")
    print(f"1. Go to https://www.makesense.ai/")
    print(f"2. Upload images from: {output_dir}")
    print(f"3. Annotate questions with bounding boxes")
    print(f"4. Export as YOLO format to train/labels/")
    print(f"5. Run: python train_yolo.py")


def batch_convert(pdf_folder: str, output_dir: str = "train/images", dpi: int = 300):
    """
    Convert all PDFs in a folder.
    
    Args:
        pdf_folder: Folder containing PDFs
        output_dir: Where to save PNGs
        dpi: Resolution
    """
    
    pdf_folder = Path(pdf_folder)
    pdf_files = list(pdf_folder.glob("*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in: {pdf_folder}")
        return
    
    print(f"Found {len(pdf_files)} PDF files")
    print("=" * 60)
    
    for i, pdf_file in enumerate(pdf_files, 1):
        print(f"\n[{i}/{len(pdf_files)}] Processing: {pdf_file.name}")
        pdf_to_pngs(pdf_file, output_dir, dpi)
    
    print("\n" + "=" * 60)
    print(f"✓ Batch conversion complete!")
    print(f"Total images: {len(list(Path(output_dir).glob('*.png')))}")


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Single file:  python pdf_to_png.py worksheet.pdf [output_dir] [dpi]")
        print("  Batch:        python pdf_to_png.py --batch pdf_folder/ [output_dir] [dpi]")
        print()
        print("Examples:")
        print("  python pdf_to_png.py worksheet.pdf")
        print("  python pdf_to_png.py worksheet.pdf train/images 600")
        print("  python pdf_to_png.py --batch worksheets/ train/images 300")
        print()
        print("DPI recommendations:")
        print("  300 - Good quality, smaller files (default)")
        print("  400 - High quality")
        print("  600 - Maximum quality, large files")
        sys.exit(1)
    
    # Parse arguments
    if sys.argv[1] == "--batch":
        if len(sys.argv) < 3:
            print("ERROR: --batch requires a folder path")
            sys.exit(1)
        
        pdf_folder = sys.argv[2]
        output_dir = sys.argv[3] if len(sys.argv) > 3 else "train/images"
        dpi = int(sys.argv[4]) if len(sys.argv) > 4 else 300
        
        batch_convert(pdf_folder, output_dir, dpi)
    else:
        pdf_path = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else "train/images"
        dpi = int(sys.argv[3]) if len(sys.argv) > 3 else 300
        
        pdf_to_pngs(pdf_path, output_dir, dpi)


if __name__ == "__main__":
    main()