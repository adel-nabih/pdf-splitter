#!/usr/bin/env python3
"""
Test script to verify backend is working correctly
Run this in the backend directory: python test_backend.py
"""

import requests
import sys
from pathlib import Path

def test_backend():
    print("="*70)
    print("BACKEND DIAGNOSTICS")
    print("="*70)
    print()
    
    # Check if model exists
    model_path = Path("best.pt")
    if model_path.exists():
        print(f"✓ Model found: {model_path} ({model_path.stat().st_size / 1024 / 1024:.1f} MB)")
    else:
        print("✗ Model NOT found: best.pt")
        print("  Run: python train_yolo.py")
    
    print()
    
    # Check if split_pdf.py exists
    split_pdf = Path("split_pdf.py")
    if split_pdf.exists():
        print(f"✓ Found: split_pdf.py")
    else:
        print("✗ NOT found: split_pdf.py")
    
    print()
    
    # Test backend endpoints
    base_url = "http://localhost:8000"
    
    print(f"Testing backend at: {base_url}")
    print("-"*70)
    
    # Test 1: Root endpoint
    try:
        response = requests.get(f"{base_url}/api", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✓ GET /api - {response.status_code}")
            print(f"  Service: {data.get('service')}")
            print(f"  Version: {data.get('version')}")
            print(f"  Model Status: {data.get('model_status')}")
        else:
            print(f"✗ GET /api - {response.status_code}")
    except Exception as e:
        print(f"✗ GET /api - ERROR: {e}")
        print()
        print("Backend is NOT running!")
        print("Start it with: python main.py")
        return False
    
    print()
    
    # Test 2: Health endpoint
    try:
        response = requests.get(f"{base_url}/api/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✓ GET /api/health - {response.status_code}")
            print(f"  Status: {data.get('status')}")
            print(f"  Model Ready: {data.get('model_ready')}")
        else:
            print(f"✗ GET /api/health - {response.status_code}")
    except Exception as e:
        print(f"✗ GET /api/health - ERROR: {e}")
    
    print()
    
    # Test 3: Info endpoint
    try:
        response = requests.get(f"{base_url}/api/info", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✓ GET /api/info - {response.status_code}")
            print(f"  Supported formats: {data.get('supported_formats')}")
            print(f"  Max file size: {data.get('max_file_size')}")
        else:
            print(f"✗ GET /api/info - {response.status_code}")
    except Exception as e:
        print(f"✗ GET /api/info - ERROR: {e}")
    
    print()
    print("="*70)
    print()
    
    # Check CORS
    print("CORS Configuration Check:")
    print("-"*70)
    try:
        response = requests.options(f"{base_url}/split", timeout=5)
        cors_headers = {
            'Access-Control-Allow-Origin': response.headers.get('Access-Control-Allow-Origin', 'NOT SET'),
            'Access-Control-Allow-Methods': response.headers.get('Access-Control-Allow-Methods', 'NOT SET'),
            'Access-Control-Allow-Headers': response.headers.get('Access-Control-Allow-Headers', 'NOT SET'),
        }
        
        for header, value in cors_headers.items():
            status = "✓" if value != "NOT SET" else "✗"
            print(f"{status} {header}: {value}")
    except Exception as e:
        print(f"✗ CORS check failed: {e}")
    
    print()
    print("="*70)
    print()
    
    if not model_path.exists():
        print("⚠️  WARNING: Model file (best.pt) not found!")
        print("   The backend will reject all requests until you train the model.")
        print("   Run: python train_yolo.py")
        print()
    
    print("SUMMARY:")
    print("-"*70)
    print("✓ Backend is running")
    print(f"✓ Accessible at: {base_url}")
    print()
    print("Next steps:")
    print("1. Open your browser to: http://localhost:8000")
    print("2. Upload a test PDF/image")
    print("3. Check the browser console (F12) for any errors")
    print()
    
    return True


if __name__ == "__main__":
    try:
        success = test_backend()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted")
        sys.exit(1)