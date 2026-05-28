#!/usr/bin/env python3
from __future__ import annotations
OCR_EXT={'.pdf','.png','.jpg','.jpeg','.webp','.gif','.bmp'}
IMAGE_EXT={'.png','.jpg','.jpeg','.webp','.gif','.bmp'}

def classify_ocr(path: str, *, ocr_available: bool=False) -> dict:
    ext='.'+path.lower().rsplit('.',1)[-1] if '.' in path else ''
    if ext not in OCR_EXT:
        return {'ocr_status':'OCR_NOT_REQUIRED','ocr_required':False,'ocr_available':ocr_available,'reason':''}
    if ocr_available:
        return {'ocr_status':'OCR_READY','ocr_required':True,'ocr_available':True,'reason':'OCR_REQUIRED'}
    return {'ocr_status':'OCR_BLOCKED','ocr_required':True,'ocr_available':False,'reason':'OCR_REQUIRED_BACKEND_NOT_WIRED'}
