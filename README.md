# Faded Text Restoration

## Space Link
https://huggingface.co/spaces/aabdoo234/fadedTextRestoration

## Overview
This project restores faded English text from historical documents using digital image processing and OCR techniques. It applies a series of enhancements and evaluates the results at each stage using **Tesseract OCR**. The extracted texts are then processed by a **Gemini LLM**, which predicts the original text based on all outputs. Finally, the **Levenshtein distance** is used to measure accuracy.

## Image Processing Pipeline
The restoration process involves several steps:

1. **Thresholding**  
   - Adaptive thresholding (Gaussian) or manual thresholding.  
2. **Denoising**  
   - **BM3D denoising** for noise reduction.  
   - **Median blurring** and **filtering** to remove artifacts.  
3. **Sharpening**  
   - Enhances text visibility using convolution filters.  

## Optical Character Recognition (OCR)
Tesseract OCR extracts text after each processing step:
- **Original Image Text**
- **Thresholded Image Text**
- **BM3D Denoised Image Text**
- **Denoised Image Text**
- **Sharpened Image Text**

## AI-Based Text Prediction
A **Gemini LLM** combines extracted texts to predict the original content. The model is prompted with OCR results from multiple preprocessing techniques to generate a refined text output.

## Accuracy Evaluation
The restoration accuracy is measured using **Levenshtein distance**, calculated as a percentage comparison between:
- The extracted text and the **user-provided transcription** (if available), or  
- The LLM-generated text.  

### Accuracy Metrics:
- **Original Image Accuracy**
- **Thresholded Image Accuracy**
- **BM3D Denoised Image Accuracy**
- **Denoised Image Accuracy**
- **Sharpened Image Accuracy**
- **Model Response Accuracy**  

## Interactive Gradio Interface
The project includes a **Gradio-based web app** for interactive restoration:
- **Upload an image** of a faded text document.  
- **Adjust thresholding settings** (manual or adaptive).  
- **View processed images** and extracted texts.  
- **Compare accuracy metrics** against a correct transcription.  
- **Generate a refined text output** using Gemini LLM.  

### Features:
- Image enhancement previews for each processing step.
- Side-by-side text extraction comparisons.
- Real-time accuracy calculations.
- User-friendly slider and checkbox for thresholding control.

## Installation & Execution
This application is built with Python and requires the following dependencies:
```bash
pip install gradio opencv-python numpy pytesseract google-generativeai rapidfuzz
