import gradio as gr
import cv2
import numpy as np
import pytesseract
import re
import google.generativeai as genai
from rapidfuzz.distance import Levenshtein
import os

os.system('apt-get update && apt-get install -y tesseract-ocr')
# Configure Generative AI
OPENAI_API_KEY = os.getenv("API_KEY")
genai.configure(api_key=OPENAI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

# Image processing functions
def threshold_image(img, threshold_value=None):
    if threshold_value is None:  # Adaptive thresholding
        thresholded_image = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                  cv2.THRESH_BINARY, 11, 2)
    else:  # Manual thresholding
        _, thresholded_image = cv2.threshold(img, threshold_value, 255, cv2.THRESH_BINARY)
    return thresholded_image

def bm3d_denoising(img, sigma_psd=55):
    return cv2.fastNlMeansDenoising(img, None, sigma_psd)

def remove_noise(img, kernel_size=3):
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size**2)
    denoised = cv2.filter2D(img, -1, kernel)
    return cv2.medianBlur(denoised, 3)

def sharpen_image(img):
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    return cv2.filter2D(img, -1, kernel)

def remove_extra_spaces_and_lines(text):
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'\n\s*\n', '\n\n', text)
    return text

def calculate_accuracy(text1, text2):
    # matcher = difflib.SequenceMatcher(None, generated_text, transcribed_text)
    # return matcher.ratio()
    distance = Levenshtein.distance(text1, text2)
    max_length = max(len(text1), len(text2))
    accuracy = (1 - (distance / max_length))
    return accuracy

# Gradio app
def process_image(image, threshold_value=None, correct_transcription=None):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Process the image
    thresholded = threshold_image(img, threshold_value)
    bm3d_denoised_image = bm3d_denoising(thresholded)
    denoised = remove_noise(thresholded)
    sharpened_image = sharpen_image(bm3d_denoised_image)

    # OCR
    original_text = pytesseract.image_to_string(img)
    thresholded_text = pytesseract.image_to_string(thresholded)
    bm3d_denoised_text = pytesseract.image_to_string(bm3d_denoised_image)
    denoised_text = pytesseract.image_to_string(denoised)
    sharpened_text = pytesseract.image_to_string(sharpened_image)
    
    # Clean up text
    original_text = remove_extra_spaces_and_lines(original_text)
    thresholded_text = remove_extra_spaces_and_lines(thresholded_text)
    bm3d_denoised_text = remove_extra_spaces_and_lines(bm3d_denoised_text)
    denoised_text = remove_extra_spaces_and_lines(denoised_text)
    sharpened_text = remove_extra_spaces_and_lines(sharpened_text)

    # Generative AI model response
    user_prompt = user_prompt = f"""
    below are the output texts of OCR on multiple image processing techniques of a faded image with text written in English, can you use all the texts to predict the original text, provide only the text.
    Pre-Processing Image Text:
    {original_text}
    Sharpened Image Text:
    {sharpened_text}
    Thresholded Image Text:
    {thresholded_text}
    BM3D Denoised Image Text:
    {bm3d_denoised_text}
    Denoised Image Text:
    {denoised_text}
    """  
    response = model.generate_content(user_prompt)
    model_text = response.text

    if not correct_transcription:
        correct_transcription = model_text
    # Accuracy metrics
    if correct_transcription:
        original_accuracy = calculate_accuracy(original_text, correct_transcription)
        thresholded_accuracy = calculate_accuracy(thresholded_text, correct_transcription)
        bm3d_denoised_accuracy = calculate_accuracy(bm3d_denoised_text, correct_transcription)
        denoised_accuracy = calculate_accuracy(denoised_text, correct_transcription)
        sharpened_accuracy = calculate_accuracy(sharpened_text, correct_transcription)
        model_accuracy = calculate_accuracy(model_text, correct_transcription)
        accuracy_metrics = f"""
        Original Image Accuracy: {original_accuracy:.2%}
        Thresholded Image Accuracy: {thresholded_accuracy:.2%}
        BM3D Denoised Image Accuracy: {bm3d_denoised_accuracy:.2%}
        Denoised Image Accuracy: {denoised_accuracy:.2%}
        Sharpened Image Accuracy: {sharpened_accuracy:.2%}
        Model Response Accuracy: {model_accuracy:.2%}
        """
    else:
        accuracy_metrics = "No correct transcription provided."

    # Return results
    return (
        image, thresholded, bm3d_denoised_image, denoised, sharpened_image,
        original_text, thresholded_text, bm3d_denoised_text, denoised_text, sharpened_text,
        model_text, accuracy_metrics
    )

# Interface
with gr.Blocks() as demo:
    with gr.Row():
        gr.Markdown("## Faded text restoration")
    with gr.Row():
        gr.Markdown("""
        ### Legend
        - **Model Response**: Text generated by the Generative AI model.
        - **Accuracy Metrics**: Comparison of OCR results with the provided correct transcription if provided, otherwise with the model response.
        """)
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label="Upload Image", type="numpy")
            threshold_slider = gr.Slider(label="Threshold Value", minimum=0, maximum=255, step=1, value=242)
            adaptive_checkbox = gr.Checkbox(label="Use Adaptive Thresholding", value=False)
            transcription_input = gr.Textbox(label="Correct Transcription (Optional)")
            process_button = gr.Button("Process Image")

        with gr.Column():
            tabs = gr.Tabs()
            with tabs:
                with gr.TabItem("Original"):
                    original_image_display = gr.Image(label="Original Image")
                    original_text_display = gr.Textbox(label="Original Image Text", lines=5)
                with gr.TabItem("Thresholded"):
                    thresholded_image_display = gr.Image(label="Thresholded Image")
                    thresholded_text_display = gr.Textbox(label="Thresholded Image Text", lines=5)
                with gr.TabItem("BM3D Denoised"):
                    bm3d_denoised_image_display = gr.Image(label="BM3D Denoised Image")
                    bm3d_denoised_text_display = gr.Textbox(label="BM3D Denoised Image Text", lines=5)
                with gr.TabItem("Denoised"):
                    denoised_image_display = gr.Image(label="Denoised Image")
                    denoised_text_display = gr.Textbox(label="Denoised Image Text", lines=5)
                with gr.TabItem("Sharpened"):
                    sharpened_image_display = gr.Image(label="Sharpened Image")
                    sharpened_text_display = gr.Textbox(label="Sharpened Image Text", lines=5)
            accuracy_output = gr.Textbox(label="Accuracy Metrics")
            model_text_display = gr.Textbox(label="Model Response Text")

    # Link button to processing function
    def update_process(image, threshold_value, use_adaptive, correct_transcription):
        threshold_value = None if use_adaptive else threshold_value
        return process_image(image, threshold_value, correct_transcription)

    process_button.click(
        update_process,
        inputs=[image_input, threshold_slider, adaptive_checkbox, transcription_input],
        outputs=[
            original_image_display, thresholded_image_display,
            bm3d_denoised_image_display, denoised_image_display, 
            sharpened_image_display, original_text_display,
            thresholded_text_display, bm3d_denoised_text_display,
            denoised_text_display, sharpened_text_display,
            model_text_display, accuracy_output
            ],
    )

# Launch app
demo.launch()