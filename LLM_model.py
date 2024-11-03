import streamlit as st
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor
import io

def load_model():
    """Load the Llama vision model and processor"""
    try:
        model_id = "meta-llama/Llama-3.2-11B-Vision"
        model = MllamaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        processor = AutoProcessor.from_pretrained(model_id)
        return model, processor
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def extract_text(image, model, processor):
    """Extract text from the image using Llama vision model"""
    try:
        # Prepare the prompt for text extraction only
        prompt = "<|image|><|begin_of_text|>Extract and list all text visible in this image:"
        
        # Process the image
        inputs = processor(image, prompt, return_tensors="pt").to(model.device)
        
        # Generate output
        output = model.generate(**inputs, max_new_tokens=100)
        
        # Decode and return the result
        return processor.decode(output[0])
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None

def main():
    st.title("Image Text Extraction")
    st.write("Upload an image to extract text using Llama Vision model")

    # Initialize session state for model and processor
    if 'model' not in st.session_state or 'processor' not in st.session_state:
        with st.spinner("Loading Llama Vision model..."):
            model, processor = load_model()
            if model and processor:
                st.session_state['model'] = model
                st.session_state['processor'] = processor
            else:
                st.error("Failed to load model. Please try again.")
                return

    # File uploader
    uploaded_file = st.file_uploader("Choose an image file", type=['png', 'jpg', 'jpeg'])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Add extract button
        if st.button("Extract Text"):
            with st.spinner("Extracting text..."):
                extracted_text = extract_text(
                    image, 
                    st.session_state['model'], 
                    st.session_state['processor']
                )
                
                if extracted_text:
                    st.subheader("Extracted Text:")
                    st.write(extracted_text)

if __name__ == "__main__":
    main()
