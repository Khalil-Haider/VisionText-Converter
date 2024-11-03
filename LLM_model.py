import streamlit as st
from PIL import Image
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
import requests
from io import BytesIO

def load_llama_model():
    """Load the Llama model and tokenizer"""
    tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
    return model, tokenizer

def extract_text_from_image(image, model, tokenizer):
    """Extract text from image using Llama model"""
    # Convert image to format suitable for model
    if isinstance(image, str):
        # If image is a URL
        response = requests.get(image)
        image = Image.open(BytesIO(response.content))
    elif isinstance(image, BytesIO):
        # If image is uploaded
        image = Image.open(image)
    
    # Create prompt for the model
    prompt = "Extract only the text from this image, without any additional description or commentary:"
    
    # Process image and generate text
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        inputs.input_ids,
        max_length=200,
        num_return_sequences=1,
        temperature=0.7
    )
    
    # Decode the output
    extracted_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return extracted_text

def main():
    st.title("Image Text Extraction App")
    st.write("Upload an image to extract text using Llama model")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image file", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        try:
            # Display the uploaded image
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
            
            # Add a button to trigger text extraction
            if st.button("Extract Text"):
                with st.spinner("Loading Llama model..."):
                    model, tokenizer = load_llama_model()
                
                with st.spinner("Extracting text..."):
                    extracted_text = extract_text_from_image(uploaded_file, model, tokenizer)
                    
                # Display results
                st.subheader("Extracted Text:")
                st.write(extracted_text)
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
