# app.py
import streamlit as st
from utils.video_processor import video_frame_processor
from utils.ocr_processor import OCRProcessor
from utils.export import DocumentExporter
import tempfile
import os
from PIL import Image

def main():
    st.title("Document Text Extraction Tool")
    st.write("Extract text from images and videos using OCR")

    # Initialize processors
    video_processor = video_frame_processor()
    ocr_processor = OCRProcessor()

    # File upload section
    st.header("Upload Files")
    file_type = st.radio("Select file type:", ["Image", "Video"])
    
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov', 'wmv'] if file_type == "Video" 
        else ['png', 'jpg', 'jpeg']
    )

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            file_path = tmp_file.name

        try:
            extracted_text = ""
            
            if file_type == "Image":
                # Process image
                image = Image.open(file_path)
                st.image(image, caption="Uploaded Image", use_column_width=True)
                
                with st.spinner("Extracting text from image..."):
                    extracted_text = ocr_processor.process_image(image)
                
            else:  # Video
                st.video(file_path)
                
                with st.spinner("Extracting frames and processing text..."):
                    frames = video_processor.extract_unique_frames(file_path)
                    
                    # Process each unique frame
                    text_segments = []
                    for frame in frames:
                        frame_text = ocr_processor.process_image(frame.image)
                        if frame_text:
                            text_segments.append(f"[Timestamp: {frame.timestamp:.2f}s]\n{frame_text}")
                    
                    extracted_text = "\n\n".join(text_segments)

            # Display results section
            st.header("Extracted Text")
            if st.button("Show/Hide Extracted Text"):
                st.text_area("", extracted_text, height=300)

            # Export options
            st.header("Export Options")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Export as PDF"):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
                        DocumentExporter.export_to_pdf(extracted_text, tmp_pdf.name)
                        with open(tmp_pdf.name, "rb") as f:
                            st.download_button(
                                "Download PDF",
                                f,
                                file_name="extracted_text.pdf"
                            )

            with col2:
                if st.button("Export as DOCX"):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp_docx:
                        DocumentExporter.export_to_docx(extracted_text, tmp_docx.name)
                        with open(tmp_docx.name, "rb") as f:
                            st.download_button(
                                "Download DOCX",
                                f,
                                file_name="extracted_text.docx"
                            )

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
        finally:
            # Cleanup temporary files
            os.unlink(file_path)

if __name__ == "__main__":
    main()

