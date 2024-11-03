# app.py
import streamlit as st
from utils.video_frame_processor import video_frame_processor
from utils.ocr_processor import OCRProcessor
from utils.export import DocumentExporter
import tempfile
import os
from PIL import Image
from datetime import datetime

def initialize_session_state():
    """Initialize session state variables if they don't exist"""
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'current_text' not in st.session_state:
        st.session_state.current_text = None
    if 'current_filename' not in st.session_state:
        st.session_state.current_filename = None
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False

def add_to_history(filename: str, text: str, file_type: str):
    """Add processed item to history"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.history.append({
        'timestamp': timestamp,
        'filename': filename,
        'text': text,
        'type': file_type
    })

def clear_history():
    """Clear processing history"""
    st.session_state.history = []
    st.session_state.current_text = None
    st.session_state.current_filename = None
    st.session_state.processing_complete = False

def show_sidebar():
    """Display sidebar with history and controls"""
    with st.sidebar:
        st.header("Processing History")
        
        if len(st.session_state.history) > 0:
            if st.button("Clear All History"):
                clear_history()
            
            st.write("Previously processed files:")
            for idx, item in enumerate(st.session_state.history):
                with st.expander(f"{item['filename']} ({item['timestamp']})"):
                    st.write(f"Type: {item['type']}")
                    if st.button("Load this result", key=f"load_{idx}"):
                        st.session_state.current_text = item['text']
                        st.session_state.current_filename = item['filename']
                        st.session_state.processing_complete = True
        else:
            st.write("No processing history yet.")

def main():
    st.title("Document Text Extraction Tool")
    st.write("Extract text from images and videos using OCR")

    # Initialize session state
    initialize_session_state()
    
    # Show sidebar
    show_sidebar()

    # Initialize processors
    video_processor = video_frame_processor()
    ocr_processor = OCRProcessor()

    # File upload section
    st.header("Upload Files")
    file_type = st.radio("Select file type:", ["Image", "Video"])
    
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov', 'wmv'] if file_type == "Video" 
        else ['png', 'jpg', 'jpeg'],
        key="file_uploader"
    )

    # Process uploaded file
    if uploaded_file is not None and (
        st.session_state.current_filename != uploaded_file.name or 
        not st.session_state.processing_complete
    ):
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            file_path = tmp_file.name

        try:
            # Reset session state for new file
            st.session_state.current_filename = uploaded_file.name
            st.session_state.processing_complete = False
            
            if file_type == "Image":
                # Process image
                image = Image.open(file_path)
                st.image(image, caption="Uploaded Image", use_column_width=True)
                
                with st.spinner("Extracting text from image..."):
                    extracted_text = ocr_processor.process_image(image)
                    st.session_state.current_text = extracted_text
                    
            else:  # Video
                st.video(file_path)
                
                with st.spinner("Extracting frames and processing text..."):
                    frames = video_processor.extract_unique_frames(file_path)
                    
                    # Process each unique frame
                    text_segments = []
                    progress_bar = st.progress(0)
                    
                    for idx, frame in enumerate(frames):
                        frame_text = ocr_processor.process_image(frame.image)
                        if frame_text:
                            text_segments.append(f"[Timestamp: {frame.timestamp:.2f}s]\n{frame_text}")
                        progress_bar.progress((idx + 1) / len(frames))
                    
                    extracted_text = "\n\n".join(text_segments)
                    st.session_state.current_text = extracted_text

            # Add to history
            add_to_history(uploaded_file.name, st.session_state.current_text, file_type)
            st.session_state.processing_complete = True

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
        finally:
            # Cleanup temporary files
            os.unlink(file_path)

    # Display results section
    if st.session_state.processing_complete:
        st.header("Extracted Text")
        show_text = st.checkbox("Show Extracted Text", key="show_text")
        if show_text:
            st.text_area("", st.session_state.current_text, height=300)

        # Export options
        st.header("Export Options")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Export as PDF"):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
                    DocumentExporter.export_to_pdf(st.session_state.current_text, tmp_pdf.name)
                    with open(tmp_pdf.name, "rb") as f:
                        st.download_button(
                            "Download PDF",
                            f,
                            file_name=f"{st.session_state.current_filename}_extracted.pdf",
                            mime="application/pdf"
                        )

        with col2:
            if st.button("Export as DOCX"):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp_docx:
                    DocumentExporter.export_to_docx(st.session_state.current_text, tmp_docx.name)
                    with open(tmp_docx.name, "rb") as f:
                        st.download_button(
                            "Download DOCX",
                            f,
                            file_name=f"{st.session_state.current_filename}_extracted.docx",
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                        )

if __name__ == "__main__":
    main()
