# utils/ocr_processor.py
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from PIL import Image
import cv2
import numpy as np
import tempfile
import textwrap
from typing import Union, List

class OCRProcessor:
    def __init__(self, det_arch: str = 'db_resnet50', reco_arch: str = 'crnn_vgg16_bn'):
        self.model = ocr_predictor(det_arch=det_arch, reco_arch=reco_arch, pretrained=True)

    def process_image(self, image: Union[str, np.ndarray, Image.Image]) -> str:
        """Process a single image and extract text"""
        try:
            if isinstance(image, np.ndarray):
                image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            if isinstance(image, Image.Image):
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                    image.save(tmp.name)
                    doc = DocumentFile.from_images(tmp.name)
            else:
                doc = DocumentFile.from_images(image)

            result = self.model(doc)
            
            all_text = []
            for page in result.pages:
                for block in page.blocks:
                    for line in block.lines:
                        for word in line.words:
                            all_text.append(word.value)
            
            combined_text = ' '.join(all_text)
            formatted_text = textwrap.fill(combined_text, width=80)
            
            return formatted_text

        except Exception as e:
            raise Exception(f"Error processing image: {str(e)}")

