
import pytesseract
from PIL import Image #here PIL stands for Python Imaging Library

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

img = Image.open("images/image_hdfc.jpg")
text = pytesseract.image_to_string(img)

print("OCR Result using PyTesseract:")
print(text)