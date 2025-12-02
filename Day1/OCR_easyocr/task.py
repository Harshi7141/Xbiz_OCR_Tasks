import easyocr

reader = easyocr.Reader(['en'])
# Creates an OCR reader object. ['en'] tells EasyOCR to recognize English text.

result = reader.readtext("images/image_hdfc.jpg", detail=0)
# detail=0, result will be a list of strings.

print("OCR Result using EasyOCR:")
print("\n".join(result))