from paddleocr import PaddleOCR
# importing PaddleOCR class from paddleocr library.

ocr = PaddleOCR(lang='en')
# Here we are creating the instance of PaddleOCR

image_path = "images/image_hdfc.jpg"

result = ocr.predict(image_path)
# This runs OCR on image. predict() is the newer method instead of ocr(). It returns the result in a list of dictionaries, where each dictionary corresponds to a page or image processed.


print("\nOCR Result using PaddleOCR:\n")

for res in result:
    rec_texts = res['rec_texts']
    for text in rec_texts:
        if text: 
            print(text)