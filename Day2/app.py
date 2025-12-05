
'''from flask import Flask, request, jsonify
import pytesseract
from PIL import Image
import os
import base64
from pdf2image import convert_from_path
import io

app = Flask(__name__)
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/upload', methods=['POST'])
def upload_file():
    response = {
        "input_image": "",
        "ocr_result": "",
        "txn_id": "",
        "documentType": "",
        "msg": "",
        "remark": ""
    }
    
    # Get txn_id and documentType from form data (optional)
    txn_id = request.form.get("txn_id", "")
    document_type = request.form.get("documentType", "")

    response["txn_id"] = txn_id
    response["documentType"] = document_type

    if 'file' not in request.files:
        response["msg"] = "No file part in the request"
        response["remark"] = "failed"
        return jsonify(response), 400

    file = request.files['file']

    if file.filename == '':
        response["msg"] = "No file selected"
        response["remark"] = "failed"
        return jsonify(response), 400

    try:
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        # Convert file to base64 for input_image field
        with open(filepath, "rb") as f:
            encoded_string = base64.b64encode(f.read()).decode()
            response["input_image"] = encoded_string

        # OCR processing
        ocr_text = ""
        if file.filename.lower().endswith('.pdf'):
            pages = convert_from_path(filepath)
            for page in pages:
                ocr_text += pytesseract.image_to_string(page)
        else:
            img = Image.open(filepath)
            ocr_text = pytesseract.image_to_string(img)

        response["ocr_result"] = ocr_text
        response["msg"] = "File processed successfully"
        response["remark"] = "success"

    except Exception as e:
        response["msg"] = str(e)
        response["remark"] = "failed"

    

    return jsonify(response)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
'''

from flask import Flask, request, jsonify
import pytesseract
from PIL import Image
from pdf2image import convert_from_path
import os
import base64
import easyocr
from paddleocr import PaddleOCR
import numpy as np
import uuid

app = Flask(__name__)
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize OCR engines
easyocr_reader = easyocr.Reader(['en', 'mr'], gpu=False)
paddleocr_reader = PaddleOCR(use_angle_cls=True, lang='en')  # 'en' for English, 'ch' for Chinese, etc.

@app.route('/upload', methods=['POST'])
def upload_file():
    # print("request data",request)
    txn_id = request.form.get("txn_id")
    # print("txn_id",txn_id)
    if not txn_id:
        txn_id = str(uuid.uuid4())

    if request.is_json:
        document_type = request.json.get("documentType", "")
    else:
        document_type = request.form.get("documentType", "")

    # Base response template
    response = {
        "tesseract_result": {"input_image": "", "ocr_result": "", "txn_id": txn_id, "documentType": document_type, "msg": "", "remark": ""},
        "easyocr_result": {"input_image": "", "ocr_result": "", "txn_id": txn_id, "documentType": document_type, "msg": "", "remark": ""},
        "paddleocr_result": {"input_image": "", "ocr_result": "", "txn_id": txn_id, "documentType": document_type, "msg": "", "remark": ""}
    }

    if 'file' not in request.files:
        for key in response:
            response[key]["msg"] = "No file part"
            response[key]["remark"] = "failed"
        return jsonify(response), 400
    
    file = request.files['file']

    if file.filename == '':
        for key in response:
            response[key]["msg"] = "No file selected"
            response[key]["remark"] = "failed"
        return jsonify(response), 400

    try:
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        # Convert file to base64 for input_image
        with open(filepath, "rb") as f:
            encoded_string = base64.b64encode(f.read()).decode()
            for key in response:
                response[key]["input_image"] = encoded_string

        # ---------- Tesseract OCR ----------
        tesseract_text = ""
        if file.filename.lower().endswith('.pdf'):
            pages = convert_from_path(filepath)
            for page in pages:
                tesseract_text += pytesseract.image_to_string(page)
        else:
            img = Image.open(filepath)
            tesseract_text = pytesseract.image_to_string(img)
        response["tesseract_result"]["ocr_result"] = tesseract_text
        response["tesseract_result"]["msg"] = "Tesseract OCR success"
        response["tesseract_result"]["remark"] = "success"

        # ---------- EasyOCR ----------
        easy_text = ""
        if file.filename.lower().endswith('.pdf'):
            pages = convert_from_path(filepath)
            for page in pages:
                result = easyocr_reader.readtext(np.array(page))
                easy_text += " ".join([text for (_, text, _) in result]) #ithe list of tuple aste so aapn join direct use karu shakto.
        else:
            img = Image.open(filepath)
            result = easyocr_reader.readtext(np.array(img))
            easy_text = " ".join([text for (_, text, _) in result])
        response["easyocr_result"]["ocr_result"] = easy_text
        response["easyocr_result"]["msg"] = "EasyOCR success"
        response["easyocr_result"]["remark"] = "success"

        # ---------- PaddleOCR ----------
        paddle_text = ""
        if file.filename.lower().endswith('.pdf'):
            pages = convert_from_path(filepath)
            for page in pages:
                result = paddleocr_reader.ocr(np.array(page))
                for line in result:
                    for _, text, _ in line:
                        paddle_text += text + " " #ithe list of list of tuple aste so join function use karna possible nahi.
        else:
            img = Image.open(filepath)
            result = paddleocr_reader.predict(np.array(img))
            for line in result:
                for _, text, _ in line:
                    paddle_text += text + " "
        response["paddleocr_result"]["ocr_result"] = paddle_text
        response["paddleocr_result"]["msg"] = "PaddleOCR success"
        response["paddleocr_result"]["remark"] = "success"

    except Exception as e:
        error_msg = str(e)
        for key in response:
            response[key]["msg"] = error_msg
            response[key]["remark"] = "failed"

    return jsonify(response)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
