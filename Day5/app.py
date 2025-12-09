

from flask import Flask, request, jsonify
import pytesseract
from PIL import Image, ImageSequence
from pdf2image import convert_from_path
import os
import base64
import easyocr
from paddleocr import PaddleOCR
import numpy as np
import uuid
import cv2
import random
import json

app = Flask(__name__)
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize OCR engines
TESS_LANGS = "eng+hin+mar+guj+tam+tel+kan+mal+ben+pan"
easyocr_reader = easyocr.Reader(['en', 'hi'], gpu=False)
# paddleocr_reader = PaddleOCR(use_angle_cls=True, lang='hi')
paddleocr_reader = PaddleOCR(lang='en', use_textline_orientation=True)

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def generate_txn_id():
    return "TXN-" + str(random.randint(1000, 9999))

def write_log(log_folder, message):
    log_path = os.path.join(log_folder, "logs.txt")
    with open(log_path, "a") as log_file:
        log_file.write(message + "\n")


#----- document type

# def detect_document_type(text):
#     text_lower = text.lower()
#     doctype = ""
#     # PAN CARD keywords
#     if (
#         "income tax" in text_lower or
#         "permanent account" in text_lower or
#         "govt. of india" in text_lower and "card" in text_lower or
#         "pan" in text_lower
#     ):
#         doctype = doctype + "PAN" + " "

#     # AADHAAR CARD keywords
#     elif (
#         "aadhaar" in text_lower or
#         "uidai" in text_lower or
#         "unique identification authority" in text_lower or
#         "आधार" in text_lower or
#         "ಆಧಾರ್" in text_lower or
#         "ஆதார்" in text_lower or
#         "ఆధార్" in text_lower
#     ):
#         doctype = doctype + "AADHAAR" + " "

#     elif (
#         "election commission" in text_lower or
#         "epic" in text_lower or
#         "voter id" in text_lower or
#         "elector photo identity" in text_lower or
#         "eci" in text_lower or
#         "मतदाता पहचान पत्र" in text_lower or
#         "मतदाता" in text_lower or
#         "निर्वाचन आयोग" in text_lower or
#         "मतदार ओळखपत्र" in text_lower or
#         "নির্বাচন কমিশন" in text_lower
#     ):
#         doctype = doctype + "VOTER ID" + " "
    
#     elif (
#         "driving licence" in text_lower or
#         "driving license" in text_lower or
#         "dl no" in text_lower or
#         "d.l. no" in text_lower or
#         "licence number" in text_lower or
#         "license number" in text_lower or
#         "transport department" in text_lower or
#         "rto" in text_lower or
#         "regional transport office" in text_lower or
#         "ड्राइविंग लाइसेंस" in text_lower or
#         "वाहन चालविण्याचा परवाना" in text_lower or
#         "परवाना क्रमांक" in text_lower or
#         "டிரைவிங் லைசன்ஸ்" in text_lower or
#         "ಡ್ರೈವಿಂಗ್ ಲೈಸೆನ್ಸ್" in text_lower
#     ):
#         doctype += "DRIVING_LICENSE"

#     else:
#         doctype = doctype + "OTHER" + " "
#     return doctype




def detect_document_type(text):
    text_lower = text.lower()
    detected = []

    # --- PAN ---
    if (
        "income tax" in text_lower or
        "permanent account" in text_lower or
        "pan" in text_lower or
        "govt. of india" in text_lower
    ):
        detected.append("PAN")

    # --- AADHAAR ---
    if (
        "aadhaar" in text_lower or
        "uidai" in text_lower or
        "unique identification authority" in text_lower or
        "आधार" in text_lower
    ):
        detected.append("AADHAAR")

    # --- DRIVING LICENCE ---
    if (
        "driving licence" in text_lower or
        "driving license" in text_lower or
        "transport department" in text_lower or
        "mh" in text_lower and "dl" in text_lower
    ):
        detected.append("DRIVING_LICENSE")

    # --- VOTER ID ---
    if (
        "election commission" in text_lower or
        "epic" in text_lower or
        "voter id" in text_lower or
        "eci" in text_lower
    ):
        detected.append("VOTER_ID")

    if not detected:
        return ["OTHER"]

    return detected

def run_ocr(filepath, engine_id):
    file_ext = filepath.lower()

    # ---- Tesseract ----
    if engine_id == 1:
        tesseract_text = ""
        if file_ext.endswith('.pdf'):
            pages = convert_from_path(filepath, dpi=300, poppler_path=r"C:\poppler-25.12.0\Library\bin")
            for page in pages:
                tesseract_text += pytesseract.image_to_string(page, lang=TESS_LANGS)
        
        elif file_ext.endswith('.tiff') or file_ext.endswith('.tif'):
            img = Image.open(filepath)
            for frame in ImageSequence.Iterator(img):
                tesseract_text += pytesseract.image_to_string(frame, lang=TESS_LANGS)

        else:
            img = Image.open(filepath)
            tesseract_text = pytesseract.image_to_string(img, lang=TESS_LANGS)

        return tesseract_text, "Tesseract OCR success"

    # ---- EasyOCR ----
    elif engine_id == 2:
        easy_text = ""
        if file_ext.endswith('.pdf'):
            pages = convert_from_path(filepath, poppler_path=r"C:\poppler-25.12.0\Library\bin")
            for page in pages:
                result = easyocr_reader.readtext(np.array(page))
                easy_text += " ".join([text for (_, text, _) in result])
        
        elif file_ext.endswith('.tiff') or file_ext.endswith('.tif'):
            img = Image.open(filepath)
            for frame in ImageSequence.Iterator(img):
                result = easyocr_reader.readtext(np.array(frame))
                easy_text += " ".join([text for (_, text, _) in result])

        else:
            img = Image.open(filepath)
            result = easyocr_reader.readtext(np.array(img))
            easy_text = " ".join([text for (_, text, _) in result])

        return easy_text.strip(), "EasyOCR success"

    # ---- PaddleOCR ----
    elif engine_id == 3:
        paddle_text = ""

        if file_ext.endswith('.pdf'):
            pages = convert_from_path(filepath, dpi=300, poppler_path=r"C:\poppler-25.12.0\Library\bin")
            for page in pages:
                page = page.convert("RGB")
                opencv_img = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
                result = paddleocr_reader.ocr(np.array(page))

                if not result:
                    continue

                lines = result[0] if isinstance(result[0], list) else result

                for item in lines:
                    if len(item) > 1:
                        paddle_text += item[1][0] + " "

            return paddle_text.strip(), "PaddleOCR success"

        elif file_ext.endswith('.tif') or file_ext.endswith('.tiff'):
            img = Image.open(filepath)
            for frame in ImageSequence.Iterator(img):

                frame = frame.convert("RGB")
                result = paddleocr_reader.ocr(np.array(frame)) #yaat mala list of dict bhetal.

                if not result:
                    continue

                # extract recognized texts
                rec_texts = result[0]['rec_texts']
                paddle_text += " ".join(rec_texts) + " "

            return paddle_text.strip(), "PaddleOCR success (TIFF)"

        else:
            img = Image.open(filepath).convert("RGB")
            result = paddleocr_reader.ocr(np.array(img))
            if result and isinstance(result, list) and len(result) > 0:
                ocr_dict = result[0]

                if "rec_texts" in ocr_dict:
                    paddle_text = " ".join(ocr_dict["rec_texts"])
                else:
                    paddle_text = ""

            else:
                paddle_text = ""

            return paddle_text.strip(), "PaddleOCR success"

    else:
        return "", "Invalid OCR Engine ID"

#------Routign----------------------

@app.route('/upload', methods=['POST'])
def upload_file():

    if request.is_json:
        data = request.json
    else:
        data = request.form

    txn_id = data.get("txn_id", generate_txn_id())
    ocr_engine = int(data.get("ocr_engine", 1))

    response = {
        "input_image": "",
        "ocr_result": "",
        "txn_id": txn_id,
        "documentType": "",
        "msg": "",
        "remark": ""
    }

    # ---------- Create folders ----------
    txn_folder = os.path.join(UPLOAD_FOLDER, txn_id)
    logs_folder = os.path.join(txn_folder, "logs")
    os.makedirs(txn_folder, exist_ok=True)
    os.makedirs(logs_folder, exist_ok=True)

    write_log(logs_folder, "Transaction Started")

    try:
        if request.is_json and "input_image" in request.json:
            base64_str = request.json["input_image"]

            # Convert base64 → bytes → image save
            img_bytes = base64.b64decode(base64_str)
            filepath = os.path.join(txn_folder, "input_image.png")

            with open(filepath, "wb") as f:
                f.write(img_bytes)

            response["input_image"] = base64_str
            write_log(logs_folder, "Base64 image received & saved")

        
        elif 'file' in request.files:
            file = request.files['file']
            filepath = os.path.join(txn_folder, file.filename)
            file.save(filepath)

            with open(filepath, "rb") as f:
                encoded_string = base64.b64encode(f.read()).decode()
                response["input_image"] = encoded_string

            write_log(logs_folder, f"File uploaded: {file.filename}")

        else:
            response["msg"] = "No file or base64 image provided"
            response["remark"] = "failed"
            return jsonify(response), 400

        # ----- Run OCR -----
        ocr_text, msg = run_ocr(filepath, ocr_engine)

        doc_type = detect_document_type(ocr_text)

        response["ocr_result"] = ocr_text
        response["documentType"] = doc_type
        response["msg"] = msg
        response["remark"] = "success"

        # Save response JSON
        response_json_path = os.path.join(txn_folder, "response.json")
        with open(response_json_path, "w") as f:
            json.dump(response, f, indent=4)

        write_log(logs_folder, "Response JSON saved")

    except Exception as e:
        response["msg"] = str(e)
        response["remark"] = "failed"

    return jsonify(response)



if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)  
