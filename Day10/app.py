from flask import Flask, request, jsonify
import pytesseract
from PIL import Image
import fitz
import os
import base64
import easyocr
from paddleocr import PaddleOCR
import numpy as np
import cv2
import random
import traceback
import re
from skimage.transform import radon
from scipy.ndimage import rotate as scipy_rotate
import json

app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False

UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------- OCR Engines ----------
TESS_LANGS = "eng"
easyocr_reader = easyocr.Reader(['en', 'hi'], gpu=False)
paddleocr_reader = PaddleOCR(lang='en', use_textline_orientation=True)
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"


# ---------- radon transform ---------

def correct_tilt_and_rotation(img_bgr, save_prefix):
    try:
        cv2.imwrite(f"{save_prefix}_original.jpg", img_bgr)
    except:
        pass


    try:
        # Small image → faster, stable
        small = cv2.resize(img_bgr, (0,0), fx=0.4, fy=0.4)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        
        # Edge image makes radon more effective
        edges = cv2.Canny(gray, 50, 150)

        # Radon angles to test
        theta = np.linspace(-20, 20, 200)

        sinogram = radon(edges, theta=theta, circle=False)

        # Highest response = tilt angle
        tilt_angle = theta[np.argmax(np.sum(sinogram, axis=0))]

        # Limit for safety
        if abs(tilt_angle) > 15:
            tilt_angle = 0

    except Exception as e:
        tilt_angle = 0


    try:
        rotated_tilt = scipy_rotate(img_bgr, -tilt_angle, reshape=False, order=1, mode='nearest')
    except:
        rotated_tilt = img_bgr.copy()

    try:
        cv2.imwrite(f"{save_prefix}_deskewed_radon.jpg", rotated_tilt)
    except:
        pass

   
    rotations = {
        "0": rotated_tilt,
        "90": cv2.rotate(rotated_tilt, cv2.ROTATE_90_CLOCKWISE),
        "180": cv2.rotate(rotated_tilt, cv2.ROTATE_180),
        "270": cv2.rotate(rotated_tilt, cv2.ROTATE_90_COUNTERCLOCKWISE)
    }

    best_img = rotated_tilt
    best_score = -1
    best_angle = "0"

    for angle, test_img in rotations.items():
        try:
            text = pytesseract.image_to_string(
                test_img,
                lang="eng",
                config="--oem 3 --psm 6"
            )
            score = sum(c.isalnum() for c in text)
        except:
            score = 0

        if score > best_score:
            best_score = score
            best_angle = angle
            best_img = test_img.copy()

        try:
            cv2.imwrite(f"{save_prefix}_rotation_test_{angle}.jpg", test_img)
        except:
            pass

    try:
        cv2.imwrite(f"{save_prefix}_final_best_{best_angle}.jpg", best_img)
    except:
        pass

    return best_img



# ------------ BEST preprocess for Tesseract ----------------------
def preprocess_for_tesseract(img_np):
    img = cv2.resize(img_np, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    _, thresh = cv2.threshold(
        enhanced, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return thresh


# ---------- PDF to Images ----------
def pdf_to_images_fitz(pdf_path, zoom=4.0):
    pages_out = []
    pdf = fitz.open(pdf_path)

    for page in pdf:
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        pages_out.append(img)

    return pages_out


# ---------- Document Detection ----------
def score(text, keywords):
    return sum(1 for k in keywords if k in text)


def detect_document_type(text):
    if not text:
        return ["UNKNOWN"]
    
    t = text.lower()
    detected = []

    # ---------- PAN ----------
    pan_score = 0

    pan_regex = re.search(r"\b[A-Z]{5}\d{4}[A-Z]\b", text, re.I)
    if pan_regex:
        pan_score += 5

    pan_keywords_front = [
        "income tax department",
        "government of india",
        "govt of india",
        "permanent account number",
        "signature",
        "father's name",
        "date of birth",
        "dob",
        "pan card"
    ]

    pan_keywords_back = [
        "income tax department",
        "department",
        "यह कार्ड भारत सरकार की संपत्ति है",
        "this card is the property of",
        "verify", "verification",
        "barcode", "qr", "qr code"
    ]

    pan_score += score(t, pan_keywords_front)
    pan_score += score(t, pan_keywords_back)

    if pan_score >= 3:
        detected.append("PAN")

    # ---------- AADHAAR CARD ----------
    aadhaar_score = 0

    aadhaar_num = re.search(r"\b\d{4}\s*\d{4}\s*\d{4}\b", text)
    masked = re.search(r"x{4}\s*x{4}\s*\d{4}", text, re.I)

    if aadhaar_num or masked:
        aadhaar_score += 5

    aadhaar_keywords_front = [
        "aadhaar", "aadhar", "uidai",
        "unique identification authority",
        "govt of india", "government of india",
        "year of birth", "yob", "dob",
        "qr", "qr code",
        "help@uidai.gov.in",
        "www.uidai.gov.in",
        "आधार", "आधार कार्ड",
    ]

    aadhaar_keywords_back = [
        "address",
        "s/o", "w/o", "c/o", "care of",
        "pincode", "pin code",
        "mobile update", "address update",
        "enrolment", "enrollment",
        "validity", "note", "instructions",
    ]

    aadhaar_score += score(t, aadhaar_keywords_front)
    aadhaar_score += score(t, aadhaar_keywords_back)

    if aadhaar_score >= 3:
        detected.append("AADHAAR")

    # ---------- VOTER ID ----------
    voter_score = 0

    voter_keywords_front = [
        "election commission of india",
        "epic",
        "voter id",
        "elector photo identity card",
        "identity card",
        "भारत निर्वाचन आयोग",
        "निर्वाचन आयोग",
        "voter",
        "epic no",
        "photo",
    ]

    voter_keywords_back = [
        "voter helpline",
        "ceo",
        "nvsp",
        "www.nvsp.in",
        "elector details",
        "polling station",
        "assembly constituency",
        "parliamentary constituency",
        "serial no",
        "age",
        "address",
    ]

    voter_score += score(t, voter_keywords_front)
    voter_score += score(t, voter_keywords_back)

    if voter_score >= 3:
        detected.append("VOTER_ID")

    # ---------- DRIVING LICENSE ----------
    dl_score = 0

    dl_num = re.search(r"\b[A-Z]{2}\d{2}\/?\d{4,7}\b", text, re.I)
    if dl_num:
        dl_score += 5

    dl_keywords_front = [
        "driving licence",
        "driving license",
        "licence",
        "license",
        "dl no",
        "dlno",
        "dln",
        "transport",
        "non transport",
        "permanent licence",
        "cov",
        "class of vehicles",
        "authorisation",
        "authorisation to drive",
        "state motor",
        "rto",
        "moto",
    ]

    dl_keywords_back = [
        "validity",
        "badge",
        "hazardous",
        "issue date",
        "expiry date",
        "blood group",
        "emergency contact",
        "test date",
        "badge no",
        "batch",
        "address",
    ]

    dl_score += score(t, dl_keywords_front)
    dl_score += score(t, dl_keywords_back)

    if dl_score >= 3:
        detected.append("DRIVING_LICENSE")

    # ---------- BANK DOCUMENT ----------
    bank_score = 0

    ifsc_regex = re.search(r"\b[A-Z]{4}0[A-Z0-9]{6}\b", text)
    acc_regex = re.search(r"\b\d{9,18}\b", text)

    if ifsc_regex:
        bank_score += 5
    if acc_regex:
        bank_score += 5

    bank_keywords_front = [
        "bank", "bank of", "branch",
        "account", "account number", "a/c", "acc", "acc no", "acct",
        "savings account", "current account",
        "ifsc", "ifsc code",
        "micr", "micr code",
        "cheque", "cancelled cheque",
        "passbook", "pass book",
        "statement", "bank statement",
        "neft", "rtgs", "imps",
        "upi", "virtual payment address",
        "customer id", "cust id",
        "cif", "cif no", "cif number",
        "txn", "transaction", "transactions",
        "deposit", "withdrawal", "balance",
        "mobile banking", "internet banking",
        "netbanking", "online banking",
    ]

    bank_keywords_back = [
        "address",
        "customer care",
        "helpline",
        "phone banking",
        "mobile banking",
        "internet banking",
    ]

    bank_score += score(t, bank_keywords_front)
    bank_score += score(t, bank_keywords_back)

    if bank_score >= 5:
        detected.append("BANK_DOCUMENT")

    return detected if detected else ["OTHER"]


# ---------- Main OCR ----------
def run_ocr(filepath, engine_id, txn_folder):
    file_ext = filepath.lower()
    text = ""

    if engine_id == 1:  # Tesseract
        if file_ext.endswith('.pdf'):
            pages = pdf_to_images_fitz(filepath, zoom=4.0)
            for page in pages:
                img_np = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
                rotated = correct_tilt_and_rotation(
                    img_np,
                    save_prefix=os.path.join(txn_folder, "images/tess")
                )
                processed = preprocess_for_tesseract(rotated)
                text += pytesseract.image_to_string(
                    processed,
                    lang=TESS_LANGS,
                    config="--oem 3 --psm 6"
                ) + " "
        else:
            img = Image.open(filepath).convert("RGB")
            img_np = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            rotated = correct_tilt_and_rotation(
                img_np,
                save_prefix=os.path.join(txn_folder, "images/tess")
            )
            processed = preprocess_for_tesseract(rotated)
            text = pytesseract.image_to_string(
                processed,
                lang=TESS_LANGS,
                config="--oem 3 --psm 6"
            )

        return text.strip(), "Tesseract OCR success"

    elif engine_id == 2:  # EasyOCR

        if file_ext.endswith('.pdf'):
            pages = pdf_to_images_fitz(filepath)
            for page in pages:
                img_np = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)

                rotated = correct_tilt_and_rotation(
                    img_np,
                    save_prefix=os.path.join(txn_folder, "images/easy")
                )

                rotated = cv2.resize(
                    rotated,
                    None,
                    fx=1.5, fy=1.5,
                    interpolation=cv2.INTER_CUBIC
                )

                result = easyocr_reader.readtext(rotated)
                text += " ".join([i[1] for i in result]) + " "
        else:
            img = Image.open(filepath).convert("RGB")
            img_np = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

            rotated = correct_tilt_and_rotation(
                img_np,
                save_prefix=os.path.join(txn_folder, "images/easy")
            )

            rotated = cv2.resize(
                rotated,
                None,
                fx=1.5, fy=1.5,
                interpolation=cv2.INTER_CUBIC
            )

            result = easyocr_reader.readtext(rotated)
            text = " ".join([i[1] for i in result])

        return text.strip(), "EasyOCR success"

    elif engine_id == 3:  # PaddleOCR
        print("Enter in Paddle Block")
        text = ""

        if file_ext.endswith('.pdf'):
            pages = pdf_to_images_fitz(filepath)
            for page in pages:
                img_np = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)

                rotated = correct_tilt_and_rotation(
                    img_np,
                    save_prefix=os.path.join(txn_folder, "images/paddle")
                )

                result = paddleocr_reader.ocr(rotated)
                if result and isinstance(result, list):
                    # Paddle returns list of [ [[box],[text,conf]], ... ]
                    for line in result:
                        try:
                            txt = line[1][0]
                            text += txt + " "
                        except:
                            pass
        else:
            img = Image.open(filepath).convert("RGB")
            img_np = np.array(img)

            rotated = correct_tilt_and_rotation(
                img_np,
                save_prefix=os.path.join(txn_folder, "images/paddle")
            )

            result = paddleocr_reader.ocr(rotated)
            if result and isinstance(result, list):
                for line in result:
                    try:
                        txt = line[1][0]
                        text += txt + " "
                    except:
                        pass

        return text.strip(), "PaddleOCR success"

    return "", "Invalid engine"


# ---------- Upload Route ----------
@app.route('/upload', methods=['POST'])
def upload_file():

    data = request.json if request.is_json else request.form
    txn_id = data.get("txn_id", "TXN-" + str(random.randint(1000, 9999)))
    ocr_engine = int(data.get("ocr_engine", 1))

    txn_folder = os.path.join(UPLOAD_FOLDER, txn_id)
    os.makedirs(txn_folder, exist_ok=True)
    os.makedirs(os.path.join(txn_folder, "images"), exist_ok=True)

    response = {
        "input_image": "",
        "ocr_result": "",
        "txn_id": txn_id,
        "documentType": "",
        "msg": "",
        "remark": ""
    }

    try:
        # BASE64 INPUT
        if request.is_json and "input_image" in request.json:
            base64_str = request.json["input_image"]
            if "," in base64_str:
                base64_str = base64_str.split(",")[1]

            raw = base64.b64decode(base64_str)

            ext = ".jpg"
            filepath = os.path.join(txn_folder, "images/input" + ext)

            with open(filepath, "wb") as f:
                f.write(raw)

            response["input_image"] = base64_str

        elif "file" in request.files:
            file = request.files["file"]
            ext = os.path.splitext(file.filename)[1]
            filepath = os.path.join(txn_folder, "images/input" + ext)
            file.save(filepath)

            with open(filepath, "rb") as f:
                response["input_image"] = base64.b64encode(f.read()).decode()

        else:
            return jsonify({"msg": "No image provided", "remark": "failed"}), 400

        # RUN OCR
        ocr_text, msg = run_ocr(filepath, ocr_engine, txn_folder)
        response["ocr_result"] = ocr_text
        response["msg"] = msg
        response["documentType"] = detect_document_type(ocr_text)
        response["remark"] = "success"

        # ---------- SAVE LOGS ----------
        try:
            with open(os.path.join(txn_folder, "request.json"), "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)

            with open(os.path.join(txn_folder, "response.json"), "w", encoding="utf-8") as f:
                json.dump(response, f, ensure_ascii=False, indent=4)

            with open(os.path.join(txn_folder, "ocr_text.txt"), "w", encoding="utf-8") as f:
                f.write(ocr_text)

            with open(os.path.join(txn_folder, "logs.txt"), "a", encoding="utf-8") as f:
                f.write("\n--------- NEW REQUEST ---------\n")
                f.write(f"TXN: {txn_id}\n")
                f.write(f"OCR Engine: {ocr_engine}\n")
                f.write(f"OCR MSG: {msg}\n")
                f.write(f"Detected: {response['documentType']}\n")
                f.write("Raw OCR Output:\n")
                f.write(ocr_text + "\n")

        except Exception as log_err:
            print("LOG SAVE ERROR:", log_err)

    except Exception as e:
        print(traceback.format_exc())
        response["remark"] = "failed"
        response["msg"] = str(e)

    return jsonify(response)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
