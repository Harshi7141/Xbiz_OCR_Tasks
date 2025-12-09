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

app = Flask(__name__)
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------- OCR Engines ----------
TESS_LANGS = "eng+hin+mar+guj+tam+tel+kan+mal+ben+pan"
easyocr_reader = easyocr.Reader(['en', 'hi'], gpu=False)
paddleocr_reader = PaddleOCR(lang='en', use_textline_orientation=True)

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def generate_txn_id():
    return "TXN-" + str(random.randint(1000, 9999))


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


# ---------- Card Cropping ----------
def extract_cards_from_page(page_image):
    opencv_img = cv2.cvtColor(np.array(page_image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    card_images = []
    h_img, w_img = gray.shape[:2]

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        if w < 0.25 * w_img or h < 0.20 * h_img:
            continue

        aspect_ratio = w / float(h)
        if 1.0 < aspect_ratio < 3.8:
            crop = opencv_img[y:y+h, x:x+w]
            card_images.append(crop)

    if not card_images:
        card_images.append(opencv_img)

    return card_images


# ---------- Detect Doc Type ----------
def detect_document_type(text):
    if not text:
        return ["UNKNOWN"]

    t = text.lower()
    doc_types = set()
    
    # ---- pan ------
    pan_regex = re.search(r"\b[A-Z]{5}\s*\d{4}\s*[A-Z]\b", text, flags=re.I)

    pan_card_keywords = [
        "permanent account number",
        "income tax department",
        "govt. of india",
        "signature",
        "father's name",
        "date of birth"
    ]

    pan_false_keywords = [
        "income tax return",
        "itr",
        "acknowledgement",
        "assessment year",
        "taxable income",
        "updated return"
    ]

    if pan_regex:
        if not any(kw in t for kw in pan_false_keywords):
            if any(kw in t for kw in pan_card_keywords):
                doc_types.add("PAN")

    #----- aadhaar ------

    aadhaar_num = re.search(r"\b\d{4}\s*\d{4}\s*\d{4}\b", text)

    aadhaar_card_keywords = [
        "unique identification authority of india",
        "uidai",
        "government of india",
        "govt. of india",
        "aadhaar",
        "eaadhaar",
        "address:",
        "year of birth",
        "yob",
        "dob",
        "male",
        "female",
        "qr code",
        "proof of identity"
    ]

    aadhaar_false_keywords = [
        "itr",
        "income tax return",
        "acknowledgement",
        "assessment year",
        "bank",
        "branch",
        "ifsc",
        "statement",
        "kyc",
        "form",
        "voter id",
        "pan",
        "driving licence",
        "transport",
        "cov",
        "mcwg"
    ]

    if aadhaar_num:
        if not any(kw in t for kw in aadhaar_false_keywords):
            if any(kw in t for kw in aadhaar_card_keywords):
                doc_types.add("AADHAAR")


    #---- voter id------
    if ("voter id" in t or "epic" in t or "election commission" in t):
        doc_types.add("VOTER_ID")

    # --- Driving License-----
    dl_keywords = [
    "driving", "drivng", "drivng licence", "licence", "lcence", 
    "dl", "d l", "dln", "dlno", "dl no", 
    "maharashtra", "state motor", 
    "authorisationtodrive", "authorisation", 
    "form7", "form 7",
    "mcwg", "mcwg12", "cov", "class of vehicles"
    ]

    if any(kw in t for kw in dl_keywords):
        doc_types.add("DRIVING_LICENSE")

    # ---- bank-----
    bank_keywords = [
        "bank", "branch", "ifsc", "micr", "cheque", "passbook", "statement",
        "account", "a/c", "neft", "rtgs", "imps", "savings", "current account",
        "cancelled cheque"
    ]

    ifsc_regex = re.search(r"\b[A-Z]{4}0[A-Z0-9]{6}\b", text)
    micr_regex = re.search(r"\b\d{9}\b", text)
    account_number = re.search(r"\b\d{9,18}\b", text)

    is_bank_doc = (
        any(kw in t for kw in bank_keywords)
        or ifsc_regex
        or micr_regex
        or account_number
    )

    if is_bank_doc:
        doc_types.add("BANK DOCUMENT")

    if not doc_types:
        return ["OTHER"]
    
    return list(doc_types)
    


# ---------- Main OCR ----------
def run_ocr(filepath, engine_id):
    file_ext = filepath.lower()
    text = ""
    # --- Tesseract ---
    if engine_id == 1:
        def preprocess_variants(pil_img):
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            den = cv2.fastNlMeansDenoising(gray, h=10)
            thr = cv2.adaptiveThreshold(den, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2) #samjla nahi
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            sharp = cv2.filter2D(den, -1, kernel)

            return [
                pil_img,
                Image.fromarray(thr), #threshold array or image
                Image.fromarray(sharp) # sharped arary or image
            ]
        
        

        if file_ext.endswith('.pdf'):
            pages = pdf_to_images_fitz(filepath, zoom=4.0)
            for page in pages:
                variants = preprocess_variants(page)
                for v in variants:
                    text += pytesseract.image_to_string(v, lang=TESS_LANGS, config=r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz:/.-') + " "
        else:
            img = Image.open(filepath).convert("RGB")
            img_np = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            img_np = cv2.resize(img_np, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)

            kernel = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]])
            img_np = cv2.filter2D(img_np, -1, kernel)

            img_np = cv2.convertScaleAbs(img_np, alpha=1.4, beta=20)

            # Convert to PIL
            up_img = Image.fromarray(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))

            variants = preprocess_variants(up_img)
            for v in variants:
                text += pytesseract.image_to_string(v, lang=TESS_LANGS, config=r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz:/.-') + " "
        return text.strip(), "Tesseract OCR success"

    # --- EasyOCR ---
    elif engine_id == 2:
        if file_ext.endswith('.pdf'):
            pages = pdf_to_images_fitz(filepath)
            for page in pages:
                result = easyocr_reader.readtext(np.array(page))
                text += " ".join([i[1] for i in result])
        else:
            img = Image.open(filepath)
            result = easyocr_reader.readtext(np.array(img))
            text = " ".join([i[1] for i in result])

        return text.strip(), "EasyOCR success"

    # --- PaddleOCR ---
    elif engine_id == 3:
        if file_ext.endswith('.pdf'):
            pages = pdf_to_images_fitz(filepath)
            for pg in pages:
                card_imgs = extract_cards_from_page(pg)

                for card in card_imgs:
                    img_np = np.array(Image.fromarray(card[:, :, ::-1]))
                    result = paddleocr_reader.ocr(img_np)

                    if not result:
                        print("NO RESULT FROM PADDLE OCR")
                        continue
                    # result = [[ [box], [text, score] ], ... ]
                    for block in result:
                        if "rec_texts" in block:
                            for txt in block["rec_texts"]:
                                text += txt + " "

        else:
            img = Image.open(filepath).convert("RGB")
            result = paddleocr_reader.ocr(np.array(img))
            
            if result:
                for block in result:
                    if isinstance(block, dict) and "rec_texts" in block:
                        for txt in block["rec_texts"]:
                            text += txt + " "
                    else:
                        try:
                            text += block[1][0] + " "
                        except:
                            pass
        return text.strip(), "PaddleOCR success"

    return "", "Invalid engine"


# ---------- Upload Route ----------
@app.route('/upload', methods=['POST'])
def upload_file():
    data = request.json if request.is_json else request.form

    txn_id = data.get("txn_id", generate_txn_id())
    ocr_value = data.get("ocr_engine", "1")

    if ocr_value and str(ocr_value).isdigit():
        ocr_engine = int(ocr_value)
    else:
        ocr_engine = 1

    response = {
        "input_image": "",
        "ocr_result": "",
        "txn_id": txn_id,
        "documentType": "",
        "msg": "",
        "remark": ""
    }

    txn_folder = os.path.join(UPLOAD_FOLDER, txn_id)
    os.makedirs(txn_folder, exist_ok=True)

    try:
        # JSON base64 upload
        if request.is_json and "input_image" in request.json:
            base64_str = request.json["input_image"]

            if "," in base64_str:
                base64_str = base64_str.split(",")[1]

            raw = base64.b64decode(base64_str)

            if raw.startswith(b"%PDF"):
                ext = ".pdf"

            elif raw.startswith(b"II*\x00") or raw.startswith(b"MM\x00*"):
                ext = ".tif"     # TIFF image

            elif raw.startswith(b"\x89PNG\r\n\x1a\n"):
                ext = ".png"     # PNG image

            elif raw[0:2] == b"\xff\xd8":
                ext = ".jpg"     # JPEG image

            else:
                ext = ".bin"    
            filepath = os.path.join(txn_folder, "input" + ext)

            with open(filepath, "wb") as f:
                f.write(raw)

            response["input_image"] = base64_str

        # File upload
        elif "file" in request.files:
            file = request.files["file"]
            filepath = os.path.join(txn_folder, file.filename)
            file.save(filepath)

            with open(filepath, "rb") as f:
                response["input_image"] = base64.b64encode(f.read()).decode()

        else:
            return jsonify({"msg": "No file or base64 provided", "remark": "failed"}), 400

        # Run OCR
        ocr_text, msg = run_ocr(filepath, ocr_engine)
        response["ocr_result"] = ocr_text
        if ocr_text:
            response["documentType"] = detect_document_type(ocr_text)
        else:
            response["documentType"] = "UNKNOWN"
        response["msg"] = msg
        response["remark"] = "success"

    except Exception as e:
        print(traceback.format_exc())   
        response["msg"] = f"Error: {str(e)}"
        response["remark"] = "failed"

    return jsonify(response)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
