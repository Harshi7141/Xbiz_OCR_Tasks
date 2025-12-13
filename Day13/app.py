from flask import Flask, request, jsonify, render_template
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
import json

app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False

UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------- OCR Engines ----------
TESS_LANGS = "eng+hin"
easyocr_reader = easyocr.Reader(['en'], gpu=False)
paddleocr_reader = PaddleOCR(lang='en', use_textline_orientation=True)
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"



# ------ DESKEW + AUTO-ROTATE -----------

# def deskew_and_autorotate(img_bgr, save_prefix):

#     try:
#         cv2.imwrite(f"{save_prefix}_original.jpg", img_bgr)
#     except:
#         pass

#     gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
#     _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
#     coords = np.column_stack(np.where(thresh > 0))
    

#     if len(coords) > 20:
#         angle = cv2.minAreaRect(coords)[-1]
#         if angle < -45:
#             angle = -(90 + angle)
#         else:
#             angle = -angle

#         (h, w) = img_bgr.shape[:2]
#         M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
#         deskewed = cv2.warpAffine(
#             img_bgr, M, (w, h),
#             flags=cv2.INTER_CUBIC,
#             borderMode=cv2.BORDER_CONSTANT,
#             borderValue=(255, 255, 255)
#         )
#     else:
#         deskewed = img_bgr.copy()

#     pad = 80
#     deskewed = cv2.copyMakeBorder(
#         deskewed, pad, pad, pad, pad,
#         cv2.BORDER_CONSTANT, value=[255, 255, 255]
#     )

#     try:
#         cv2.imwrite(f"{save_prefix}_deskew.jpg", deskewed)
#     except:
#         pass

#     try:
#         osd = pytesseract.image_to_osd(deskewed)
#         rot = int(re.search(r"Rotate: (\d+)", osd).group(1))
#     except:
#         rot = 0

#     if rot == 90:
#         final = cv2.rotate(deskewed, cv2.ROTATE_90_CLOCKWISE)
#     elif rot == 180:
#         final = cv2.rotate(deskewed, cv2.ROTATE_180)
#     elif rot == 270:
#         final = cv2.rotate(deskewed, cv2.ROTATE_90_COUNTERCLOCKWISE)
#     else:
#         final = deskewed

#     try:
#         cv2.imwrite(f"{save_prefix}_final.jpg", final)
#     except:
#         pass

#     return final

# def deskew_and_autorotate(img_bgr, save_prefix):

#     try:
#         cv2.imwrite(f"{save_prefix}_original.jpg", img_bgr)
#     except:
#         pass

#     # ---------- DESKEW WITH TILT CHECK ----------
#     gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
#     _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
#     coords = np.column_stack(np.where(thresh > 0))

#     deskewed = img_bgr.copy()

#     if len(coords) > 20:

#         angle = cv2.minAreaRect(coords)[-1]

#         # Normalize angle
#         if angle < -45:
#             angle = -(90 + angle)
#         else:
#             angle = -angle

#         print(f"Detected angle = {angle}")

#         # 👉 NEW CONDITION: Deskew only if tilt > 2°
#         if abs(angle) > 2:
#             print("Tilt detected → Applying deskew")

#             (h, w) = img_bgr.shape[:2]
#             M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)

#             deskewed = cv2.warpAffine(
#                 img_bgr, M, (w, h),
#                 flags=cv2.INTER_CUBIC,
#                 borderMode=cv2.BORDER_CONSTANT,
#                 borderValue=(255, 255, 255)
#             )
#         else:
#             print("No tilt → Deskew skipped")
#             deskewed = img_bgr.copy()

#     else:
#         print("Not enough text pixels → Deskew skipped")
#         deskewed = img_bgr.copy()

#     # ---------- ADD BORDER ----------
#     # pad = 80
#     # deskewed = cv2.copyMakeBorder(
#     #     deskewed, pad, pad, pad, pad,
#     #     cv2.BORDER_CONSTANT, value=[255, 255, 255]
#     # )

#     try:
#         cv2.imwrite(f"{save_prefix}_deskew.jpg", deskewed)
#     except:
#         pass

#     # ---------- OCR ORIENTATION (OSD) ----------
#     try:
#         osd = pytesseract.image_to_osd(deskewed)
#         rot = int(re.search(r"Rotate: (\d+)", osd).group(1))
#     except:
#         rot = 0

#     print(f"Tesseract OSD rotation = {rot}")


#     # ---------- APPLY FINAL ROTATION ----------
#     if rot == 90:
#         final = cv2.rotate(deskewed, cv2.ROTATE_90_CLOCKWISE)
#     # elif rot == 180:
#     #     final = cv2.rotate(deskewed, cv2.ROTATE_180)
#     elif rot == 270:
#         final = cv2.rotate(deskewed, cv2.ROTATE_90_COUNTERCLOCKWISE)
#     else:
#         final = deskewed

#     try:
#         cv2.imwrite(f"{save_prefix}_final.jpg", final)
#     except:
#         pass

#     return final

# ------------------------------------------------------
# Helper: Save image safely
# ------------------------------------------------------
def safe_imwrite(path, img):
    try:
        cv2.imwrite(path, img)
    except:
        pass



def deskew_and_autorotate(img_bgr, save_prefix):
    safe_imwrite(f"{save_prefix}_original.jpg", img_bgr)

    # ---------------------------------------------------------------
    # STEP 1 — DESKEW (FIX TILT) USING HOUGH LINE TRANSFORM
    # ---------------------------------------------------------------
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray_blur, 50, 150, apertureSize=3)

    lines = cv2.HoughLines(edges, 1, np.pi / 180, 150)
    deskewed = img_bgr.copy()

    if lines is not None:
        angles = []

        for line in lines:
            rho, theta = line[0]
            angle = (theta * 180 / np.pi) - 90

            # Aadhaar has many horizontal visual lines → focus on near-horizontal
            if -45 < angle < 45:
                angles.append(angle)

        if len(angles) > 0:
            median_angle = np.median(angles)
            print("Detected tilt angle:", median_angle)

            if abs(median_angle) > 1:
                h, w = img_bgr.shape[:2]
                M = cv2.getRotationMatrix2D((w // 2, h // 2), median_angle, 1.0)
                deskewed = cv2.warpAffine(
                    img_bgr, M, (w, h),
                    flags=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(255, 255, 255)
                )
    else:
        print("No lines detected → skipping deskew")

    safe_imwrite(f"{save_prefix}_deskew.jpg", deskewed)

    # ---------------------------------------------------------------
    # STEP 2 — FIND UPRIGHT ROTATION (0°, 90°, 180°, 270°) USING OCR
    # ---------------------------------------------------------------
    def rotate(img, deg):
        if deg == 0: return img
        if deg == 90: return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        if deg == 180: return cv2.rotate(img, cv2.ROTATE_180)
        if deg == 270: return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    def ocr_score(img):
        try:
            small = cv2.resize(img, None, fx=0.6, fy=0.6)
            pil = Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
            txt = pytesseract.image_to_string(pil, config="--psm 6")
            # Count readable English/Hindi/Gujarati tokens length >= 3
            tokens = re.findall(r"[A-Za-z0-9]{3,}", txt)
            return len(tokens), txt
        except:
            return 0, ""

    angles = [0, 90, 180, 270]
    best_img = None
    best_score = -1
    best_text = ""

    for ang in angles:
        rimg = rotate(deskewed, ang)
        score, text = ocr_score(rimg)

        if score > best_score:
            best_score = score
            best_img = rimg
            best_text = text

    upright = best_img  # This is ALWAYS the correct orientation

    # ---------------------------------------------------------------
    # STEP 3 — PHOTO SHOULD BE ON LEFT (Aadhaar Standard Layout)
    # ---------------------------------------------------------------
    h, w = upright.shape[:2]
    left = upright[:, :w // 2]
    right = upright[:, w // 2:]

    # Aadhaar photo is darker → if right is darker, rotate 180°
    if np.mean(left) > np.mean(right):
        upright = cv2.rotate(upright, cv2.ROTATE_180)

    # ---------------------------------------------------------------
    # SAVE FINAL OUTPUT
    # ---------------------------------------------------------------
    safe_imwrite(f"{save_prefix}_final.jpg", upright)
    return upright


# ------------ IMPROVED PREPROCESSING ------------
def preprocess_for_tesseract(img_np):

    # 1) Upscale 2x for clarity
    img = cv2.resize(img_np, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

    # 2) Convert to gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3) Bilateral filter (keeps edges, removes noise)
    noise_free = cv2.bilateralFilter(gray, 9, 75, 75)

    # 4) CLAHE contrast boost
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    boosted = clahe.apply(noise_free)

    # 5) Sharpen image
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    sharp = cv2.filter2D(boosted, -1, kernel)

    # 6) Adaptive threshold for ID cards
    thresh = cv2.adaptiveThreshold(
        sharp, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 10
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



# ------------- MAIN OCR ----------------------
def run_ocr(filepath, engine_id, txn_folder):

    file_ext = filepath.lower()
    text = ""

    # ------- TESSERACT --------
    if engine_id == 1:
        print('Entered in tesseract Engine')

        config = "--oem 3 --psm 6"   # BEST FOR ID CARDS
        # config = r'--oem 3 --psm 4'
        # config= r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz:/.-'

        if file_ext.endswith('.pdf'):
            pages = pdf_to_images_fitz(filepath, zoom=4.0)
            for page in pages:
                img_np = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
                rotated = deskew_and_autorotate(img_np, save_prefix=os.path.join(txn_folder, "images/tess"))
                processed = preprocess_for_tesseract(rotated)
                text += pytesseract.image_to_string(processed, lang=TESS_LANGS, config=config) + " "
        else:
            img = Image.open(filepath).convert("RGB")
            img_np = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            rotated = deskew_and_autorotate(img_np, save_prefix=os.path.join(txn_folder, "images/tess"))
            processed = preprocess_for_tesseract(rotated)
            text = pytesseract.image_to_string(processed, lang=TESS_LANGS, config=config)

        return text.strip(), "Tesseract OCR success"

    # ------- EASY OCR --------
    elif engine_id == 2:
        if file_ext.endswith('.pdf'):
            pages = pdf_to_images_fitz(filepath)
            for page in pages:
                img_np = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
                rotated = deskew_and_autorotate(img_np, save_prefix=os.path.join(txn_folder, "easy"))
                resized = cv2.resize(rotated, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
                result = easyocr_reader.readtext(resized)
                text += " ".join([i[1] for i in result]) + " "
        else:
            img = Image.open(filepath).convert("RGB")
            img_np = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            rotated = deskew_and_autorotate(img_np, save_prefix=os.path.join(txn_folder, "easy"))
            resized = cv2.resize(rotated, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
            result = easyocr_reader.readtext(resized)
            text = " ".join([i[1] for i in result])

        return text.strip(), "EasyOCR success"

    # ------- PADDLE OCR --------
    elif engine_id == 3:
        if file_ext.endswith('.pdf'):
            pages = pdf_to_images_fitz(filepath)
            for page in pages:
                img_np = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
                rotated = deskew_and_autorotate(img_np, save_prefix=os.path.join(txn_folder, "paddle"))
                result = paddleocr_reader.ocr(rotated)
                if result and len(result) > 0:
                    ocr = result[0]
                    text += " ".join(ocr.get("rec_texts", [])) + " "
        else:
            img = Image.open(filepath).convert("RGB")
            img_np = np.array(img)
            rotated = deskew_and_autorotate(img_np, save_prefix=os.path.join(txn_folder, "paddle"))
            result = paddleocr_reader.ocr(rotated)
            if result and len(result) > 0:
                ocr = result[0]
                text = " ".join(ocr.get("rec_texts", []))

        return text.strip(), "PaddleOCR success"

    return "", "Invalid engine"


@app.route("/")
def home():
    return render_template("index1.html")

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
        if request.is_json and "input_image" in request.json:
            base64_str = request.json["input_image"]
            if "," in base64_str:
                base64_str = base64_str.split(",")[1]

            raw = base64.b64decode(base64_str)
            filepath = os.path.join(txn_folder, "images/input.jpg")

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
