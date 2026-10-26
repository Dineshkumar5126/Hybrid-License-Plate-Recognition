import re
import cv2
import numpy as np

def load_image(image_path):
    """Loads an image from the specified path."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image at {image_path}. Please check the path.")
    return img

def resize_to_standard_width(img, target_width=800):
    """
    Standardizes input vehicle image width to target_width while preserving aspect ratio:
    - If image width is already target_width, returns original image unchanged.
    - If downscaling (w > target_width), uses cv2.INTER_AREA to prevent aliasing.
    - If upscaling (w < target_width), uses cv2.INTER_CUBIC to maintain edge sharpness.
    """
    h, w = img.shape[:2]
    if w == target_width:
        return img.copy()

    aspect_ratio = h / float(w)
    target_height = int(round(target_width * aspect_ratio))
    interp = cv2.INTER_AREA if w > target_width else cv2.INTER_CUBIC
    return cv2.resize(img, (target_width, target_height), interpolation=interp)

def convert_to_grayscale(img):

    """Converts a color image to grayscale."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def apply_noise_reduction(gray_img):
    """Applies bilateral filtering to reduce noise while keeping edges sharp."""
    # Parameters: d (diameter of pixel neighborhood), sigmaColor, sigmaSpace
    return cv2.bilateralFilter(gray_img, 11, 17, 17)

def detect_edges(blurred_img):
    """Detects edges using the Canny edge detector."""
    # 30 and 200 are the lower and upper thresholds for the Canny algorithm
    return cv2.Canny(blurred_img, 30, 200)

def find_plate_contours(edged_img, original_img):
    """
    Finds license plate candidate contours using classical DIP techniques:
    1. Grayscale & edge-preserving bilateral filtering
    2. Sobel horizontal gradient (Sobel-X) to emphasize vertical character strokes
    3. Otsu automatic binarization
    4. Morphological closing with a horizontal rectangular kernel to connect character strokes
    5. Contour extraction and geometric / spatial candidate scoring
    """
    img_h, img_w = original_img.shape[:2]

    # Convert to grayscale if not already
    if len(original_img.shape) == 3:
        gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = original_img.copy()

    # Step 1: Smooth image while preserving edges
    blurred = cv2.bilateralFilter(gray_img, 11, 17, 17)

    # Step 2: Sobel gradient in X direction (horizontal gradient).
    # License plate characters are vertical strokes with high contrast.
    # Sobel-X highlights these vertical transitions while ignoring horizontal bumper lines.
    sobelx = cv2.Sobel(blurred, cv2.CV_16S, 1, 0, ksize=3)
    abs_sobelx = cv2.convertScaleAbs(sobelx)

    # Step 3: Threshold gradient image using Otsu's binarization
    _, thresh = cv2.threshold(abs_sobelx, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Step 4: Morphological Closing with a horizontal rectangular structuring element.
    # Because license plate characters are aligned side-by-side horizontally,
    # a wide rectangular kernel connects adjacent characters into a solid plate region.
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, rect_kernel)

    # Step 5: Morphological Opening with a small kernel to remove isolated background noise
    clean = cv2.morphologyEx(closed, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

    # Step 6: Find contours of the connected candidate regions
    contours, _ = cv2.findContours(clean.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        rect_area = w * h
        aspect_ratio = float(w) / max(h, 1)

        # 1. Aspect Ratio: Standard license plates have width roughly 1.8x to 6.5x of height
        if not (1.8 <= aspect_ratio <= 6.5):
            continue

        # 2. Area: Must be large enough to contain readable text, but not take up entire vehicle
        if not (1000 <= rect_area <= (img_w * img_h * 0.30)):
            continue

        # 3. Size constraints relative to image dimensions:
        # Typical plate occupies 10% to 65% width and 2.5% to 22% height
        if not (0.10 * img_w <= w <= 0.65 * img_w):
            continue
        if not (0.025 * img_h <= h <= 0.22 * img_h):
            continue

        # 4. Spatial Location: Plates are mounted on vehicle bumpers in the lower-middle region
        if not (0.35 * img_h <= y <= 0.90 * img_h):
            continue

        # 5. Edge Density Check: A real license plate contains high internal edge density from characters
        roi_edged = edged_img[y:y+h, x:x+w]
        edge_density = cv2.countNonZero(roi_edged) / float(rect_area)
        if not (0.05 <= edge_density <= 0.65):
            continue

        # 6. Candidate Scoring:
        # Penalize distance from horizontal center, reward edge density and typical aspect ratio (~4.5)
        cx = x + (w / 2.0)
        dx = abs(cx - (img_w / 2.0))
        ar_score = -abs(aspect_ratio - 4.5) * 50
        score = (edge_density * 500) - (dx * 1.5) + ar_score + (rect_area * 0.01)

        box_contour = np.array([[[x, y]], [[x, y+h]], [[x+w, y+h]], [[x+w, y]]], dtype=np.int32)
        candidates.append((score, box_contour))

    # Return contour with the highest score
    if candidates:
        candidates.sort(key=lambda c: c[0], reverse=True)
        return candidates[0][1]

    return None

def crop_plate(original_img, gray_img, plate_contour):
    """Crops the image to extract just the license plate region with small boundary padding."""
    if plate_contour is None:
        return None

    # Create a blank binary mask
    mask = np.zeros(gray_img.shape, np.uint8)

    # Draw and fill the candidate contour on mask
    cv2.drawContours(mask, [plate_contour], 0, 255, -1)

    # Find the bounding coordinates of the mask
    (rows, cols) = np.where(mask == 255)
    if len(rows) == 0 or len(cols) == 0:
        return None

    min_row, min_col = np.min(rows), np.min(cols)
    max_row, max_col = np.max(rows), np.max(cols)

    # Apply slight padding (4% vertical, 2% horizontal) to preserve edge character strokes
    h_box = max_row - min_row + 1
    w_box = max_col - min_col + 1
    pad_y = int(0.04 * h_box)
    pad_x = int(0.02 * w_box)

    r1 = max(0, min_row - pad_y)
    r2 = min(gray_img.shape[0], max_row + 1 + pad_y)
    c1 = max(0, min_col - pad_x)
    c2 = min(gray_img.shape[1], max_col + 1 + pad_x)

    # Crop the grayscale image using padded coordinates
    cropped_plate = gray_img[r1:r2, c1:c2]

    return cropped_plate

def preprocess_for_ocr(cropped_plate):
    """
    Prepares the cropped license plate image for OCR:
    1. Grayscale conversion.
    2. Upscaling (2.0x) with cubic interpolation for improved character resolution.
    3. Contrast enhancement using CLAHE.
    4. Gaussian smoothing to reduce pixel noise.
    5. Otsu thresholding with polarity normalization (dark characters on light background).
    """
    if cropped_plate is None:
        return None

    if len(cropped_plate.shape) == 3:
        gray = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2GRAY)
    else:
        gray = cropped_plate.copy()

    # 1. Upscale by 2.0x
    resized = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

    # 2. Contrast enhancement via CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(resized)

    # 3. Gaussian blur
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)

    # 4. Otsu binarization
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 5. Polarity normalization: ensure background is light and characters are dark
    if thresh.mean() < 127:
        thresh = cv2.bitwise_not(thresh)

    return thresh

def preprocess_for_ocr_adaptive(cropped_plate):
    """
    Fallback DIP preprocessing using Adaptive Gaussian Thresholding for non-uniform lighting/shadows.
    """
    if cropped_plate is None:
        return None

    if len(cropped_plate.shape) == 3:
        gray = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2GRAY)
    else:
        gray = cropped_plate.copy()

    resized = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(resized)
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    adapt = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 19, 9)

    if adapt.mean() < 127:
        adapt = cv2.bitwise_not(adapt)

    return adapt

def recognize_plate_text(plate_img, reader=None):
    """
    Performs Optical Character Recognition (OCR) on the license plate using EasyOCR.
    Returns:
        raw_text (str): Exact detected text segments from EasyOCR.
        cleaned_text (str): Filtered alphanumeric string without symbols/noise.
        confidence (float): Average confidence score of detected characters.
    """
    import easyocr

    if reader is None:
        # Use English model on CPU without verbose download messages
        reader = easyocr.Reader(['en'], gpu=False, verbose=False)

    # Run EasyOCR with an alphanumeric allowlist
    results = reader.readtext(plate_img, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')

    if not results:
        # Fallback without allowlist if nothing was detected
        results = reader.readtext(plate_img)

    if not results:
        return "", "", 0.0

    raw_segments = []
    confs = []
    for bbox, text, conf in results:
        raw_segments.append(text)
        confs.append(float(conf))

    raw_text = " ".join(raw_segments)
    avg_conf = float(sum(confs) / len(confs)) if confs else 0.0

    # Clean text to alphanumeric characters
    cleaned_text = "".join([c for c in raw_text if c.isalnum()]).upper()

    return raw_text, cleaned_text, avg_conf

# Standard Indian State and Union Territory codes
INDIAN_STATE_CODES = {
    'AN', 'AP', 'AR', 'AS', 'BR', 'CG', 'CH', 'DD', 'DL', 'DN',
    'GA', 'GJ', 'HP', 'HR', 'JH', 'JK', 'KA', 'KL', 'LA', 'LD',
    'MH', 'ML', 'MN', 'MP', 'MZ', 'NL', 'OD', 'PB', 'PY', 'RJ',
    'SK', 'TN', 'TR', 'TS', 'UK', 'UP', 'US', 'WB', 'BH'
}

# Position-aware character ambiguity translation tables
DIGIT_FROM_LETTER = {
    'O': '0', 'Q': '0', 'D': '0', 'I': '1', 'L': '1', 'Z': '2', 'S': '5', 'B': '8', 'G': '6', 'T': '7'
}
LETTER_FROM_DIGIT = {
    '0': 'O', '1': 'I', '2': 'Z', '5': 'S', '8': 'B', '6': 'G'
}

# Common OCR state code single-stroke confusions on license plate fonts
STATE_AMBIGUITIES = {
    'TM': 'TN', 'TH': 'TN', 'IN': 'TN', '1N': 'TN', 'IT': 'TN', 'TI': 'TN',
    'T5': 'TS', '7S': 'TS', '0L': 'DL', 'D1': 'DL', 'K4': 'KA', 'K8': 'KA',
    'HH': 'MH', 'PH': 'MH', 'MM': 'MH', 'N1': 'NL'
}

def resolve_indian_plate_syntax(raw_text):
    """
    Applies position-aware Indian registration syntax post-processing:
    Standard format: [2 Letters State][2 Digits District][1-2 Letters Series][4 Digits Registration]

    1. Strips punctuation and non-alphanumeric noise.
    2. Strips leading 'IND' or hologram artifacts if followed by a recognized state code.
    3. Resolves known state code character ambiguities (e.g. TH/TM/IN -> TN).
    4. Strips trailing border/screw artifacts (e.g. trailing '1', 'I', or mounting screw digits after 4 registration digits).
    5. Applies position-aware character ambiguity corrections:
       - In digit positions (RTO district and final 4-digit unique ID), maps digit-like letters (Z->2, O->0, S->5, B->8).
       - In letter positions (State code and series), maps letter-like digits (0->O, 2->Z, 8->B, 4->A).
    6. Validates the result against standard Indian registration format.
       - Returns (formatted_string, status, is_valid)
       - If valid: e.g. ('MH 20 DV 2366', 'VALIDATED', True)
       - If unresolvable: returns cleaned string without inventing characters, e.g. ('...', 'UNCERTAIN', False)
    """
    if not raw_text:
        return "", "UNCERTAIN", False

    # Step 1: Normalize to uppercase alphanumeric
    cleaned = re.sub(r'[^A-Z0-9]', '', raw_text.upper())
    if not cleaned:
        return "", "UNCERTAIN", False

    # Step 2: Strip leading 'IND' badge text
    if cleaned.startswith("IND"):
        cleaned = cleaned[3:]

    # Step 3: Strip leading single/double artifact if a valid State Code follows immediately
    # (e.g. 'PMH' -> 'MH', 'MMH' -> 'MH', 'KDL' -> 'DL', 'HTS' -> 'TS')
    if len(cleaned) > 2 and cleaned[:2] not in INDIAN_STATE_CODES:
        if len(cleaned) > 3 and cleaned[1:3] in INDIAN_STATE_CODES:
            cleaned = cleaned[1:]
        elif len(cleaned) > 4 and cleaned[2:4] in INDIAN_STATE_CODES:
            cleaned = cleaned[2:]

    # Step 3.1: Resolve single-character state code ambiguity if still not in INDIAN_STATE_CODES
    if len(cleaned) >= 2 and cleaned[:2] not in INDIAN_STATE_CODES:
        prefix = cleaned[:2]
        if prefix in STATE_AMBIGUITIES:
            cleaned = STATE_AMBIGUITIES[prefix] + cleaned[2:]

    # Step 4: Strip trailing border line or bumper screw artifact
    # (e.g. 11-char string ending in an isolated screw digit after 4 registration digits)
    if len(cleaned) == 11 and cleaned[-1].isalnum():
        if any(c.isdigit() or c in DIGIT_FROM_LETTER for c in cleaned[-5:-1]):
            cleaned = cleaned[:10]
    elif len(cleaned) > 10 and cleaned[-1] in ('1', 'I'):
        if any(c.isdigit() or c in DIGIT_FROM_LETTER for c in cleaned[-5:-1]):
            cleaned = cleaned[:-1]

    chars = list(cleaned)
    n = len(chars)

    # Step 5: Identify expected position template
    # 10 characters: LL DD LL DDDD (e.g., MH 20 DV 2366)
    # 9 characters:  LL DD L  DDDD (e.g., DL 01 A  1234)
    if n == 10:
        pattern = ['L', 'L', 'D', 'D', 'L', 'L', 'D', 'D', 'D', 'D']
    elif n == 9:
        pattern = ['L', 'L', 'D', 'D', 'L', 'D', 'D', 'D', 'D']
    else:
        pattern = None

    # Step 6: Position-aware ambiguity correction
    if pattern:
        for i, expected in enumerate(pattern):
            c = chars[i]
            if expected == 'D' and c in DIGIT_FROM_LETTER:
                chars[i] = DIGIT_FROM_LETTER[c]
            elif expected == 'L':
                if c in LETTER_FROM_DIGIT:
                    chars[i] = LETTER_FROM_DIGIT[c]
                elif c == '4':
                    chars[i] = 'A'

    candidate = "".join(chars)

    # Step 7: Strict syntax regex validation
    match = re.match(r'^([A-Z]{2})([0-9]{2})([A-Z]{1,2})([0-9]{4})$', candidate)
    if match and match.group(1) in INDIAN_STATE_CODES:
        formatted_plate = f"{match.group(1)} {match.group(2)} {match.group(3)} {match.group(4)}"
        return formatted_plate, "VALIDATED", True
    else:
        # Do not invent characters if format cannot be confirmed
        return candidate, "UNCERTAIN", False


