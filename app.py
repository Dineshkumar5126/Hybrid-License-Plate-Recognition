import os
import sys
import time
import cv2
import numpy as np
import streamlit as st

# Add src/ to path so dip_pipeline can be imported directly
base_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(base_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import importlib
import dip_pipeline as dip
importlib.reload(dip)

# ----------------- PAGE CONFIGURATION -----------------
# 1x1 transparent PNG data URI ensures the browser tab shows a completely blank/neutral icon area
BLANK_FAVICON = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

st.set_page_config(
    page_title="Hybrid License Plate Recognition System",
    page_icon=BLANK_FAVICON,
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": None
    }
)

# ----------------- CACHED OCR ENGINE -----------------
@st.cache_resource(show_spinner="Initializing EasyOCR character recognition model...")
def get_ocr_reader():
    import easyocr
    return easyocr.Reader(['en'], gpu=False, verbose=False)

# ----------------- SESSION STATE INITIALIZATION -----------------
if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "Dark"

if "pipeline_result" not in st.session_state:
    st.session_state.pipeline_result = None

if "analyzed_image_id" not in st.session_state:
    st.session_state.analyzed_image_id = None

# ----------------- THEME STYLING GENERATOR -----------------
def get_custom_css(is_dark: bool) -> str:
    if is_dark:
        bg_primary = "#181a1f"
        bg_secondary = "#21252b"
        bg_card = "#21252b"
        bg_card_inner = "#282c34"
        border_color = "#333842"
        border_highlight = "#3b82f6"
        text_primary = "#e2e8f0"
        text_secondary = "#94a3b8"
        text_muted = "#64748b"
        accent_color = "#3b82f6"
        accent_hover = "#2563eb"
        valid_bg = "rgba(34, 197, 94, 0.15)"
        valid_border = "#16a34a"
        valid_text = "#4ade80"
        uncertain_bg = "rgba(245, 158, 11, 0.15)"
        uncertain_border = "#d97706"
        uncertain_text = "#fbbf24"
    else:
        bg_primary = "#f8f9fa"
        bg_secondary = "#ffffff"
        bg_card = "#ffffff"
        bg_card_inner = "#f1f3f5"
        border_color = "#dee2e6"
        border_highlight = "#2563eb"
        text_primary = "#212529"
        text_secondary = "#495057"
        text_muted = "#868e96"
        accent_color = "#2563eb"
        accent_hover = "#1d4ed8"
        valid_bg = "#e8f5e9"
        valid_border = "#81c784"
        valid_text = "#2e7d32"
        uncertain_bg = "#fff8e1"
        uncertain_border = "#ffd54f"
        uncertain_text = "#b78103"

    return f"""
    <style>
        /* Base typography and layout */
        html, body, [data-testid="stAppViewContainer"] {{
            background-color: {bg_primary} !important;
            color: {text_primary} !important;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }}

        [data-testid="stSidebar"] {{
            background-color: {bg_secondary} !important;
            border-right: 1px solid {border_color} !important;
        }}

        /* Hide Streamlit developer and toolbar controls */
        #MainMenu {{
            visibility: hidden !important;
            display: none !important;
        }}

        [data-testid="stToolbar"] {{
            visibility: hidden !important;
            display: none !important;
        }}

        [data-testid="stToolbarActions"] {{
            visibility: hidden !important;
            display: none !important;
        }}

        .stDeployButton, [data-testid="stDeployButton"] {{
            visibility: hidden !important;
            display: none !important;
        }}

        [data-testid="stBaseButton-header"] {{
            display: none !important;
        }}

        [data-testid="stDecoration"] {{
            visibility: hidden !important;
            display: none !important;
        }}

        footer, [data-testid="stFooter"] {{
            visibility: hidden !important;
            display: none !important;
        }}

        [data-testid="stHeader"] {{
            background-color: transparent !important;
            pointer-events: none;
        }}

        [data-testid="stHeader"] [data-testid="stSidebarCollapsedControl"] {{
            pointer-events: auto;
        }}

        /* Headings */
        h1, h2, h3, h4, h5, h6 {{
            color: {text_primary} !important;
            font-weight: 600 !important;
        }}

        p, span, label, div {{
            color: {text_primary};
        }}

        .stCaption, [data-testid="stCaptionContainer"] p {{
            color: {text_muted} !important;
            font-size: 0.82rem !important;
        }}

        /* Main Academic Header */
        .academic-header {{
            border-bottom: 1px solid {border_color};
            padding-bottom: 12px;
            margin-bottom: 20px;
        }}

        .academic-title {{
            font-size: 1.85rem !important;
            font-weight: 700 !important;
            color: {text_primary} !important;
            margin: 0 0 4px 0 !important;
            letter-spacing: -0.3px;
        }}

        .academic-subtitle {{
            font-size: 1.0rem !important;
            color: {text_secondary} !important;
            margin: 0 !important;
            font-weight: 400;
        }}

        /* Recognition Result Box */
        .result-container {{
            background-color: {bg_card};
            border: 1px solid {border_color};
            border-radius: 6px;
            padding: 16px 20px;
            margin-bottom: 18px;
            text-align: center;
        }}

        .result-label {{
            font-size: 0.78rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: {text_muted};
            margin-bottom: 6px;
        }}

        .result-plate {{
            font-family: "JetBrains Mono", "Consolas", monospace;
            font-size: 2.1rem;
            font-weight: 700;
            letter-spacing: 3px;
            color: {text_primary};
        }}

        /* Metric Box */
        .metric-card {{
            background-color: {bg_card};
            border: 1px solid {border_color};
            border-radius: 6px;
            padding: 12px 14px;
            height: 100%;
        }}

        .metric-title {{
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: {text_muted};
            margin-bottom: 6px;
        }}

        .metric-value {{
            font-size: 1.25rem;
            font-weight: 700;
            color: {text_primary};
            margin-bottom: 4px;
        }}

        .metric-sub {{
            font-size: 0.75rem;
            color: {text_secondary};
        }}

        /* Status Badges */
        .status-badge-valid {{
            display: inline-block;
            background: {valid_bg};
            border: 1px solid {valid_border};
            color: {valid_text};
            font-weight: 700;
            font-size: 0.88rem;
            padding: 3px 10px;
            border-radius: 4px;
        }}

        .status-badge-uncertain {{
            display: inline-block;
            background: {uncertain_bg};
            border: 1px solid {uncertain_border};
            color: {uncertain_text};
            font-weight: 700;
            font-size: 0.88rem;
            padding: 3px 10px;
            border-radius: 4px;
        }}

        /* Buttons */
        div.stButton > button[kind="primary"] {{
            background-color: {accent_color} !important;
            color: #ffffff !important;
            font-weight: 600 !important;
            border: none !important;
            border-radius: 4px !important;
            padding: 8px 18px !important;
            box-shadow: none !important;
            transition: background-color 0.15s ease !important;
        }}

        div.stButton > button[kind="primary"]:hover {{
            background-color: {accent_hover} !important;
        }}

        div.stButton > button[kind="secondary"] {{
            background-color: {bg_card} !important;
            border: 1px solid {border_color} !important;
            color: {text_primary} !important;
            font-weight: 500 !important;
            border-radius: 4px !important;
            box-shadow: none !important;
        }}

        div.stButton > button[kind="secondary"]:hover {{
            border-color: {accent_color} !important;
            color: {accent_color} !important;
        }}

        /* Expander */
        [data-testid="stExpander"] {{
            background-color: {bg_card} !important;
            border: 1px solid {border_color} !important;
            border-radius: 6px !important;
            margin-top: 14px !important;
        }}

        /* File Uploader Container */
        [data-testid="stFileUploader"] {{
            background-color: {bg_card_inner};
            border: 1px dashed {border_color};
            border-radius: 6px;
            padding: 14px;
        }}
    </style>
    """

# ----------------- SIDEBAR CONTROLS -----------------
st.sidebar.markdown("### Theme")
theme_choice = st.sidebar.radio(
    "Theme Selection:",
    options=["Dark Mode", "Light Mode"],
    index=0 if st.session_state.theme_mode == "Dark" else 1,
    horizontal=True,
    key="theme_radio_selector"
)
st.session_state.theme_mode = "Dark" if "Dark" in theme_choice else "Light"
is_dark = (st.session_state.theme_mode == "Dark")

# Inject Custom Dynamic CSS
st.markdown(get_custom_css(is_dark), unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### Input Mode")

input_mode = st.sidebar.radio(
    "Choose Vehicle Input Method:",
    options=["Benchmark Test Dataset", "Upload Vehicle Photo"],
    key="input_mode_selector"
)

data_dir = os.path.join(base_dir, "data")
sample_files = {
    "sample_car.jpg": "Skoda Octavia (Silver, Front View)",
    "car_1_mercedes_white.jpg": "Mercedes-Benz Sedan (White, Front View)",
    "car_2_taxi_angle.jpg": "Commercial Taxi (Angled Front View, Yellow Plate)",
    "car_3_ev_shadow.jpg": "Tata Nexon EV (Tailgate Shadow, Green Plate)"
}

input_image = None
image_name = ""
image_id = ""

if input_mode == "Benchmark Test Dataset":
    selected_key = st.sidebar.selectbox(
        "Select benchmark test vehicle:",
        options=list(sample_files.keys()),
        format_func=lambda k: f"{k} - {sample_files[k]}",
        key="benchmark_select"
    )
    image_name = selected_key
    image_id = f"benchmark_{selected_key}"
    file_path = os.path.join(data_dir, selected_key)
    if os.path.exists(file_path):
        input_image = cv2.imread(file_path)
    else:
        st.sidebar.error(f"Image not found at {file_path}")

st.sidebar.markdown("---")
st.sidebar.markdown("### DIP Pipeline Overview")
st.sidebar.markdown("""
- **Width Standardization:** 800 px aspect ratio preserved
- **Color Space:** 3-channel BGR to 1-channel Grayscale
- **Noise Reduction:** Bilateral smoothing (d=11, sigma=17)
- **Edge Gradient:** Sobel-X operator for vertical character strokes
- **Morphology:** Otsu binarization and rectangular closing
- **Enhancement:** Bicubic upscale, CLAHE, and Otsu binarization
- **OCR Engine:** EasyOCR character inference and regex validation
""")

# ----------------- MAIN HEADER -----------------
st.markdown("""
<div class="academic-header">
    <h1 class="academic-title">Hybrid License Plate Recognition System</h1>
    <p class="academic-subtitle">Digital Image Processing using OpenCV and OCR</p>
</div>
""", unsafe_allow_html=True)

# ----------------- INPUT SECTION -----------------
st.markdown("### Input Vehicle Image")

if input_mode == "Upload Vehicle Photo":
    uploaded_file = st.file_uploader(
        "Upload vehicle photo (JPG, JPEG, PNG):",
        type=["jpg", "jpeg", "png"],
        key="main_file_uploader",
        help="Supported formats: JPG, JPEG, PNG."
    )

    if uploaded_file is not None:
        try:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            input_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            image_name = uploaded_file.name
            image_id = f"upload_{uploaded_file.name}_{len(file_bytes)}"

            if input_image is None:
                st.warning("The uploaded file could not be decoded as an image. Please ensure it is a valid JPG, JPEG, or PNG file.")
        except Exception as e:
            st.warning("Could not read the uploaded image. Please try uploading another photo.")
            input_image = None
    else:
        st.info("Upload a vehicle photo above, or select 'Benchmark Test Dataset' from the sidebar.")

elif input_mode == "Benchmark Test Dataset" and input_image is not None:
    st.caption(f"Currently Selected: **{image_name}** - {sample_files.get(image_name, '')}")

# ----------------- PIPELINE EXECUTION HELPER -----------------
def execute_pipeline(img):
    """
    Executes the classical DIP localization + EasyOCR character recognition pipeline.
    Preserves exact detection, preprocessing, OCR, and validation logic without changes.
    """
    # 1. Standardize resolution
    std_img = dip.resize_to_standard_width(img, target_width=800)
    img_h, img_w = std_img.shape[:2]

    # 2. Grayscale, bilateral filter, Canny edges
    gray_img = dip.convert_to_grayscale(std_img)
    blurred_img = dip.apply_noise_reduction(gray_img)
    edged_img = dip.detect_edges(blurred_img)

    # 3. Contour Detection & Localization
    plate_contour = dip.find_plate_contours(edged_img, std_img)
    is_localized = (plate_contour is not None)

    if not is_localized:
        return {
            "is_localized": False,
            "std_img": std_img,
            "gray_img": gray_img,
            "blurred_img": blurred_img,
            "edged_img": edged_img,
            "img_w": img_w,
            "img_h": img_h
        }

    # Crop plate
    cropped_plate = dip.crop_plate(std_img, gray_img, plate_contour)

    # Preprocess plate for OCR
    preprocessed_plate = dip.preprocess_for_ocr(cropped_plate)

    # Run OCR with EasyOCR & evaluate multiple DIP representations
    reader = get_ocr_reader()
    candidates = []

    # 1. Preprocessed Plate (CLAHE + Otsu with Polarity Normalization)
    raw_pre, clean_pre, conf_pre = dip.recognize_plate_text(preprocessed_plate, reader=reader)
    res_pre, st_pre, val_pre = dip.resolve_indian_plate_syntax(raw_pre)
    candidates.append((val_pre, conf_pre, raw_pre, res_pre, st_pre, preprocessed_plate))

    # 2. Raw Cropped Plate
    raw_crop, clean_crop, conf_crop = dip.recognize_plate_text(cropped_plate, reader=reader)
    res_crop, st_crop, val_crop = dip.resolve_indian_plate_syntax(raw_crop)
    candidates.append((val_crop, conf_crop, raw_crop, res_crop, st_crop, cropped_plate))

    # 3. Fallback: Adaptive Gaussian Thresholding if initial representations are uncertain
    if not val_pre and not val_crop:
        preprocess_adaptive_fn = getattr(dip, 'preprocess_for_ocr_adaptive', dip.preprocess_for_ocr)
        preproc_adapt = preprocess_adaptive_fn(cropped_plate)
        raw_ad, clean_ad, conf_ad = dip.recognize_plate_text(preproc_adapt, reader=reader)
        res_ad, st_ad, val_ad = dip.resolve_indian_plate_syntax(raw_ad)
        candidates.append((val_ad, conf_ad, raw_ad, res_ad, st_ad, preproc_adapt))

    # Select best candidate: prioritize valid syntax, then highest confidence
    candidates.sort(key=lambda c: (1 if c[0] else 0, c[1]), reverse=True)
    _, conf, raw_text, validated_text, status, best_display_plate = candidates[0]

    # Annotated image with bounding box and label
    annotated_img = std_img.copy()
    cv2.drawContours(annotated_img, [plate_contour], -1, (0, 255, 0), 3)
    x, y, w, h = cv2.boundingRect(plate_contour)

    display_text = validated_text if validated_text else (raw_text if raw_text else "NOT RECOGNIZED")
    badge_label = f"Plate: {display_text} ({conf:.0%})"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.65
    thickness = 2
    (label_w, label_h), _ = cv2.getTextSize(badge_label, font, font_scale, thickness)
    badge_y1 = max(0, y - label_h - 14)
    badge_y2 = y
    cv2.rectangle(annotated_img, (x, badge_y1), (x + label_w + 12, badge_y2), (0, 0, 0), -1)
    cv2.rectangle(annotated_img, (x, badge_y1), (x + label_w + 12, badge_y2), (0, 255, 0), 2)
    cv2.putText(annotated_img, badge_label, (x + 6, badge_y2 - 6), font, font_scale, (0, 255, 0), thickness)

    return {
        "is_localized": True,
        "std_img": std_img,
        "gray_img": gray_img,
        "blurred_img": blurred_img,
        "edged_img": edged_img,
        "plate_contour": plate_contour,
        "cropped_plate": cropped_plate,
        "preprocessed_plate": preprocessed_plate,
        "best_display_plate": best_display_plate,
        "annotated_img": annotated_img,
        "x": x, "y": y, "w": w, "h": h,
        "conf": conf,
        "raw_text": raw_text,
        "validated_text": validated_text,
        "status": status,
        "display_text": display_text,
        "img_w": img_w,
        "img_h": img_h
    }

# ----------------- ACTION & EXECUTION SECTION -----------------
if input_image is not None:
    is_already_analyzed = (
        st.session_state.pipeline_result is not None
        and st.session_state.analyzed_image_id == image_id
    )

    # Show preview before analysis if not yet analyzed
    if not is_already_analyzed:
        col_prev, col_info = st.columns([1.2, 1])
        with col_prev:
            st.markdown("**Vehicle Image Preview:**")
            st.image(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB), width="stretch")
        with col_info:
            st.markdown(f"**Selected Image:** `{image_name}`")
            st.markdown(f"**Original Resolution:** `{input_image.shape[1]} × {input_image.shape[0]} px`")
            st.markdown(f"**Target Normalized Width:** `800 px`")
            if input_mode == "Benchmark Test Dataset":
                st.markdown(f"**Dataset Note:** {sample_files.get(image_name, '')}")
            st.markdown("<br>", unsafe_allow_html=True)

            analyze_clicked = st.button("Analyze Vehicle", type="primary", use_container_width=True)
            if analyze_clicked:
                with st.spinner("Processing through Digital Image Processing pipeline and OCR model..."):
                    try:
                        start_t = time.time()
                        res = execute_pipeline(input_image)
                        res["elapsed_sec"] = time.time() - start_t
                        st.session_state.pipeline_result = res
                        st.session_state.analyzed_image_id = image_id
                        st.rerun()
                    except Exception as e:
                        st.error("An unexpected error occurred while processing this vehicle image.")
                        with st.expander("Technical Error Diagnostics"):
                            st.code(str(e))
    else:
        # Action bar when already analyzed
        bar_col1, bar_col2 = st.columns([3, 1])
        with bar_col1:
            st.success(f"Analysis completed for {image_name}.")
        with bar_col2:
            if st.button("Analyze Another Image", type="secondary", use_container_width=True):
                st.session_state.pipeline_result = None
                st.session_state.analyzed_image_id = None
                st.rerun()

# ----------------- RESULTS SECTION -----------------
if st.session_state.pipeline_result is not None and st.session_state.analyzed_image_id == image_id:
    result = st.session_state.pipeline_result

    if result["is_localized"]:
        display_text = result["display_text"]
        status = result["status"]
        conf = result["conf"]
        raw_text = result["raw_text"]
        x, y, w, h = result["x"], result["y"], result["w"], result["h"]
        annotated_img = result["annotated_img"]
        cropped_plate = result["cropped_plate"]
        best_display_plate = result["best_display_plate"]
        elapsed = result.get("elapsed_sec", 0.0)

        st.markdown("---")
        st.markdown("### Recognition Result")

        # 1. Recognized License Plate Box
        st.markdown(f"""
        <div class="result-container">
            <div class="result-label">Recognized License Plate</div>
            <div class="result-plate">{display_text}</div>
        </div>
        """, unsafe_allow_html=True)

        # 2. Results Metrics Row
        mcol1, mcol2, mcol3, mcol4 = st.columns(4)

        with mcol1:
            status_html = (
                "<div class='status-badge-valid'>VALIDATED</div>"
                if status == "VALIDATED"
                else "<div class='status-badge-uncertain'>UNCERTAIN</div>"
            )
            sub_label = "Indian HSRP format matched" if status == "VALIDATED" else "Format unverified"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Validation Status</div>
                <div style="margin: 6px 0;">{status_html}</div>
                <div class="metric-sub">{sub_label}</div>
            </div>
            """, unsafe_allow_html=True)

        with mcol2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">OCR Confidence</div>
                <div class="metric-value">{conf:.1%}</div>
                <div class="metric-sub">EasyOCR character score</div>
            </div>
            """, unsafe_allow_html=True)

        with mcol3:
            raw_display = raw_text if raw_text else "None"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Raw OCR Output</div>
                <div class="metric-value" style="font-family: monospace; font-size: 1.15rem;">{raw_display}</div>
                <div class="metric-sub">Unfiltered character text</div>
            </div>
            """, unsafe_allow_html=True)

        with mcol4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Plate Geometry</div>
                <div class="metric-value" style="font-size: 1.15rem;">{w} × {h} px</div>
                <div class="metric-sub">Aspect ratio: {w/max(h,1):.2f}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # 3. Visual Verification Images
        st.markdown("### Visual Verification")
        vcol1, vcol2, vcol3 = st.columns([1.6, 1.2, 1.2])

        with vcol1:
            st.markdown("**1. Original Vehicle with Localized Plate**")
            st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), width="stretch")
            st.caption(f"Bounding contour localized at (x={x}, y={y}, w={w}, h={h})")

        with vcol2:
            st.markdown("**2. Detected and Cropped Plate**")
            if cropped_plate is not None:
                st.image(cropped_plate, clamp=True, width="stretch")
                st.caption(f"Extracted plate region: {cropped_plate.shape[1]} × {cropped_plate.shape[0]} px")

        with vcol3:
            st.markdown("**3. Enhanced Plate for OCR**")
            if best_display_plate is not None:
                st.image(best_display_plate, clamp=True, width="stretch")
                st.caption(f"Bicubic scaling, CLAHE, and Otsu thresholding ({best_display_plate.shape[1]} × {best_display_plate.shape[0]} px)")

        # 4. Expandable Digital Image Processing (DIP) Breakdown
        with st.expander("Step-by-Step DIP Processing Pipeline Stages", expanded=False):
            st.markdown("Inspect each algorithmic stage of the image processing pipeline:")

            dcol1, dcol2, dcol3 = st.columns(3)

            with dcol1:
                st.markdown("**1. Grayscale Conversion**")
                st.caption("Converts 3-channel BGR matrix to single-channel luminance intensity: Y = 0.299R + 0.587G + 0.114B.")
                st.image(result["gray_img"], clamp=True, width="stretch")

            with dcol2:
                st.markdown("**2. Noise Reduction**")
                st.caption("Edge-preserving bilateral smoothing (d=11, sigma_color=17, sigma_space=17) reduces noise while maintaining edge contrast.")
                st.image(result["blurred_img"], clamp=True, width="stretch")

            with dcol3:
                st.markdown("**3. Edge Detection**")
                st.caption("Canny edge detector computes gradient magnitude, applies non-maximum suppression, and hysteresis thresholding.")
                st.image(result["edged_img"], clamp=True, width="stretch")

            st.markdown("<br>", unsafe_allow_html=True)

            dcol4, dcol5, dcol6 = st.columns(3)

            with dcol4:
                st.markdown("**4. License Plate Localization**")
                st.caption("Sobel-X vertical gradient, Otsu binarization, and rectangular morphological closing bridge character strokes into a unified plate contour.")
                st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), width="stretch")

            with dcol5:
                st.markdown("**5. Plate Preprocessing**")
                st.caption("Bicubic upscaling (2x), CLAHE local contrast enhancement, and Otsu thresholding prepare character glyphs for OCR.")
                st.image(best_display_plate, clamp=True, width="stretch")

            with dcol6:
                st.markdown("**6. OCR and Validation**")
                st.caption("EasyOCR inference detects alphanumeric characters, followed by Indian Motor Vehicles Act regex pattern matching.")
                st.image(cropped_plate, clamp=True, width="stretch")

    else:
        # Localization Failed
        st.markdown("---")
        st.error("License plate could not be localized. The classical DIP pipeline did not find a matching plate contour.")

        st.markdown("""
        **Localization Diagnostics (Classical DIP Heuristics)**

        The classical morphological localization pipeline filters candidates based on geometric constraints:
        - **Aspect Ratio:** Expected range is between 1.8 and 6.5.
        - **Relative Width:** Typical plates occupy 10% to 65% of the standardized image width.
        - **Stroke Edge Density:** Sobel-X vertical gradient closing requires distinct vertical character stroke transitions.

        **Recommendations for demonstration:**
        - Ensure the vehicle photo is taken from a direct front or rear angle.
        - Avoid extreme oblique angles or heavy shadows obscuring plate characters.
        - Select one of the benchmark vehicles from the sidebar to inspect a verified compliant plate.
        """)

        st.markdown("**Standardized Input Image Inspected by Pipeline:**")
        st.image(cv2.cvtColor(result["std_img"], cv2.COLOR_BGR2RGB), width="stretch")
