# Hybrid License Plate Recognition System

A 2-member college-level Digital Image Processing (DIP) semester project that detects, crops, preprocesses, and recognizes characters from vehicle license plates using **classical OpenCV image processing** and **EasyOCR**, accompanied by an **Indian syntax ambiguity resolver** and an interactive **Streamlit web application**.

---

## 1. Problem Statement

Automated License Plate Recognition (ALPR) in real-world conditions faces frequent challenges:
- High-resolution or varying vehicle image resolutions lead to unpredictable filter behaviors.
- Vehicle bodies contain high-contrast geometric structures (radiator grilles, bumper honeycomb meshes, badge emblems, and headlights) that easily trigger false-positive edge contours.
- Low-contrast or non-uniform illumination degrades character segmentation.
- Standard OCR engines frequently confuse visually similar characters (such as `2` ↔ `Z`, `0` ↔ `O`, `8` ↔ `B`, and `5` ↔ `S`) or pick up manufacturer logos (e.g. `IND` emblem).

---

## 2. Project Objectives

1. **Resolution Standardization:** Normalize input images to a uniform width while maintaining the natural aspect ratio to make morphological kernel sizes resolution-invariant.
2. **Classical DIP Localization:** Isolate the license plate region using explainable digital image processing techniques (Sobel horizontal gradients, Otsu thresholding, rectangular morphological closing, and contour geometry filtering) without relying on heavy deep-learning object detectors (YOLO/SSD).
3. **Plate Image Enhancement:** Apply multi-stage preprocessing ($2.0\times$ bicubic upscaling, CLAHE local contrast equalization, and Otsu binarization) to enhance character legibility prior to OCR.
4. **Character Recognition & Validation:** Extract alphanumeric characters using EasyOCR and refine raw text via a syntax-aware Indian registration parser with ambiguity correction.
5. **Interactive Demonstration:** Provide both a reproducible command-line interface (CLI) and an interactive web interface (Streamlit) for live evaluation of benchmark test vehicles or custom uploads.

---

## 3. Technologies & Libraries Used

| Technology / Library | Purpose in Project |
| :--- | :--- |
| **Python 3.10+** | Core programming language |
| **OpenCV (`opencv-python`)** | Image ingestion, color conversion, bilateral filtering, Canny edge detection, Sobel gradients, morphology, contour analysis, and CLAHE |
| **NumPy** | Array manipulations, coordinate masks, and bounding-box slicing |
| **Matplotlib** | Generating multi-panel pipeline diagnostic figures |
| **EasyOCR** | Alphanumeric optical character recognition |
| **Streamlit** | Interactive user interface with file upload and live visual stage rendering |

---

## 4. Complete DIP Pipeline Architecture

The end-to-end recognition pipeline operates in six sequential stages:

```
[Input Vehicle Image]
         │
         ▼
[1. Resolution Standardization] ──► Target width: 800 px (INTER_AREA / INTER_CUBIC)
         │
         ▼
[2. Grayscale & Bilateral Blur] ──► cv2.bilateralFilter(d=11, sigma=17) for edge-preserving smoothing
         │
         ▼
[3. License Plate Localization] ──► Sobel-X Gradient + Otsu Binarization + Morphological Close (25×7)
         │
         ▼
[4. Plate Extraction & Crop]    ──► Mask-based ROI extraction
         │
         ▼
[5. Plate OCR Preprocessing]    ──► 2.0× Bicubic Upscaling + CLAHE (clipLimit=2.0) + Otsu Binarization
         │
         ▼
[6. OCR & Syntax Resolution]    ──► EasyOCR inference + Indian Syntax Validator (Regex & Ambiguity Map)
         │
         ▼
[Final Output Visualizations]   ──► Terminal summary, annotated image, and Streamlit UI display
```

### Detailed DIP Stage Explanations

1. **Resolution Standardization (`resize_to_standard_width`):**
   - Resizes any input vehicle image to a standard width of $800\text{ px}$ while preserving its original aspect ratio.
   - Uses `cv2.INTER_AREA` when downscaling high-resolution photos to avoid aliasing artifacts, and `cv2.INTER_CUBIC` when upscaling smaller images to retain sharp edge definitions.
2. **Grayscale & Bilateral Filtering (`convert_to_grayscale`, `apply_noise_reduction`):**
   - Converts 3-channel BGR to single-channel luminance ($Y = 0.299R + 0.587G + 0.114B$).
   - Applies bilateral filtering (`d=11, sigmaColor=17, sigmaSpace=17`) which replaces each pixel value with a weighted average of nearby pixels, smoothing homogeneous vehicle paint while strictly preserving sharp character boundaries.
3. **Horizontal Gradient & Morphological Closing (`find_plate_contours`):**
   - **Sobel-X Gradient:** Computes the horizontal derivative ($\frac{\partial I}{\partial x}$) using `cv2.Sobel(..., dx=1, dy=0, ksize=3)`. Because license plate characters consist of dense vertical strokes, Sobel-X strongly highlights letters and numbers while suppressing horizontal bumper lines and grilles.
   - **Otsu Automatic Binarization:** Calculates the optimal global threshold dynamically by minimizing intra-class variance of the gradient magnitudes.
   - **Morphological Closing:** Applies `cv2.morphologyEx(..., cv2.MORPH_CLOSE, kernel)` using a wide horizontal rectangular structuring element ($25 \times 7\text{ px}$). This dilates and bridges the individual vertical character strokes into a unified, solid rectangular blob, followed by erosion to restore outer boundaries.
   - **Morphological Opening:** A small $3 \times 3$ kernel cleans up isolated point noise.
   - **Contour Filtering & Scoring:** Extracts external contours and scores candidates based on:
     - Vertical position filter ($0.40 \cdot H \le y \le 0.85 \cdot H$) to target the bumper zone.
     - Aspect ratio constraint ($2.0 \le \frac{w}{h} \le 6.0$).
     - Relative width constraint ($0.12 \le \frac{w}{W} \le 0.50$).
     - High character edge density ($0.08 \le \text{density} \le 0.60$).
4. **Plate Region Cropping (`crop_plate`):**
   - Slices the region of interest (ROI) from both the color original and grayscale image using the mask bounding coordinates.
5. **OCR Preprocessing (`preprocess_for_ocr`):**
   - **2.0× Bicubic Upscaling:** Expands character pixel dimensions to ensure stroke widths meet EasyOCR's minimum receptive field.
   - **CLAHE (Contrast Limited Adaptive Histogram Equalization):** Divides the plate into an $8 \times 8$ grid of tiles and equalizes local histograms with a clipping limit of $2.0$ to prevent noise amplification while maximizing letter-to-background contrast.
   - **Gaussian Blur & Otsu Binarization:** Suppresses background specular reflections and creates high-contrast black characters on white background.
6. **Character Recognition & Syntax Ambiguity Resolver (`recognize_plate_text`, `resolve_indian_plate_syntax`):**
   - EasyOCR reads character text using an alphanumeric allowlist (`0-9, A-Z`).
   - The dual-representation selector evaluates both the preprocessed binary plate and raw color crop, selecting the candidate with higher confidence and syntax validity.
   - **Syntax Validation:** Tests against standard Indian vehicle registration format:
     $$\underbrace{\text{[A-Z]\{2\}}}_{\text{State Code}}\ \underbrace{\text{[0-9]\{2\}}}_{\text{District RTO}}\ \underbrace{\text{[A-Z]\{1,2\}}}_{\text{Series}}\ \underbrace{\text{[0-9]\{4\}}}_{\text{Unique Number}}$$
   - Strips leading `IND` or `MM` High Security Plate logo artifacts.
   - Applies position-aware ambiguity corrections (`Z` $\rightarrow$ `2` in digit positions; `O` $\rightarrow$ `0`; `S` $\rightarrow$ `5`; `B` $\rightarrow$ `8`).
   - Validates state codes against all 36 Indian States and Union Territories (e.g. `MH`, `DL`, `KA`, `TN`, `TS`, etc.).
   - Marks non-conforming or low-confidence outputs as **`UNCERTAIN`** rather than hallucinating characters.

---

## 5. Project Directory Structure

```
Hybrid-License-Plate-Recognition/
├── .streamlit/
│   └── config.toml             # Streamlit server and UI configuration (toolbarMode = minimal)
├── assets/
│   └── favicon.png             # Neutral project tab favicon (no emoji)
├── app.py                      # Interactive Streamlit Web Application
├── requirements.txt            # Project dependencies
├── README.md                   # Comprehensive project documentation
├── .gitignore                  # Git ignore rules for clean repository
├── data/                       # Benchmark test vehicle images
│   ├── sample_car.jpg          # Silver Škoda Octavia (Baseline Front View)
│   ├── car_1_mercedes_white.jpg# White Mercedes-Benz Sedan (Daylight Front View)
│   ├── car_2_taxi_angle.jpg    # Commercial Taxi (Angled Front View, Yellow Plate)
│   └── car_3_ev_shadow.jpg     # Tata Nexon EV (Tailgate Shadow, Green Plate)
├── output/                     # Generated visual outputs & intermediate artifacts
│   ├── .gitkeep                # Directory preservation marker
│   ├── 1_grayscale.jpg         # Grayscale converted image
│   ├── 2_blurred.jpg           # Bilateral filtered image
│   ├── 3_edges.jpg             # Canny edge map
│   ├── 4_contour.jpg           # Vehicle with localized bounding box
│   ├── 5_cropped_plate.jpg     # Extracted license plate crop
│   ├── 6_pipeline_summary.png  # 6-stage Matplotlib visual summary
│   ├── 7_preprocessed_plate.jpg# Upscaled, CLAHE-enhanced binarized plate
│   └── 8_recognized_plate.jpg  # Final annotated output with text badge
└── src/                        # Core DIP algorithms and CLI runner
    ├── __init__.py             # Python package marker
    ├── dip_pipeline.py         # Complete modular DIP & OCR functions
    └── main.py                 # Standalone CLI execution script
```

---

## 6. Setup and Installation

### Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13 installed.
- Recommended: A virtual environment (`venv` or `conda`).

### Step 1: Clone or Navigate to the Repository
```bash
cd Hybrid-License-Plate-Recognition
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 7. How to Run the Project

### Option A: Interactive Streamlit Web Interface (Recommended for Demonstration)
Launch the web interface in your browser:
```bash
streamlit run app.py
```
*The web interface will open automatically at `http://localhost:8501`.*
- **Benchmark Dataset Mode:** Select any of the 4 benchmark cars from the sidebar dropdown to run instant analysis.
- **Upload Mode:** Select "Upload Vehicle Photo" from the sidebar radio button to upload any custom vehicle image (`.jpg`, `.jpeg`, `.png`).
- **Intermediate Inspection:** Click the expander at the bottom to inspect intermediate Grayscale, Bilateral, Canny, and morphological stages.

### Option B: Standalone CLI Execution
Run the classical CLI pipeline on the default sample car:
```bash
python src/main.py
```
Or specify any custom vehicle image path as a command-line argument:
```bash
python src/main.py data/car_1_mercedes_white.jpg
```
*All 8 visual artifacts will be generated and saved inside the `output/` directory.*

---

## 8. Deployment Considerations

- **Streamlit Community Cloud (Recommended):**
  This project is built using Streamlit and OpenCV, requiring a continuous Python process with WebSocket connections for real-time state management. The natural and recommended free deployment path is:
  ```
  GitHub Repository ──► Streamlit Community Cloud (https://share.streamlit.io)
  ```
  Simply connect the GitHub repository and specify `app.py` as the main entry point.
- **Serverless Platforms (e.g., Vercel):**
  Serverless platforms like Vercel are designed for short-lived, stateless HTTP request/response lambdas (e.g., Next.js, static frontends, or FastAPI endpoints). They do **not** natively support long-running, stateful Python WebSocket servers required by Streamlit. Therefore, deploying directly to Vercel without rewriting the application into a different backend/frontend architecture is not feasible. Streamlit Community Cloud or container-based hosts (Render, Railway, Hugging Face Spaces) are the appropriate deployment targets.

---

## 9. Benchmark Evaluation & Test Results

The system was evaluated across 4 diverse vehicle test images. The results are reported transparently below:

| Image File | Vehicle Details | Ground Truth | Localized? | Raw OCR | Cleaned / Validated Text | Status | Confidence | Analysis |
| :--- | :--- | :---: | :---: | :--- | :--- | :---: | :---: | :--- |
| [`data/sample_car.jpg`](data/sample_car.jpg) | Silver Škoda Octavia (Front View) | `MH20DV2366` | **YES** | `PMHZODV23661` | `MH 20 DV 2366` | **`VALIDATED`** | `24.45%` | **Complete Success:** Sobel-X bypassed the lower honeycomb grille. Prefixes and digit ambiguities (`O` $\rightarrow$ `0`) were resolved, matching valid Maharashtra syntax. |
| [`data/car_1_mercedes_white.jpg`](data/car_1_mercedes_white.jpg) | White Mercedes-Benz (Front View) | `TN09BY1314` | **YES** | `TH09BY1314` | `TH09BY1314` | **`UNCERTAIN`** | `47.90%` | **Localization Success:** Plate tightly isolated. OCR confused `N` with `H`. Because `TH` is an unverified state code, the validator safely marked it `UNCERTAIN` without inventing letters. |
| [`data/car_2_taxi_angle.jpg`](data/car_2_taxi_angle.jpg) | Commercial Taxi (Angled Front View) | `TN59AQ1515` | **YES** | `259A01515` | `ZS9AO1515` | **`UNCERTAIN`** | `58.56%` | **Localization Success:** Localized yellow taxi plate at an angle. Initial `TN` was clipped by bumper bracket into `2`. Ambiguity mapping adjusted digits to letters, correctly keeping `ZS` as unverified syntax. |
| [`data/car_3_ev_shadow.jpg`](data/car_3_ev_shadow.jpg) | Tata Nexon EV (Tailgate Shadow) | `TS08HM2223` | **NO** | *N/A* | *N/A* | **`FAILED_LOCALIZATION`** | *N/A* | **Controlled Heuristic Failure:** Close-up framing caused the plate contour to occupy $59.5\%$ of image width, intentionally filtered by the $50\%$ relative width cap. Dark green EV contrast also altered Otsu edge gradients. |

---

## 10. Current Limitations & Honest Academic Analysis

As appropriate for a classical Digital Image Processing project, the system has known boundaries:
1. **Camera Framing & Geometric Heuristics:**
   The localization stage assumes the license plate occupies between $12\%$ and $50\%$ of the image width and is located in the lower half of the vehicle ($y \ge 0.40 \cdot H$). Extreme close-ups (where the plate fills $>50\%$ of the frame) or high-mounted plates on tall commercial trucks may be rejected by the geometric candidate filters.
2. **Perspective Angle & Skew:**
   Plates captured from steep side angles suffer from horizontal character compression. Classical DIP bounding boxes (`cv2.boundingRect`) extract axis-aligned rectangles; implementing rotated bounding boxes (`cv2.minAreaRect`) and affine deskewing would further improve angled OCR accuracy.
3. **Non-Standard Plate Contrast:**
   Standard white plates with black lettering achieve high contrast under Otsu thresholding. Inverted contrast plates (e.g. green Electric Vehicle plates with white lettering) produce inverted binary silhouettes unless contrast polarity is explicitly normalized.
4. **OCR Font Ambiguities:**
   Without deep language modeling, standard Latin OCR models can confuse condensed sans-serif glyphs (`N` vs `H` or `8` vs `B`) under soft lighting.

---

## 11. Summary for Project Review

- **Core Academic Focus:** Classical Digital Image Processing (Bilateral filtering, Sobel gradients, Otsu binarization, morphological filtering, and CLAHE contrast enhancement).
- **Novelty:** Hybrid integration of classical DIP with EasyOCR and a post-processing Indian syntax ambiguity resolver.
- **User Interface:** Interactive Streamlit web app (`app.py`) allowing instant review demonstration on multiple vehicles and user uploads.
