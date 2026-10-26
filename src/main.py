import os
import sys
import cv2
import matplotlib.pyplot as plt
import dip_pipeline as dip

def main():
    # 1. Define paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if len(sys.argv) > 1:
        input_image_path = sys.argv[1]
        if not os.path.isabs(input_image_path):
            input_image_path = os.path.join(base_dir, input_image_path)
    else:
        input_image_path = os.path.join(base_dir, "data", "sample_car.jpg")
    output_dir = os.path.join(base_dir, "output")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Check if input image exists, if not, print a helpful message
    if not os.path.exists(input_image_path):
        print(f"Error: Could not find input image at {input_image_path}")
        print("Please place a vehicle image named 'sample_car.jpg' in the 'data' directory.")
        return

    try:
        # 2. Image Loading
        print("Loading image...")
        original_img = dip.load_image(input_image_path)
        
        # 2.1 Standardize Image Resolution (width=800px)
        print("Standardizing image resolution (width=800px)...")
        original_img = dip.resize_to_standard_width(original_img, target_width=800)

        # 3. Grayscale Conversion
        print("Converting to grayscale...")
        gray_img = dip.convert_to_grayscale(original_img)
        cv2.imwrite(os.path.join(output_dir, "1_grayscale.jpg"), gray_img)
        
        # 4. Noise Reduction
        print("Applying noise reduction...")
        blurred_img = dip.apply_noise_reduction(gray_img)
        cv2.imwrite(os.path.join(output_dir, "2_blurred.jpg"), blurred_img)
        
        # 5. Edge Detection
        print("Detecting edges...")
        edged_img = dip.detect_edges(blurred_img)
        cv2.imwrite(os.path.join(output_dir, "3_edges.jpg"), edged_img)
        
        # 6. Contour Detection & Localization
        print("Finding license plate contour...")
        plate_contour = dip.find_plate_contours(edged_img, original_img)
        
        if plate_contour is not None:
            print("License plate contour found!")
            # Draw contour on a copy of the original image to visualize
            contour_img = original_img.copy()
            cv2.drawContours(contour_img, [plate_contour], -1, (0, 255, 0), 3)
            cv2.imwrite(os.path.join(output_dir, "4_contour.jpg"), contour_img)
            
            # 7. Cropping
            print("Cropping license plate...")
            cropped_plate = dip.crop_plate(original_img, gray_img, plate_contour)
            cv2.imwrite(os.path.join(output_dir, "5_cropped_plate.jpg"), cropped_plate)
            
            # 8. Displaying Results using Matplotlib
            fig, axs = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle('License Plate Localization Pipeline')
            
            axs[0, 0].imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
            axs[0, 0].set_title('Original Image')
            axs[0, 0].axis('off')
            
            axs[0, 1].imshow(cv2.cvtColor(gray_img, cv2.COLOR_BGR2RGB))
            axs[0, 1].set_title('Grayscale')
            axs[0, 1].axis('off')
            
            axs[0, 2].imshow(cv2.cvtColor(blurred_img, cv2.COLOR_BGR2RGB))
            axs[0, 2].set_title('Noise Reduction (Bilateral)')
            axs[0, 2].axis('off')
            
            axs[1, 0].imshow(cv2.cvtColor(edged_img, cv2.COLOR_BGR2RGB))
            axs[1, 0].set_title('Canny Edges')
            axs[1, 0].axis('off')
            
            axs[1, 1].imshow(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB))
            axs[1, 1].set_title('Localized Contour')
            axs[1, 1].axis('off')
            
            axs[1, 2].imshow(cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2RGB), cmap='gray')
            axs[1, 2].set_title('Cropped Plate')
            axs[1, 2].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "6_pipeline_summary.png"))
            plt.close(fig)
            print("Pipeline summary saved to output/6_pipeline_summary.png")

            # 9. Plate Preprocessing for OCR
            print("Preprocessing cropped plate for OCR...")
            preprocessed_plate = dip.preprocess_for_ocr(cropped_plate)
            cv2.imwrite(os.path.join(output_dir, "7_preprocessed_plate.jpg"), preprocessed_plate)
            print("Preprocessed plate saved to output/7_preprocessed_plate.jpg")

            # 10. Character Recognition via EasyOCR & Multi-Representation Evaluation
            print("Recognizing characters with EasyOCR...")
            candidates = []

            # Representation 1: Preprocessed Plate (CLAHE + Otsu with Polarity Normalization)
            raw_pre, clean_pre, conf_pre = dip.recognize_plate_text(preprocessed_plate)
            res_pre, st_pre, val_pre = dip.resolve_indian_plate_syntax(raw_pre)
            candidates.append((val_pre, conf_pre, raw_pre, res_pre, st_pre))

            # Representation 2: Raw Cropped Plate
            raw_crop, clean_crop, conf_crop = dip.recognize_plate_text(cropped_plate)
            res_crop, st_crop, valid_crop = dip.resolve_indian_plate_syntax(raw_crop)
            candidates.append((valid_crop, conf_crop, raw_crop, res_crop, st_crop))

            # Representation 3 (Fallback): Adaptive Gaussian Thresholding if initial attempts are uncertain
            if not val_pre and not valid_crop:
                preproc_adapt = dip.preprocess_for_ocr_adaptive(cropped_plate)
                raw_ad, clean_ad, conf_ad = dip.recognize_plate_text(preproc_adapt)
                res_ad, st_ad, val_ad = dip.resolve_indian_plate_syntax(raw_ad)
                candidates.append((val_ad, conf_ad, raw_ad, res_ad, st_ad))

            # Select best candidate: prioritize valid syntax, then highest confidence
            candidates.sort(key=lambda c: (1 if c[0] else 0, c[1]), reverse=True)
            _, conf, raw_text, validated_text, status = candidates[0]

            display_text = validated_text if validated_text else "NOT RECOGNIZED"

            print("\n" + "=" * 55)
            print("       LICENSE PLATE RECOGNITION RESULT")
            print("=" * 55)
            print(f"  Raw OCR Output:          {raw_text}")
            print(f"  Cleaned/Validated Text:  {validated_text}")
            print(f"  Validation Status:       {status}")
            print(f"  Confidence Score:        {conf:.2%}")
            print("=" * 55 + "\n")

            # 11. Generate Recognized Plate Image
            annotated_img = original_img.copy()
            cv2.drawContours(annotated_img, [plate_contour], -1, (0, 255, 0), 3)

            x, y, w, h = cv2.boundingRect(plate_contour)
            label = f"Plate: {display_text} ({conf:.0%})"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.65
            thickness = 2
            (label_w, label_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

            # Draw background badge above plate
            badge_y1 = max(0, y - label_h - 14)
            badge_y2 = y
            cv2.rectangle(annotated_img, (x, badge_y1), (x + label_w + 12, badge_y2), (0, 0, 0), -1)
            cv2.rectangle(annotated_img, (x, badge_y1), (x + label_w + 12, badge_y2), (0, 255, 0), 2)
            cv2.putText(annotated_img, label, (x + 6, badge_y2 - 6), font, font_scale, (0, 255, 0), thickness)

            cv2.imwrite(os.path.join(output_dir, "8_recognized_plate.jpg"), annotated_img)
            print("Recognized plate visualization saved to output/8_recognized_plate.jpg")
            print(f"All processing complete! Results saved to {output_dir}")

            
        else:
            print("Warning: Could not find a suitable license plate contour in the image.")

            
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
