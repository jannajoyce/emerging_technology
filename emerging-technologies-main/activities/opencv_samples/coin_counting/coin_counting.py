# Here is the simple coin counter. You may improve this program.

import cv2
import numpy as np

def white_balance(img):
 
    img_LAB = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(img_LAB[:, :, 1])
    avg_b = np.average(img_LAB[:, :, 2])
    
    # Apply white balance correction
    img_LAB[:, :, 1] = img_LAB[:, :, 1] - ((avg_a - 128) * (img_LAB[:, :, 0] / 255.0) * 6)
    img_LAB[:, :, 2] = img_LAB[:, :, 2] - ((avg_b - 128) * (img_LAB[:, :, 0] / 255.0) * 0)
    
    balanced_image = cv2.cvtColor(img_LAB, cv2.COLOR_LAB2BGR)
    return balanced_image

def resize_image(image, target_width=775):
  
    (h, w) = image.shape[:2]
    aspect_ratio = h / w
    new_height = int(target_width * aspect_ratio)
    return cv2.resize(image, (target_width, new_height))

def preprocess_image(image):

    # Apply white balance
    balanced_image = white_balance(image)
    
    # Convert to different color spaces
    lab_image = cv2.cvtColor(balanced_image, cv2.COLOR_BGR2LAB)
    hsv_image = cv2.cvtColor(balanced_image, cv2.COLOR_BGR2HSV)
    
    # Split channels
    (L, A, B) = cv2.split(lab_image)
    (H, S, V) = cv2.split(hsv_image)
    
    # Create merged image combining different channels
    merged = cv2.merge([A, S, L])
    
    # Convert to grayscale and apply Gaussian blur
    gray = cv2.cvtColor(merged, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    
    return blurred, balanced_image

def detect_coins(processed_image):
  
    # Apply Canny edge detection
    edged = cv2.Canny(processed_image, 0, 68)
    
    # Find contours
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return contours

def main():
    # Load and resize the image
    image = cv2.imread("coins (7).jpg")
    if image is None:
        print("Error: Could not load image. Please check the file path.")
        return
    
    resized_image = resize_image(image)
    
    # Preprocess the image
    processed_image, balanced_image = preprocess_image(resized_image)
    
    # Detect coins
    contours = detect_coins(processed_image)
    
    print(f"Detected coins in this image: {len(contours)}")
    
    # Draw contours on the original resized image
    coins_display = resized_image.copy()
    cv2.drawContours(coins_display, contours, -1, (0, 255, 0), 2)
    
    # Display results
    cv2.imshow("Processed", processed_image)
    cv2.imshow("Display", coins_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()