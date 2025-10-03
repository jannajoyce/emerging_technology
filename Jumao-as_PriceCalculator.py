import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def estimate_print_cost(img_path):
    # Load the input image
    src = cv.imread(img_path)
    if src is None:
        raise ValueError("Could not load the specified image.")

    # Resize to 900x600
    src = cv.resize(src, (900, 600))

    # Convert to grayscale
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

    # Split into BGR channels
    b, g, r = cv.split(src)

    colors = ('b', 'g', 'r')
    plt.figure()
    plt.title("Color Histogram")
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")
    for ch, col in zip([b, g, r], colors):
        hist = cv.calcHist([ch], [0], None, [256], [0, 256])
        hist = hist / hist.sum()
        plt.plot(hist, color=col)
    plt.xlim([0, 256])
    plt.show()

    # Difference between grayscale and each channel
    diff_b = cv.absdiff(gray, b)
    diff_g = cv.absdiff(gray, g)
    diff_r = cv.absdiff(gray, r)

    # Mean difference = colorfulness
    colorfulness = (np.mean(diff_b) + np.mean(diff_g) + np.mean(diff_r)) / 3.0

    # Normalize colorfulness between 0 and 1
    color_ratio = min(1.0, colorfulness / 128.0)  # 128 is heuristic scaling factor

    if color_ratio < 0.05:   # nearly grayscale
        price = 2.00
    else:
        base_cost = 5.00
        top_cost = 20.00
        price = base_cost + (top_cost - base_cost) * color_ratio

    return round(price, 2)

img_file = "C:\don't open this\emerging_technology\hohoho.jpg"
cost = estimate_print_cost(img_file)
print(f"Estimated printing charge: {cost:.2f} pesos")

# Show the image
preview = cv.imread(img_file)
cv.imshow("Input Image", preview)
cv.waitKey(0)
cv.destroyAllWindows()
