import cv2
import numpy as np
import matplotlib.pyplot as plt

# Preprocessing function with aspect ratio standardization
def preprocess_image(image_path, target_size=(300, 300)):
    image = cv2.imread(image_path) 
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  
    height, width = image.shape

    target_width, target_height = target_size
    scale = min(target_width / width, target_height / height)

    new_width = int(width * scale)
    new_height = int(height * scale)

    resized_image = cv2.resize(image, (new_width, new_height))

    delta_w = target_width - new_width
    delta_h = target_height - new_height
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    padded_image = cv2.copyMakeBorder(
        resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0
    )

    blurred_image = cv2.GaussianBlur(padded_image, (5, 5), 0)
    edges = cv2.Canny(blurred_image, 50, 150)

    _, thresholded = cv2.threshold(image, 200, 255, cv2.THRESH_TOZERO_INV) 

    return padded_image, edges

# Hough Pupil detection with intensity-based filtering
def detect_pupil(image, edges, min_radius, max_radius, param2=25):
    circles = cv2.HoughCircles(
        edges,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=50,
        param1=50,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius
    )

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")  # Convert to integers
        best_circle = None
        min_intensity = float('inf')

        for (x, y, r) in circles:
            roi = image[max(0, y-r):min(image.shape[0], y+r), max(0, x-r):min(image.shape[1], x+r)]
            if roi.size > 0:
                intensity = np.mean(roi)
                if intensity < min_intensity:
                    min_intensity = intensity
                    best_circle = (x, y, r)

        return best_circle  
    return None

# Hough Iris detection with ROI-based Hough Transform
def detect_iris_using_pupil(image, edges, pupil_circle, roi_size=200, param2=10):
    if pupil_circle is None:
        return None 

    x0, y0, pupil_radius = pupil_circle  

    x_start = max(0, x0 - roi_size)
    y_start = max(0, y0 - roi_size)
    roi_edges = edges[y_start:y_start + 2 * roi_size, x_start:x_start + 2 * roi_size]

    if roi_edges.size == 0:
        return None

    circles = cv2.HoughCircles(
        roi_edges,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=10,  
        param1=20,
        param2=param2,
        minRadius=pupil_radius + 10,
        maxRadius=pupil_radius + 50 
    )

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        circles[:, 0] += x_start
        circles[:, 1] += y_start

        iris = min(circles, key=lambda c: abs(c[2] - (pupil_radius + 30)))
        iris[0], iris[1] = x0, y0
        return iris
    return None

# Daugman's Integro-Differential Operator
def daugman_operator(image, x0, y0, r_min, r_max, step=1):
    if len(image.shape) == 3: 
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else: 
        gray = image

    blurred = cv2.GaussianBlur(gray, (3, 3), 0.5)  

    best_r = r_min
    max_diff = -float('inf')

    for r in range(r_min, r_max + 1, step):

        theta = np.linspace(0, 2 * np.pi, 200)
        x_circle = np.clip(x0 + r * np.cos(theta), 0, gray.shape[1] - 1).astype(int)
        y_circle = np.clip(y0 + r * np.sin(theta), 0, gray.shape[0] - 1).astype(int)

        intensities = blurred[y_circle, x_circle]

        inner_r = max(r - 2, r_min)  
        outer_r = min(r + 2, r_max) 

        x_inner = np.clip(x0 + inner_r * np.cos(theta), 0, gray.shape[1] - 1).astype(int)
        y_inner = np.clip(y0 + inner_r * np.sin(theta), 0, gray.shape[0] - 1).astype(int)
        x_outer = np.clip(x0 + outer_r * np.cos(theta), 0, gray.shape[1] - 1).astype(int)
        y_outer = np.clip(y0 + outer_r * np.sin(theta), 0, gray.shape[0] - 1).astype(int)

        inner_intensities = blurred[y_inner, x_inner]
        outer_intensities = blurred[y_outer, x_outer]

        diff = np.mean((outer_intensities - inner_intensities) ** 2) 

        if diff > max_diff:
            max_diff = diff
            best_r = r

    return best_r

# Polar coordinate transformation
def iris_to_polar(image, center, inner_radius, outer_radius):
    angles = np.linspace(0, 2 * np.pi, num=360) 
    radii = np.linspace(inner_radius, outer_radius, num=100)  
    polar_iris = np.zeros((len(radii), len(angles)), dtype=image.dtype) 

    for r_idx, radius in enumerate(radii):
        for a_idx, angle in enumerate(angles):
            x = int(center[0] + radius * np.cos(angle))
            y = int(center[1] + radius * np.sin(angle))
            if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                polar_iris[r_idx, a_idx] = image[y, x]

    return polar_iris

# Display results with superimposed circles and polar coordinates
def display_results(original, edges, pupil_circle, iris_circle_hough, iris_circle_daugman):
    plt.figure(figsize=(18, 12))

    # Original Image
    plt.subplot(2, 3, 1)
    plt.imshow(original, cmap='gray')
    plt.title("Original Image")

    # Edge Detection
    plt.subplot(2, 3, 2)
    plt.imshow(edges, cmap='gray')
    plt.title("Edge Detection")

    # Hough Transform Superimposition
    hough_result = original.copy()
    if pupil_circle is not None:
        cv2.circle(hough_result, (pupil_circle[0], pupil_circle[1]), pupil_circle[2], (255, 0, 0), 2)  
    if iris_circle_hough is not None:
        cv2.circle(hough_result, (iris_circle_hough[0], iris_circle_hough[1]), iris_circle_hough[2], (0, 255, 0), 2) 
    plt.subplot(2, 3, 3)
    plt.imshow(hough_result, cmap='gray')
    plt.title("Hough Transform")

    # Daugman's Method Superimposition
    daugman_result = original.copy()
    if pupil_circle is not None:
        cv2.circle(daugman_result, (pupil_circle[0], pupil_circle[1]), pupil_circle[2], (255, 0, 0), 2)  
    if iris_circle_daugman is not None:
        cv2.circle(daugman_result, (iris_circle_daugman[0], iris_circle_daugman[1]), iris_circle_daugman[2], (0, 0, 255), 2)  
    plt.subplot(2, 3, 4)
    plt.imshow(daugman_result, cmap='gray')
    plt.title("Daugman's Method")

    # Hough Transform Polar Coordinates (Unwrapped Iris)
    if iris_circle_hough is not None:
        polar_hough = iris_to_polar(
            original,
            (iris_circle_hough[0], iris_circle_hough[1]),
            pupil_circle[2], 
            iris_circle_hough[2]  
        )
        plt.subplot(2, 3, 5)
        plt.imshow(polar_hough, cmap='gray')
        plt.title("Hough Polar Coordinates")

    # Daugman's Method Polar Coordinates (Unwrapped Iris)
    if iris_circle_daugman is not None:
        polar_daugman = iris_to_polar(
            original,
            (iris_circle_daugman[0], iris_circle_daugman[1]),
            pupil_circle[2], 
            iris_circle_daugman[2] 
        )
        plt.subplot(2, 3, 6)
        plt.imshow(polar_daugman, cmap='gray')
        plt.title("Daugman's Polar Coordinates")

    plt.tight_layout()
    plt.show()


def main():
    for i in range(1, 7):  
        image_path = f'iris-detection/iris/iris{i}.jpg'
        original, edges = preprocess_image(image_path)

        pupil_circle = detect_pupil(original, edges, min_radius=5, max_radius=30, param2=25)

        iris_circle_hough = detect_iris_using_pupil(original, edges, pupil_circle, roi_size=200, param2=10)

        iris_circle_daugman = None
        if pupil_circle is not None:
            iris_radius_daugman = daugman_operator(original, pupil_circle[0], pupil_circle[1], 
                                                   pupil_circle[2] + 10, pupil_circle[2] + 50)
            iris_circle_daugman = (pupil_circle[0], pupil_circle[1], iris_radius_daugman)

        if pupil_circle is not None and iris_circle_hough is not None and iris_circle_daugman is not None:
            display_results(original.copy(), edges, pupil_circle, iris_circle_hough, iris_circle_daugman)
        else:
            print(f"Pupil or Iris detection failed for {image_path}")

main()
