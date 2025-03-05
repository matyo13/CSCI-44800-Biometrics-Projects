import cv2
import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

def get_user_input(prompt):
    while True:
        try:
            value = int(input(prompt))
            if value > 0:
                return value
            else:
                print("Please enter a positive integer.")
        except ValueError:
            print("Invalid input. Please enter a positive integer.")

def enhance_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Unable to load image from path: {image_path}")
        return None
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)

def detect_gap_lines(enhanced_image, window_width, step_size):
    num_windows = (enhanced_image.shape[1] - window_width) // step_size + 1
    gap_lines = []

    for i in range(num_windows):
        start_x = i * step_size
        end_x = start_x + window_width
        window = enhanced_image[:, start_x:end_x]

        vertical_profile = np.sum(window, axis=1)
        smoothed_profile = gaussian_filter1d(vertical_profile, sigma=5)

        minima, _ = find_peaks(-smoothed_profile, distance=10)

        if minima.size > 0:
            significant_minima = minima[np.argmin(smoothed_profile[minima])]
            gap_lines.append((start_x + window_width // 2, significant_minima))

    return gap_lines

def plot_gap_lines(enhanced_image, gap_lines):
    plt.figure(figsize=(10, 10))
    plt.imshow(enhanced_image, cmap='gray')
    for x, _ in gap_lines:
        plt.axvline(x=x, color='b', linestyle='--', linewidth=2)

def fit_and_plot_smooth_curve(gap_lines, enhanced_image):
    gap_x, gap_y = zip(*gap_lines)
    spline = UnivariateSpline(gap_x, gap_y, s=1)
    smooth_curve_x = np.arange(0, enhanced_image.shape[1])
    smooth_curve_y = spline(smooth_curve_x)
    plt.plot(smooth_curve_x, smooth_curve_y, 'g-', linewidth=2, label='Smooth Curve')
    plt.title('Gap Valley Detection with Smooth Curve')
    plt.legend()
    return spline

def calculate_window_means(enhanced_image, spline, window_height):
    upper_window_means = []
    lower_window_means = []

    for x in range(enhanced_image.shape[1]):
        y = int(spline(x))
        
        upper_window_start = max(0, y - window_height // 2)
        upper_window_end = min(enhanced_image.shape[0], y + window_height // 2)
        upper_window = enhanced_image[0:upper_window_start, x]
        upper_window_mean = np.mean(upper_window)
        upper_window_means.append(upper_window_mean)

        lower_window_start = min(enhanced_image.shape[0], y + window_height // 2)
        lower_window_end = enhanced_image.shape[0]
        lower_window = enhanced_image[lower_window_start:lower_window_end, x]
        lower_window_mean = np.mean(lower_window)
        lower_window_means.append(lower_window_mean)

    return upper_window_means, lower_window_means

def detect_and_plot_minima(smoothed_means, num_teeth, color, label):
    minima, _ = find_peaks(-np.array(smoothed_means), distance=10)
    valid_minima = []
    for minimum in minima:
        if minimum > 1 and minimum < len(smoothed_means) - 2:
            if smoothed_means[minimum - 1] > smoothed_means[minimum] < smoothed_means[minimum + 1]:
                if smoothed_means[minimum - 2] > smoothed_means[minimum - 1] and smoothed_means[minimum + 1] < smoothed_means[minimum + 2]:
                    valid_minima.append(minimum)
    if valid_minima:
        sorted_minima = np.array(valid_minima)[np.argsort(np.array(smoothed_means)[valid_minima])]
        lowest_minima = sorted_minima[:num_teeth]
        plt.plot(lowest_minima, np.array(smoothed_means)[lowest_minima], color + 'o', label=label)
    return lowest_minima

def plot_intensity_maps(smoothed_upper_window_means, smoothed_lower_window_means):
    plt.figure(figsize=(10, 5))
    plt.plot(smoothed_upper_window_means, label='Upper Window Mean Intensity', color='r')
    plt.plot(smoothed_lower_window_means, label='Lower Window Mean Intensity', color='b')
    plt.title('Average Intensity Maps for Upper and Lower Windows')
    plt.xlabel('X-axis (Width of Image)')
    plt.ylabel('Average Intensity')
    plt.legend()

def perpendicular_slope(spline, x):
    tangent_slope = spline.derivative()(x)
    if tangent_slope != 0:
        return -1 / tangent_slope
    else:
        return float('inf')

def find_intersection(spline, x, y, perp_slope):
    def f(xi):
        return spline(xi) - (y + perp_slope * (xi - x))
    return f

def plot_minima_and_perpendicular_lines(enhanced_image, spline, lowest_upper_minima, lowest_lower_minima, window_height):
    plt.figure(figsize=(10, 10))
    plt.imshow(enhanced_image, cmap='gray')
    smooth_curve_x = np.arange(0, enhanced_image.shape[1])
    smooth_curve_y = spline(smooth_curve_x)
    plt.plot(smooth_curve_x, smooth_curve_y, 'g-', linewidth=2, label='Smooth Curve')

    for x in lowest_upper_minima:
        y = int(spline(x))
        upper_window_start = max(0, y - window_height // 2)
        upper_window_end = min(enhanced_image.shape[0], y + window_height // 2)
        upper_window = enhanced_image[0:upper_window_start, x]
        upper_window_mean = np.mean(upper_window)
        plt.plot(x, upper_window_mean, 'ro')

        perp_slope = perpendicular_slope(spline, x)
        intersection_x = fsolve(find_intersection(spline, x, upper_window_mean, perp_slope), x)[0]
        intersection_y = spline(intersection_x)
        end_y = 0
        end_x = (end_y - intersection_y) / perp_slope + intersection_x
        plt.plot([intersection_x, end_x], [intersection_y, end_y], 'r-')

    for x in lowest_lower_minima:
        y = int(spline(x))
        lower_window_start = min(enhanced_image.shape[0], y + window_height // 2)
        lower_window_end = enhanced_image.shape[0]
        lower_window = enhanced_image[lower_window_start:lower_window_end, x]
        lower_window_mean = np.mean(lower_window)
        flipped_y = 2 * y - lower_window_mean
        plt.plot(x, flipped_y, 'bo')

        perp_slope = perpendicular_slope(spline, x)
        intersection_x = fsolve(find_intersection(spline, x, flipped_y, perp_slope), x)[0]
        intersection_y = spline(intersection_x)
        end_y = enhanced_image.shape[0]
        end_x = (end_y - intersection_y) / perp_slope + intersection_x
        plt.plot([intersection_x, end_x], [intersection_y, end_y], 'b-')

    plt.title('Enhanced Image with Detected Gaps and Minima Points')
    plt.legend()

def main():
    image_path = '..\\dental-detection\\teeth_sample.png'
    enhanced_image = enhance_image(image_path)
    if enhanced_image is None:
        return

    window_width = get_user_input("Enter the width of the sliding window (Gap Valley Detection): ")
    step_size = window_width // 2
    gap_lines = detect_gap_lines(enhanced_image, window_width, step_size)
    plot_gap_lines(enhanced_image, gap_lines)
    spline = fit_and_plot_smooth_curve(gap_lines, enhanced_image)

    window_height = get_user_input("Enter the height of the sliding window (Tooth Isolation): ")
    upper_window_means, lower_window_means = calculate_window_means(enhanced_image, spline, window_height)

    num_upper_teeth = get_user_input("Enter the number of teeth on the upper side: ")
    num_lower_teeth = get_user_input("Enter the number of teeth on the lower side: ")

    smoothed_upper_window_means = gaussian_filter1d(upper_window_means, sigma=2)
    smoothed_lower_window_means = gaussian_filter1d(lower_window_means, sigma=2)

    plot_intensity_maps(smoothed_upper_window_means, smoothed_lower_window_means)

    lowest_upper_minima = detect_and_plot_minima(smoothed_upper_window_means, num_upper_teeth, 'r', 'Upper Minima')
    lowest_lower_minima = detect_and_plot_minima(smoothed_lower_window_means, num_lower_teeth, 'b', 'Lower Minima')
    plt.show()

    plot_minima_and_perpendicular_lines(enhanced_image, spline, lowest_upper_minima, lowest_lower_minima, window_height)
    plt.show()

if __name__ == "__main__":
    main()