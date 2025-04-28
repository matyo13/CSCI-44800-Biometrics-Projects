# Iris Detection

## Description
This project implements iris detection using a combination of image processing techniques, including Hough Transform and Daugman's Integro-Differential Operator. The algorithm detects the pupil and iris boundaries and transforms the iris region into polar coordinates for further analysis.

## Features
- Load and preprocess images for iris detection.
- Detect the pupil using the Hough Transform with intensity-based filtering.
- Detect the iris boundary using:
  - **Hough Transform**: Detects circular regions around the pupil.
  - **Daugman's Integro-Differential Operator**: Optimizes the iris boundary based on intensity differences.
- Transform the detected iris region into polar coordinates.
- Visualize the original image, edge detection results, detected circles, and unwrapped iris regions.

## How to Run
1. Place the iris images in the `iris-detection/iris/` folder. The images should be named `iris1.jpg`, `iris2.jpg`, ..., `irisN.jpg`.
2. Run the script:
   ```bash
   python main.py
   ```

## Dependencies
- Python 3.x
- OpenCV
- NumPy
- Matplotlib

Install the dependencies using:
```bash
pip install opencv-python-headless numpy matplotlib
```

## Key Steps in the Implementation
1. **Preprocessing**: 
   - Convert images to grayscale.
   - Resize and pad images to a standard size.
   - Apply Gaussian blur and edge detection.
2. **Pupil Detection**:
   - Use the Hough Transform to detect circular regions corresponding to the pupil.
   - Filter detected circles based on intensity to find the best match.
3. **Iris Detection**:
   - Detect the iris boundary using:
     - **Hough Transform**: Detect circular regions around the pupil.
     - **Daugman's Integro-Differential Operator**: Optimize the iris boundary based on intensity differences.
4. **Polar Coordinate Transformation**:
   - Convert the detected iris region into polar coordinates for unwrapping and further analysis.
5. **Visualization**:
   - Display the original image, edge detection results, detected circles for the pupil and iris, and unwrapped iris regions in polar coordinates.

## Notes
- The script processes multiple images sequentially. If pupil or iris detection fails for an image, an error message will be printed in the console.
- Ensure the images are clear and well-lit for optimal detection results.

## Output
The script generates visualizations for each image, including:
- The original image with detected circles for the pupil and iris.
- Edge detection results.
- Unwrapped iris region in polar coordinates for both Hough Transform and Daugman's method.
