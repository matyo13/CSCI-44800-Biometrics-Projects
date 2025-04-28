# Hand Shape Detection

## Description
This project focuses on detecting and classifying hand shapes using contour analysis and feature extraction techniques. The pipeline preprocesses hand images, detects hand contours, and classifies hand shapes based on extracted features.

## Features
- Preprocess hand images to enhance contrast and remove noise.
- Detect hand contours using OpenCV's contour detection methods.
- Extract features from detected contours for classification.
- Classify hand shapes using machine learning models.

## How to Run
1. Place the hand images in the `Hand-Shape-Detection/images/` folder.
2. Run the script:
   ```bash
   python main.py
   ```

## Dependencies
- Python 3.x
- OpenCV
- NumPy
- Scikit-learn
- Matplotlib (optional, for visualization)

Install the dependencies using:
```bash
pip install opencv-python-headless numpy scikit-learn matplotlib
```

## Key Steps in the Implementation
1. **Preprocessing**:
   - Convert images to grayscale.
   - Apply Gaussian blur to reduce noise.
   - Use thresholding to create binary images for contour detection.
2. **Contour Detection**:
   - Detect contours using OpenCV's `findContours` method.
   - Filter contours based on size and shape to isolate hand regions.
3. **Feature Extraction**:
   - Extract geometric features (e.g., area, perimeter, convexity) from detected contours.
   - Optionally, extract additional features such as Hu moments or Fourier descriptors.
4. **Classification**:
   - Train a machine learning model (e.g., SVM, Random Forest) on extracted features.
   - Classify hand shapes based on the trained model.
5. **Visualization**:
   - Display images with detected contours highlighted.
   - Show classification results with labels and confidence scores.

## Notes
- Ensure the input images are in a supported format (e.g., JPEG, PNG).
- Adjust the parameters in the script (e.g., contour thresholds) for optimal detection results.
- Experiment with different classification models for improved accuracy.

## Output
The script processes hand images and generates the following outputs:
- **Contour Visualizations**: Images with detected hand contours highlighted.
- **Classification Results**: Labels and confidence scores for detected hand shapes.
