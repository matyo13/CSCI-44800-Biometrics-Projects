# Segmentation with K-means

## Description
This project implements K-means clustering to segment biometric images into distinct regions. The algorithm is applied repetitively to ensure robust clustering and to calculate the frequency of cluster assignments for each pixel.

## Features
- Load and resize images for processing.
- Randomly initialize centroids for K-means clustering.
- Perform K-means clustering using scikit-learn's `KMeans`.
- Reassign clusters based on reference colors for consistency.
- Calculate cluster assignment frequencies over multiple repetitions.
- Threshold probabilities to detect specific regions (e.g., skin regions).
- Visualize the original image, cluster probability maps, and detected regions.

## How to Run
1. Place the images to be segmented in the `segmentation-with-K-means` folder.
2. Update the `image_path1` and `image_path2` variables in `main.py` with the paths to your images.
3. Run the script:
   ```bash
   python main.py
   ```

## Dependencies
- Python 3.x
- NumPy
- OpenCV
- Matplotlib
- scikit-learn

Install the dependencies using:
```bash
pip install numpy opencv-python matplotlib scikit-learn
```

## Key Steps in the Implementation
1. **Load and Resize Image**: 
   - Images are loaded and resized to a specified scale for efficient processing.
2. **Random Initialization of Centroids**: 
   - Centroids are randomly selected from the image pixels.
3. **K-means Clustering**: 
   - The clustering is performed using scikit-learn's `KMeans` with the specified number of clusters (`k`).
4. **Reassign Clusters**: 
   - Clusters are reassigned based on reference colors for consistent labeling.
5. **Repetitive K-means**: 
   - The algorithm is executed multiple times to calculate the frequency of cluster assignments for each pixel.
6. **Thresholding**: 
   - Probabilities are thresholded to detect specific regions (e.g., skin regions).
7. **Visualization**: 
   - Results are visualized, including the original image, cluster probability maps, and detected regions.

## Notes
- The script processes two images (`HandImage1.jpg` and `HandImage2.jpg`) by default. You can add more images by modifying the script.
- The `k_values` variable defines the number of clusters to test (e.g., 2, 3, 5).
- The threshold for detecting regions can be adjusted in the `threshold_skin_regions` function.

## Output
The script generates visualizations for each image and each value of `k`, including:
- The original image.
- Cluster probability maps.
- Detected regions (e.g., skin regions).
