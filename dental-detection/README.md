# Dental Detection

## Description
This project implements a dental detection system to identify and isolate individual teeth in dental images. The system uses sliding window techniques, intensity mapping, and spline functions to detect gaps between teeth and visualize the results.

## Features
- Detect gaps between upper and lower teeth using intensity mapping and local minima detection.
- Isolate individual teeth in both upper and lower sections.
- Visualize the results with plotted lines and smooth curves.
- Adjustable parameters for sliding window dimensions and expected number of teeth.

## How to Run
1. Place the dental images in the `dental-detection/images/` folder.
2. Run the script:
   ```bash
   python main.py
   ```

## Dependencies
- Python 3.x
- NumPy
- OpenCV
- Matplotlib
- Scipy

Install the dependencies using:
```bash
pip install numpy opencv-python matplotlib scipy
```

## Key Steps in the Implementation
1. **Gap Valley Detection**:
   - Use a vertical sliding window to create an intensity map.
   - Smooth the intensity map with a Gaussian filter.
   - Identify local minima to detect gaps between upper and lower teeth.
   - Fit a spline function to connect the detected points and form a smooth curve.
2. **Tooth Isolation**:
   - Divide the image into upper and lower sections using the smooth curve.
   - Apply sliding windows to each section to create intensity maps.
   - Smooth the maps and detect local minima based on user-defined expected number of teeth.
   - Mark gaps between individual teeth for both sections.
3. **Visualization**:
   - Plot lines from the smooth curve to the detected points, extending to the image edges.
   - Ensure lines are perpendicular to the curve for accurate visualization.
4. **Final Output**:
   - Generate a visualization where each tooth is boxed by plotted lines.
   - Allow user adjustments to optimize detection accuracy.

## Notes
- The accuracy of the results depends on the input values provided by the user.
- Adjusting the sliding window dimensions or the expected number of teeth may improve detection precision.
- Ensure the input images are clear and well-lit for optimal results.

## Output
The script processes dental images and generates the following outputs:
- **Gap Valley Visualization**: Smooth curves representing gaps between upper and lower teeth.
- **Tooth Isolation Visualization**: Lines marking individual teeth in both upper and lower sections.

## Ideal Input Values
- **Sliding window width**: 150  
- **Sliding window height**: 10  
- **Number of upper teeth**: 5  
- **Number of lower teeth**: 5  