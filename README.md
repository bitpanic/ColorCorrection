# Color Correction Tool

This tool helps calibrate the color of one camera to match another by selecting corresponding color patches in both images and computing a color correction matrix.

## Requirements

- Python 3.6 or higher
- OpenCV
- NumPy

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Prepare your image pairs:
   - One image from the reference camera (source)
   - One image from the camera to calibrate (target)

2. Run the application:
```bash
python color_correction.py
```

3. Using the tool:
   - Draw ROIs (Regions of Interest) on both images by clicking and dragging
   - The average RGB values for each ROI will be displayed in the console
   - Press 'n' to process the current pair and move to the next
   - Press 'q' to quit

4. Output files:
   - `color_correction_matrix.npy`: The 3x3 color correction matrix
   - `roi_data.json`: Contains all ROI positions and their average color values

## How it works

1. For each image pair, select corresponding color patches in both images
2. The tool calculates average RGB values for each ROI
3. A 3x3 color correction matrix is computed using least squares regression
4. The matrix can be used to transform colors from the target camera to match the reference camera

## Notes

- Make sure to select the same number of ROIs in both images
- For best results, select ROIs that cover a wide range of colors
- The ROIs should be relatively uniform in color 