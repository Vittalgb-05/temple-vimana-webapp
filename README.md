# Temple Vimana Classification App

A demonstration web app for classifying Ancient Temple Vimana styles.

## Features
- **Upload Interface**: Drag & Drop image upload.
- **Visual Pipeline**: Shows Segmentation Mask, Cropped Vimana, and Grad-CAM Heatmap.
- **Probabilities**: Breakdown of confidence for Nagara, Dravida, and Vesara.
- **Dockerized**: Easy to run.

## Setup & Running

### Prerequisites
- Python 3.10+
- Checkpoint files in `checkpoints/`:
    - `seg_best.pth` (Segmentation Model)
    - `clf_best.pth` (Classification Model)

### Running Locally
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the app:
   ```bash
   python src/app.py
   ```
3. Open browser at `http://localhost:5000`.

### Running with Docker
1. Build image:
   ```bash
   docker build -t temple-webapp .
   ```
2. Run container:
   ```bash
   docker run -p 5000:5000 --gpus all temple-webapp
   ```

## Troubleshooting
- **Missing Checkpoints**: The app will print a warning and use uninitialized models (random predictions). Ensure `.pth` files are in `checkpoints/`.
- **Import Errors**: Ensure you have `albumentations`, `segmentation-models-pytorch` installed.
