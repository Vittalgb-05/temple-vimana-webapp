# How to Train Real Models

The web app currently runs with **dummy models** (random weights) for demonstration purposes. To get accurate predictions, you must train the models on the actual dataset.

## 1. Get the Data
You need the **Ancient Temple Vimana Images Dataset** from Kaggle.
1. Download the dataset (e.g., `archive.zip`) from Kaggle.
2. Unzip it.
3. Place the images into `temple-vimana/data/raw/`.
   Structure should look like:
   ```
   temple-vimana/data/raw/
   ├── Nagara/
   ├── Dravida/
   ├── Vesara/
   ```
   (Or similar, the script `prepare_dataset.sh` handles basic discovery).

## 2. Prepare Data
Open a terminal in `temple-vimana/`:
```bash
cd ../temple-vimana
bash scripts/prepare_dataset.sh
```

## 3. Train Segmentation
```bash
python src/train.py --config configs/seg_unet_resnet34.yaml
```
This will create `checkpoints/segmentation_best.pth`.

## 4. Train Classification
First, generate the crops using the trained segmentation model:
```bash
# Infer masks on training data
python src/infer.py --input data/raw --seg_checkpoint checkpoints/segmentation_best.pth --output data/predicted_masks

# Build crops
python scripts/build_classifier_dataset.py --masks data/predicted_masks --images data/raw --out data/classifier_crops
```

Then train the classifier:
```bash
python src/train.py --config configs/classifier_densenet121.yaml
```
This will create `checkpoints/classification_best.pth`.

## 5. Update Web App
Copy the new checkpoints to the web app:
```bash
cp checkpoints/segmentation_best.pth ../temple-vimana-webapp/checkpoints/seg_best.pth
cp checkpoints/classification_best.pth ../temple-vimana-webapp/checkpoints/clf_best.pth
```

Now restart the web app!
