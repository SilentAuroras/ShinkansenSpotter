# Shinkansen Spotter

Learning project to utilize YOLO to detect train models from video or images. This will focus on the Shinkansen
or High-speed EMUs to limit training data set preparation.

Train pictures and model information have been obtained from Wikipedia at the following [URL](https://en.wikipedia.org/wiki/Shinkansen). Attribution for each of these images from Wikipedia can be found in the following file: [Attribution](attribution.csv).

The following Shinkansen models have been added to the dataset currently. 
- N700

Training will begin automatically if no existing model has been custom trained (shinkansen.pt). Any images in the raw/ folder will be resized using CV2 to resize to 640x640. Training for the model was performed on a Nvidia 5070 but toggle exists to enable CPU training.

### Prerequisites
1. Install YOLO from ultralytics
```bash
conda install -c conda-forge ultralytics
```

2. Install CUDA Toolkit for GPU
```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

### Steps
1. Check help
```bash
python3 main.py -h
```

2. Check image for detection (image can be a local image or a url)
```bash
python3 main.py -i <IMAGE.png> 
```

3. Check video for detection (video can be local video or a URL)
```bash
python3 main.py -i <IMAGE.png> 
```