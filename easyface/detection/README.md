# Face and Landmark Detection

## Pretrained Models

* [RetinaFace-MobileNet0.25](https://drive.google.com/file/d/1g58Ju47vBbjzSunh2aI9_1gf90jKCbGq/view?usp=sharing)

> All models are trained on WiderFace dataset and weights are from official repositories.

## WiderFace Benchmark

Method | Model | Easy | Medium | Hard | Params(M) | GFLOPs 
--- | --- | --- | --- | --- | --- | ---
RetinaFace | MobileNet0.25 | 87.78 | 81.16 | 47.32 | 0.44 | 0.8
| | R50 | 94.92 | 91.90 | 64.17 | 30 | 38
SCRFD | 0.5GF | 90.57 | 88.12 | 68.51 | 0.57 | 0.5
| | 2.5GF | 93.78 | 92.16 | 77.87 | 0.67 | 2.5
| | 10GF | 95.16 | 93.87 | 83.05 | 4 | 10
| | 34GF | 96.06 | 94.92 | 85.29 | 10 | 34
YOLO5Face | YOLOv5n-0.5 (ShuffleNetv2) | 90.76 | 88.12 | 73.82 | 0.45 | 0.6
| | YOLOv5n (ShuffleNetv2) | 93.61 | 91.52 | 80.53 | 1.73 | 2.1
| | YOLOv5s | 94.33 | 92.61 | 83.15 | 7 | 6
| | YOLOv5m | 95.30 | 93.76 | 85.28 | 21 | 18


## WiderFace Dataset Preparation

* 32,303 images
* 393,703 faces
* Contains large variations in scale, pose, expression, occlusion and illumination
* 40% training 10% validation, 50% testing
* 61 scene categories