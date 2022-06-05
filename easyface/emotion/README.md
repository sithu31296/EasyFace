# Facial Expression Recognition

## Datasets

### AffectNet

* AffectNet-7: 283,901 train set, 3,500 test set and 7 labels
* AffectNet-8: 287,568 train set, 4,000 test set and 8 labels 

### RAF-DB

* 12,271 train set and 3,068 test set
* 7 baisc and 11 compound emotion labels
* Faces are aligned and cropped to 100x100 size.

### SFEW 2.0

* Newest version of SFEW dataset
* Extracted from AFEW video database
* 7 expression categories
* 958 train set, 436 val set and 372 test set
* Light and compact
* Faces are not aligned.

## Pre-training

Most facial expression recognition models are pre-trained on MS-Celeb-1M dataset. Download the pretrained model from [here](https://drive.google.com/file/d/1H421M8mosIVt8KsEWQ1UuYMkQS8X1prf/view?usp=sharing).

## Benchmarks & Pretrained Models

Model | Backbone | AffectNet-8 | AffectNet-7 | RAF-DB | SFEW 2.0 | Params (M) | GLOFPs | Checkpoints
--- | --- | --- | --- | --- | --- | --- | --- | --- 
DAN | ResNet-18 | 62.09 | 65.69 | 89.70 | 57.88 | - | - | [AffectNet8][dan_a8] \| [AffectNet7][dan_a7] \| [RAFDB][dan_r]




[dan_a8]: https://drive.google.com/file/d/1uHNADViICyJEjJljv747nfvrGu12kjtu/view?usp=sharing
[dan_a7]: https://drive.google.com/file/d/1_Z-U7rT5NJ3Vc73aN2ZBmuvCkzUQG4jT/view?usp=sharing
[dan_r]: https://drive.google.com/file/d/1ASabP5wkLUIh4VQc8CEuZbZyLJEFaTMF/view?usp=sharing