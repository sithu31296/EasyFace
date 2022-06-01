# Face Recognition

## Pretrained Models

WebFace4M

* [AdaFace-IR18](https://drive.google.com/file/d/1p_Od0F5jQClG2b3nqkWv6JEzuZlF3ACh/view?usp=sharing)

MS1MV2

* ...

## Benchmarks

Evaluation on 5 High Quality Image Validation Sets (LFW, CFPFP, CPLFW, CALFW, AGEDB)

Model | Dataset | Method | LFW | CFPFP | CPLFW | CALFW | AGEDB | AVG
--- | --- | --- | --- | --- | --- | --- | --- | --- 
R18 | WebFace4M | AdaFace | 99.53 | 97.26 | 92.28 | 95.52 | 96.47 | 96.21
R50 | MS1MV2 | AdaFace | 99.82 | 97.86 | 92.83 | 96.07 | 97.85 | 96.88
R50 | WebFace4M | AdaFace | 99.78 | 98.97 | 94.17 | 95.98 | 97.78 | 97.34

Evaluation on Mixed Quality Scenario (IJBB, IJBC)

Model | Dataset | Method | IJBB TAR@FAR=0.01% | IJBC TAR@FAR=0.01%
--- | --- | --- | --- | ---
R18 | WebFace4M | AdaFace | 93.03 | 94.99
R50 | MS1MV2 | AdaFace | 94.82 | 96.27
R50 | WebFace4M | AdaFace | 95.44 | 96.98

Evaluation on Low Quality Scenario (IJBS)

| | | | Sur-to-Single | | | Sur-to-Book | | |
--- | --- | --- | --- | --- | --- | --- | --- | ---
Model | Pretrained Dataset | Method | Rank1 | Rank5 | 1% | Rank1 | Rank5 | 1%
R100 | MS1MV2 | AdaFace | 65.26 | 70.53 | 51.66 | 66.27 | 71.61 | 50.87
R100 | WebFace4M | AdaFace | 70.42 | 75.29 | 58.27 | 70.93 | 76.11 | 58.02


| | | | Sur-to-Sur | | | TinyFace | |
--- | --- | --- | --- | --- | --- | --- | --- 
Model | Pretrained Dataset | Method | Rank1 | Rank5 | 1% | Rank1 | Rank5 
R100 | MS1MV2 | AdaFace | 23.74 | 37.47 | 2.50 | 68.21 | 71.54
R100 | WebFace4M | AdaFace | 35.05 | 48.22 | 4.96 | 72.02 | 74.52

> * Sur-to-Single: Protocol comparing surveillance video (probe) to single enrollment image (gallery)
> * Sur-to-Book: Protocol comparing surveillance video (probe) to all enrollment images (gallery)
> * Sur-to-Sur: Protocol comparing surveillance video (probe) to surveillance video (gallery)


## Datasets Preparation

Coming soon.