# Class-wise Anomaly Video Detection 

## Initial Setting
1. git clone `this repository`
2. conda env create -f vidan.yaml && conda activate vidan
3. gdown https://drive.google.com/u/0/uc?id=18nlV4YjPM93o-SdnPQrvauMN_v-oizmZ && unzip UCF_and_Shanghai.zip -d /DATA

## Datasets
* Directory tree
 ```
    DATA/
        UCF-Crime/ 
            ../all_rgbs
                ../*.npy
            ../all_flows
                ../*.npy
        train_anomaly.txt
        train_normal.txt
        test_anomaly.txt
        test_normal.txt
        
```

## train and test
```
python -u train.py \
  --mode ace \
  --classification information \
  --optimizer AdamW \
  --lr 0.0025 \
  --epochs 20 \
  --classes 13
```

## Result

| METHOD | DATASET | AUC (%) | 
|:--------:|:--------:|:--------:|
| Baseline | UCF-Crimes | 84.45 |
| Baseline + AMC | UCF-Crimes | 85.16 (+0.72) |
| Baseline + AMC + ACE (ours) | UCF-Crimes | 85.36 (+0.92) |

## AUC plot
![AUC plot](./result.png)

## Live Demo

![Demo](./sam.gif)

## Acknowledgment


The baseline model code is based on the pytorch re-implementation of "Real-world Anomaly Detection in Surveillance Videos" from [github](https://github.com/seominseok0429/Real-world-Anomaly-Detection-in-Surveillance-Videos-pytorch).

All the other codes excluding baseline model codes are implemented on our own.
