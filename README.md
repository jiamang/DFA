# Dynamic Local Feature Aggregation for Learning on Point Clouds
This repository contains PyTorch implementation for DFA : **Dynamic Local Feature Aggregation for Learning on Point Clouds.** 
Our code skeleton is borrowed from antao97/dgcnn.pytorch

Our DFA module is as follows:
![image](https://github.com/jiamang/DFA/blob/main/image/DFA.png)

We use DFA to complete the point cloud classification and segmentation task structure as follows:
![image](https://github.com/jiamang/DFA/blob/main/image/Architecture.png)

## Requirements
Python >= 3.7 , PyTorch >= 1.2  , CUDA >= 10.0  , Package: glob, h5py, sklearn, plyfile, torch_scatter


## Point Cloud Classification
Note: You can choose 1024 or 2048 points for training and evaluation.
### Run the training script:
``` 
python main_cls.py --exp_name=cls_1024 --num_points=1024 --k=20 
```
### Run the evaluation script after training finished:
``` 
python main_cls.py --exp_name=cls_1024_eval --num_points=1024 --k=20 --eval=True --model_path=outputs/cls_1024/models/model.t7
```

## Point Cloud Part Segmentation
Note: There are two options for training on the full dataset and for each class individually.
### Run the training script:
· Full dataset
```
python main_partseg.py --exp_name=partseg 
```
· With class choice, for example airplane
```
python main_partseg.py --exp_name=partseg_airplane --class_choice=airplane
```
### Run the evaluation script after training finished:
· Full dataset
```
python main_partseg.py --exp_name=partseg_eval --eval=True --model_path=outputs/partseg/models/model.t7
```
· With class choice, for example airplane
```
python main_partseg.py --exp_name=partseg_airplane_eval --class_choice=airplane --eval=True --model_path=outputs/partseg_airplane/models/model.t7
```

## Point Cloud Semantic Segmentation on the S3DIS Dataset
Note : This task uses 6-fold training.
### Run the training script:
```
python main_semseg_s3dis.py --exp_name=semseg_s3dis_5 --test_area=5 
```
### Run the evaluation script after training finished:
```
python main_semseg_s3dis.py --exp_name=semseg_s3dis_eval_5 --test_area=5 --eval=True --model_root=outputs/semseg_s3dis/models/
```
