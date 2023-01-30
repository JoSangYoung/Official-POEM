# POEM:Polarization of Embeddings for Domain-Invariant Representations (AAAI'23)


## Paper
---
["POEM:Polarization of Embeddings for Domain-Invariant Representations"]()

- ### Architecture of POEM
![alt text](https://github.com/JoSangYoung/Official-POEM/blob/main/resources/Architecture_of_POEM.PNG?raw=true)


- ### Performance of POEM
![alt text](https://github.com/JoSangYoung/Official-POEM/blob/main/resources/Performance.PNG?raw=true)

## Preparation
---
### Dependencies

```
numpy==1.23.1
pandas==1.5.2
Pillow==9.2.0
torch==1.8.1+cu111
torch-scatter==2.1.0
torchaudio==0.8.1
torchvision==0.9.1+cu111
```

### Datasets
```
mv POEM
python -m domainbed.scripts.download --data_dir=/my/datasets/path
```

## How to Run
---
```
CUDA_VISIBLE_DEVICES=0 python train_all.py PACS_01_POEM --dataset PACS --deterministic --trial_seed 'Your random seed' --data_dir ‘YOUR_DATASET_PATH’ --batch_size 32 --algorithm ERM --swad True --bool_angle True --bool_task True

CUDA_VISIBLE_DEVICES=0 python train_all.py VLCS_01_POEM --dataset VLCS --deterministic --trial_seed 'Your random seed' --data_dir ‘YOUR_DATASET_PATH’ --batch_size 32 --algorithm ERM --swad True --bool_angle True --bool_task True

CUDA_VISIBLE_DEVICES=0 python train_all.py OfficeHome_01_POEM --dataset OfficeHome --deterministic --trial_seed 'Your random seed' --data_dir ‘YOUR_DATASET_PATH’ --batch_size 32 --algorithm ERM --swad True --bool_angle True --bool_task True

CUDA_VISIBLE_DEVICES=0 python train_all.py TerraIncognita_01_POEM --dataset TerraIncognita --deterministic --trial_seed 'Your random seed' --data_dir ‘YOUR_DATASET_PATH’ --batch_size 32 --algorithm ERM --swad True --bool_angle True -- bool_task True

CUDA_VISIBLE_DEVICES=0 python train_all.py DomainNet_01_POEM --dataset DomainNet --deterministic --trial_seed 'Your random seed' --data_dir ‘YOUR_DATASET_PATH’ --batch_size 32 --algorithm ERM --swad True --bool_angle True --bool_task True
```


```
CUDA_VISIBLE_DEVICES=0 python train_all.py PACS_01_POEM --dataset PACS --deterministic --trial_seed 'Your random seed' --data_dir ‘YOUR_DATASET_PATH’ --batch_size 32 --algorithm MIRO --swad True --bool_angle True --bool_task True

CUDA_VISIBLE_DEVICES=0 python train_all.py VLCS_01_POEM --dataset VLCS --deterministic --trial_seed 'Your random seed' --data_dir ‘YOUR_DATASET_PATH’ --batch_size 32 --algorithm MIRO --swad True --bool_angle True --bool_task True

CUDA_VISIBLE_DEVICES=0 python train_all.py OfficeHome_01_POEM --dataset OfficeHome --deterministic --trial_seed 'Your random seed' --data_dir ‘YOUR_DATASET_PATH’ --batch_size 32 --algorithm MIRO --swad True --bool_angle True --bool_task True

CUDA_VISIBLE_DEVICES=0 python train_all.py TerraIncognita_01_POEM --dataset TerraIncognita --deterministic --trial_seed 'Your random seed' --data_dir ‘YOUR_DATASET_PATH’ --batch_size 32 --algorithm MIRO --swad True --bool_angle True -- bool_task True

CUDA_VISIBLE_DEVICES=0 python train_all.py DomainNet_01_POEM --dataset DomainNet --deterministic --trial_seed 'Your random seed' --data_dir ‘YOUR_DATASET_PATH’ --batch_size 32 --algorithm MIRO --swad True --bool_angle True --bool_task True
```