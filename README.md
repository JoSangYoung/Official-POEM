# POEM:Polarization of Embeddings for Domain-Invariant Representations (AAAI'23)


Official PyTorch implementation of [POEM:Polarization of Embeddings for Domain-Invariant Representations.]()

### Architecture of POEM
![alt text](https://github.com/JoSangYoung/Official-POEM/blob/main/resources/Architecture_of_POEM.PNG?raw=true)


### Performance of POEM
![alt text](https://github.com/JoSangYoung/Official-POEM/blob/main/resources/Performance.PNG?raw=true)

## Preparation
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
cd POEM
python -m domainbed.scripts.download --data_dir=/my/datasets/path
```

## How to Run
---

ERM
```
cd POEM

CUDA_VISIBLE_DEVICES=0 python train_all.py PACS_01_POEM --dataset PACS --deterministic --trial_seed 'Your random seed' --data_dir ‘YOUR_DATASET_PATH’ --batch_size 32 --algorithm ERM

CUDA_VISIBLE_DEVICES=0 python train_all.py VLCS_01_POEM --dataset VLCS --deterministic --trial_seed 'Your random seed' --data_dir ‘YOUR_DATASET_PATH’ --batch_size 32 --algorithm ERM

CUDA_VISIBLE_DEVICES=0 python train_all.py OfficeHome_01_POEM --dataset OfficeHome --deterministic --trial_seed 'Your random seed' --data_dir ‘YOUR_DATASET_PATH’ --batch_size 32 --algorithm ERM

CUDA_VISIBLE_DEVICES=0 python train_all.py TerraIncognita_01_POEM --dataset TerraIncognita --deterministic --trial_seed 'Your random seed' --data_dir ‘YOUR_DATASET_PATH’ --batch_size 32 --algorithm ERM

CUDA_VISIBLE_DEVICES=0 python train_all.py DomainNet_01_POEM --dataset DomainNet --deterministic --trial_seed 'Your random seed' --data_dir ‘YOUR_DATASET_PATH’ --batch_size 32 --algorithm ERM
```

POEM
```
cd POEM

CUDA_VISIBLE_DEVICES=0 python train_all.py PACS_01_POEM --dataset PACS --deterministic --trial_seed 'Your random seed' --data_dir ‘YOUR_DATASET_PATH’ --batch_size 32 --algorithm ERM --bool_angle True --bool_task True

CUDA_VISIBLE_DEVICES=0 python train_all.py VLCS_01_POEM --dataset VLCS --deterministic --trial_seed 'Your random seed' --data_dir ‘YOUR_DATASET_PATH’ --batch_size 32 --algorithm ERM --bool_angle True --bool_task True

CUDA_VISIBLE_DEVICES=0 python train_all.py OfficeHome_01_POEM --dataset OfficeHome --deterministic --trial_seed 'Your random seed' --data_dir ‘YOUR_DATASET_PATH’ --batch_size 32 --algorithm ERM --bool_angle True --bool_task True

CUDA_VISIBLE_DEVICES=0 python train_all.py TerraIncognita_01_POEM --dataset TerraIncognita --deterministic --trial_seed 'Your random seed' --data_dir ‘YOUR_DATASET_PATH’ --batch_size 32 --algorithm ERM --bool_angle True -- bool_task True

CUDA_VISIBLE_DEVICES=0 python train_all.py DomainNet_01_POEM --dataset DomainNet --deterministic --trial_seed 'Your random seed' --data_dir ‘YOUR_DATASET_PATH’ --batch_size 32 --algorithm ERM  --bool_angle True --bool_task True
```

MIRO + POEM
```
cd miro_POEM

CUDA_VISIBLE_DEVICES=0 python train_all.py PACS_01_POEM --dataset PACS --deterministic --trial_seed 'Your random seed' --data_dir ‘YOUR_DATASET_PATH’ --batch_size 32 --algorithm MIRO  --bool_angle True --bool_task True

CUDA_VISIBLE_DEVICES=0 python train_all.py VLCS_01_POEM --dataset VLCS --deterministic --trial_seed 'Your random seed' --data_dir ‘YOUR_DATASET_PATH’ --batch_size 32 --algorithm MIRO --bool_angle True --bool_task True

CUDA_VISIBLE_DEVICES=0 python train_all.py OfficeHome_01_POEM --dataset OfficeHome --deterministic --trial_seed 'Your random seed' --data_dir ‘YOUR_DATASET_PATH’ --batch_size 32 --algorithm MIRO --bool_angle True --bool_task True

CUDA_VISIBLE_DEVICES=0 python train_all.py TerraIncognita_01_POEM --dataset TerraIncognita --deterministic --trial_seed 'Your random seed' --data_dir ‘YOUR_DATASET_PATH’ --batch_size 32 --algorithm MIRO --bool_angle True -- bool_task True

CUDA_VISIBLE_DEVICES=0 python train_all.py DomainNet_01_POEM --dataset DomainNet --deterministic --trial_seed 'Your random seed' --data_dir ‘YOUR_DATASET_PATH’ --batch_size 32 --algorithm MIRO --bool_angle True --bool_task True
```

SWAD + POEM
```
cd POEM

CUDA_VISIBLE_DEVICES=0 python train_all.py PACS_01_POEM --dataset PACS --deterministic --trial_seed 'Your random seed' --data_dir ‘YOUR_DATASET_PATH’ --batch_size 32 --algorithm ERM --bool_swad True --bool_angle True --bool_task True

CUDA_VISIBLE_DEVICES=0 python train_all.py VLCS_01_POEM --dataset VLCS --deterministic --trial_seed 'Your random seed' --data_dir ‘YOUR_DATASET_PATH’ --batch_size 32 --algorithm ERM --bool_swad True --bool_angle True --bool_task True

CUDA_VISIBLE_DEVICES=0 python train_all.py OfficeHome_01_POEM --dataset OfficeHome --deterministic --trial_seed 'Your random seed' --data_dir ‘YOUR_DATASET_PATH’ --batch_size 32 --algorithm ERM --bool_swad True --bool_angle True --bool_task True

CUDA_VISIBLE_DEVICES=0 python train_all.py TerraIncognita_01_POEM --dataset TerraIncognita --deterministic --trial_seed 'Your random seed' --data_dir ‘YOUR_DATASET_PATH’ --batch_size 32 --algorithm ERM --bool_swad True --bool_angle True -- bool_task True

CUDA_VISIBLE_DEVICES=0 python train_all.py DomainNet_01_POEM --dataset DomainNet --deterministic --trial_seed 'Your random seed' --data_dir ‘YOUR_DATASET_PATH’ --batch_size 32 --algorithm ERM --bool_swad True --bool_angle True --bool_task True
```

MIRO + SWAD + POEM
```
cd miro_POEM

CUDA_VISIBLE_DEVICES=0 python train_all.py PACS_01_POEM --dataset PACS --deterministic --trial_seed 'Your random seed' --data_dir ‘YOUR_DATASET_PATH’ --batch_size 32 --algorithm MIRO --bool_swad True --bool_angle True --bool_task True

CUDA_VISIBLE_DEVICES=0 python train_all.py VLCS_01_POEM --dataset VLCS --deterministic --trial_seed 'Your random seed' --data_dir ‘YOUR_DATASET_PATH’ --batch_size 32 --algorithm MIRO --bool_swad True --bool_angle True --bool_task True

CUDA_VISIBLE_DEVICES=0 python train_all.py OfficeHome_01_POEM --dataset OfficeHome --deterministic --trial_seed 'Your random seed' --data_dir ‘YOUR_DATASET_PATH’ --batch_size 32 --algorithm MIRO --bool_swad True --bool_angle True --bool_task True

CUDA_VISIBLE_DEVICES=0 python train_all.py TerraIncognita_01_POEM --dataset TerraIncognita --deterministic --trial_seed 'Your random seed' --data_dir ‘YOUR_DATASET_PATH’ --batch_size 32 --algorithm MIRO --bool_swad True --bool_angle True -- bool_task True

CUDA_VISIBLE_DEVICES=0 python train_all.py DomainNet_01_POEM --dataset DomainNet --deterministic --trial_seed 'Your random seed' --data_dir ‘YOUR_DATASET_PATH’ --batch_size 32 --algorithm MIRO --bool_swad True --bool_angle True --bool_task True
```

## Citation
```
@article{jo2023poem,
  title={POEM:Polarization of Embeddings for Domain-Invariant Representations},
  author={Sang-Yeong Jo and Sung Whan Yoon},
  journal={Association for the Advancement of Artificial Intelligence (AAAI)},
  year={2023}
}
```


## License
This source code is released under the MIT license.

This project includes some code from [DomainBed](https://github.com/facebookresearch/DomainBed), [SWAD](https://github.com/khanrc/swad), [MIRO](https://github.com/kakaobrain/miro), also MIT licensed.
