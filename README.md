# Extract's fork of "Segment Anything Model for Road Network Graph Extraction"

## Demos
Predicted road network graph in a large region (2km x 2km).
![sam_road_cover](imgs/sam_road_cover.png)

Predicted road network graphs and corresponding masks in dense urban with complex and irregular structures.
![sam_road_mask_and_graph](imgs/sam_road_mask_and_graph.png)

## Installation
You will need the following:
- Nvidia GPU with latest CUDA and driver
- The latest pytorch
- PyTorch Lightning
- Weights and Biases (wandb) for logging
- Go, just for the APLS metric (we should really re-write this with pure python when time allows).

## Getting Started

### Environment Setup 
Initialise the uv environment:
```
uv sync
```

Pre-commit setup:
```bash
uvx pre-commit install
```


Create a `google_creds.json` using the [creds stored in google drive](https://drive.google.com/file/d/1e6AJ6tSdS1dVmAF4KuOcWVLpn5_opASH/view?usp=drive_link).


### SAM Preparation
Download the ViT-B checkpoint from the official SAM directory. Put it under:

```  
sam_road  
-sam_ckpts  
--sam_vit_b_01ec64.pth  
```

### Data Preparation
Refer to the instructions in the RNGDet++ repo (https://github.com/TonyXuQAQ/RNGDetPlusPlus) to download City-scale and SpaceNet datasets.

Put them in the main directory, structure like: 

``` 
-sam_road  
--cityscale  
---20cities  
--spacenet  
---RGB_1.0_meter  
```

Download links copied from https://github.com/TonyXuQAQ/RNGDetPlusPlus

#### SpaceNet
https://drive.google.com/uc?id=1FiZVkEEEVir_iUJpEH5NQunrtlG0Ff1W

The `data_split.json` is copied from the `dataset.json` in this folder.

#### CityScale
https://drive.google.com/uc?id=1R8sI1RmFe3rUfWMQaOfsYlBDHpQxFH-H

Find the `20cities` folder under this folder.

Then, run `python generate_labes.py` under both dirs.

### Training
City-scale dataset:
```
python train.py --config=config/toponet_vitb_512_cityscale.yaml  
```

SpaceNet dataset:
```  
python train.py --config=config/toponet_vitb_256_spacenet.yaml  
```

You can find the checkpoints under `lightning_logs` dir.

### Inference
```
python inferencer.py --config=path_to_the_same_config_for_training --checkpoint=path_to_ckpt 
``` 
This saves the inference results and visualizations.

Inferencing with our checkpoints:

Cityscale:
```
python inferencer.py --config=config/toponet_vitb_512_cityscale.yaml --checkpoint=/path_to/cityscale_vitb_512_e10.ckpt
```

Spacenet:
```
python inferencer.py --config=config/toponet_vitb_256_spacenet.yaml --checkpoint=/path_to/spacenet_vitb_256_e10.ckpt
```

### Test
Go to cityscale_metrics or spacenet_metrics, and run  
`bash eval_schedule.bash`.


