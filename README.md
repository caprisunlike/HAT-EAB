# HAT-EAB
This project is based on [HAT](https://arxiv.org/abs/2205.04437).

<img src = "https://github.com/user-attachments/assets/8c1616fb-3676-4318-9eb8-c53a6044b98c" width="70%">

## Updates
- :white_check_mark: Code that uses the edge map at inference

## Environment
- python 3.11
- pytorch 2.4.0
- cuda 12.4.1

## Dataset
- AID dataset
  https://captain-whu.github.io/AID/

## How To Test
```
python -m hat.test_hat_eab -opt options/test/HAT_SRx4_sam.yml
```

## How To Train
```
python -m hat.train_hat_eab -opt options/train/train_HAT_EAB_SRx4_from_scratch_sam.yml
```

## Results
| Model | PSNR | SSIM | edge-SSIM |
| --- | --- | --- | --- |
| HAT | 31.2973 | 0.8246 | 0.7614 |
| HAT-EAB | 31.5073 | 0.8293 | 0.7709 |

<img src = "https://github.com/user-attachments/assets/11fb77c2-e805-4c2a-bd3d-570733e7c183">


## Reference
- HAT model by ["Activating More Pixels in Image Super-Resolution Transformer" (2023), Chen, Xiangyu, et al](https://arxiv.org/abs/2205.04437)
