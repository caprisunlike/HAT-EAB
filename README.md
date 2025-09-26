# HAT-EAB
This project is based on [HAT](https://arxiv.org/abs/2205.04437).
- Hybrid Attention Block(HAB) with EAB
  <img src = "https://github.com/user-attachments/assets/2962d74c-6fe1-4f8d-a2b2-79d61813a41d" width="90%">
- Edge Attention Block(EAB)
  <p align="left">
    <img src = "https://github.com/user-attachments/assets/af50fa47-9275-45bf-af31-31180fdaa9b1" width="70%">
    <img src = "https://github.com/user-attachments/assets/9d642292-aaa3-42e6-ac7a-71f8c09e09e2" width="25%">
  </p>
  

## Updates
- :white_check_mark: Code that uses the edge map at inference

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
| HAT | 31.0268 | 0.8187 | 0.7506 |
| HAT-EAB | 30.8868 | 0.8154 | 0.7454 |
| HAT-EAB(canny) | 30.8771 | 0.8157 | 0.7457 |
| HAT-edgeloss | 31.0149 | 0.8189 | 0.7511 |
| HAT-edgeloss(canny) | 31.0129 | 0.8193 | 0.7520 |
| HAT-EAB+edgeloss | 30.8594 | 0.8150 | 0.7449 |
| HAT-EAB+edgeloss(canny) | 30.7528 | 0.8146 | 0.7443 |

<img src = "https://github.com/user-attachments/assets/2db8c678-1572-4911-9d24-18807e211770" width = "80%">
<img src = "https://github.com/user-attachments/assets/2a0f6d6d-a040-4d8d-82dd-d2705d8455d1">
<img src = "https://github.com/user-attachments/assets/40af2375-2cb1-4516-ad30-abe250ee797a" width = "90%">


## Reference
- HAT model by ["Activating More Pixels in Image Super-Resolution Transformer" (2023), Chen, Xiangyu, et al](https://arxiv.org/abs/2205.04437)
