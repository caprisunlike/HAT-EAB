# HAT-EAB
## Updates
- :white_check_mark: Code that uses the edge map at inference

## How To Test
```
python -m hat.test_hat_eab -opt options/test/HAT_SRx4_sam.yml
```
## How To Train
```
python -m hat.train_hat_eab -opt options/train/train_HAT_EAB_SRx4_from_scratch_sam.yml
```
## Reference
- HAT model by ["Activating More Pixels in Image Super-Resolution Transformer" (2023), Chen, Xiangyu, et al](https://arxiv.org/abs/2205.04437)
