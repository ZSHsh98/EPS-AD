# VT-PyTorch

The PyTorch implementation **(non-official)** of paper [Enhancing the Transferability of Adversarial Attacks through Variance Tuning](https://arxiv.org/abs/2103.15571)



## Requirements

+ Python >= 3.6.5
+ PyTorch >= 1.7.1
+ Numpy >= 1.15.4
+ opencv >= 3.4.2
+ scipy > 1.1.0
+ pandas >= 1.0.1
+ imageio >= 2.6.1

## Qucik Start

### Prepare the data and models

You should collect your PyTorch models and place them in `./checkpoints`.    

Clean ImageNet images have already been placed in `./data`. 

### Variance Tuning Attack

All the provided codes generate adversarial examples on inception_v3 model. If you want to attack other models, modify "model_map_fp" in utils.py and the corresponding parser argument in vmi_di_ti_si_fgsm.py

#### Runing attack

Taking vmi_di_ti_si_fgsm attack for example, you can run this attack as following:

```
CUDA_VISIBLE_DEVICES=gpuid python vmi_di_ti_si_fgsm.py 
```

The generated adversarial examples would be stored in directory `./data/adv`. Then run the file `compare.py` to evaluate the success rate of each target model:

```
CUDA_VISIBLE_DEVICES=gpuid python compare.py
```

## Acknowledgments

Code refers to [VT](https://github.com/JHL-HUST/VT).

## References

Wang X, He K. Enhancing the Transferability of Adversarial Attacks through Variance Tuning[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021: 1924-1933.

## TODO

NI-FGSM attack (You can also implement by yourself, it is not so difficult)

## Contact

Questions and suggestions can be sent to 1634751580@qq.com