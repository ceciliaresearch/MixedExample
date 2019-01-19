# Improved Mixed-Example Data Augmentation

This repository provides the code for our paper, *Improved Mixed-Example Data Augmentation*.

Code has been tested with TensorFlow version 1.12.

## Usage
First, make sure CIFAR-10 or CIFAR-100  has been downloaded  and extracted to `cifar10_data/cifar-10-batches-bin` or `cifar100_data/cifar-100-binary`.

After that, basic usage is as follows:

```
python train.py --mixed_example_method=vh_mixup --model_dir=cifar10_models/vh_mixup --dataset=cifar10 --weight_decay=1e-4
```

## Notes
On CIFAR-10, use a value of 1e-4 for weight decay for all mixed-example methods, and a value of 5e-4 for the baseline. On CIFAR-100, use a value of 5e-4 for all methods.

Due to the large maximum learning rate used by the default ResNet, training may be unstable for some methods. We recommend lowering the maximum learning rate (in `cifar_model_fn`) from `0.1 ` to `0.75` or, as done in the paper, running multiple copies for a few epochs, and only continuing the ones that didn't exhibit instability. Specifically, for the paper we ran 20 copies of each method for 3 epochs, continuing the 3 models with the lowest training loss.


## Citation
If you use this code, please cite our paper:

Improved Mixed-Example Data Augmentation.
Cecilia Summers and Michael J. Dinneen.
*IEEE Winter Conference on Applications of Computer Vision (WACV)*, 2019.


BibTeX:
```
@inproceedings{summers2019improved,
  title={Improved Mixed-Example Data Augmentation},
  author={Summers, Cecilia and Dinneen, Michael J},
  booktitle={Applications of Computer Vision (WACV), 2019 IEEE Winter Conference on},
  organization={IEEE}
  year={2019}
}
```
