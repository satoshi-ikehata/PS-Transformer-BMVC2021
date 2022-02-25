# PS-Transformer-BMVC2021

Satoshi Ikeahta. PS-Transformer: Learning Sparse Photometric Stereo Network using Self-Attention Mechanism, BMVC2021

 [paper](https://www.bmvc2021-virtualconference.com/assets/papers/0319.pdf)

 [supplementary](https://www.bmvc2021-virtualconference.com/assets/supp/0319_supp.zip)

## Getting Started

This is an official Pytorch implementation of PS-Transformer for estimating surface normals from images captured under different illumination.

### Prerequisites

- Python3.7.5
- torch
- tensorboard
- cv2
- scipy

Tested on:
- Ubuntu 20.04/Windows10, Python 3.7.5, Pytorch 1.6.0, CUDA 10.2
  - GPU: NvidiaQuadroRTX8000 (64GB)

### Running the tests
For testing network (with DiLiGenT dataset), please download [DiLiGenT dataset (DiLiGenT.zip)](https://sites.google.com/site/photometricstereodata/) by Boxin Shi [1] and extract it anywhere. 

```
DILIGENT = 'USER_PATH/DiLiGenT/pmsData'
```

Then, run main.py with the DiLiGenT path as an argument.

```
python main.py --diligent DILIGENT
```

```
diligent = 'USER_PATH/DiLiGenT/pmsData'
```

The pretrained model (our "full" configuration) is available in "pretrained" directory.

If the program properly works, you will get average angular errors (in degrees) for each dataset.

The final result [Mean] is the error about the averaged surface normal over normals predicted from K (K=10 in this case) differently rotated observation maps (See details in my paper). Finally, you will get the predicted surface normal map and the error map.

<img src="webimage/img001.png" width="600">

### Important notice about DiLiGenT datasets

As is commonly known, "bear" dataset in DiLiGenT has problem and the first 20 images in bearPNG is are skipped. 

### Running the test on othter datasets (Unsupported)
If you want to run this code on ohter dataset, please allocate your own data just in the same manner with DiLiGenT. The required files are
- images (.png format in default, but you can easily change the code for other formats)
- lights (light_directions.txt, light_intensities.txt)
- normals (normal.txt, if no ground truth surface normal is available, you can simply set all the values by zero)

### Running the training
The training script is NOT supported yet.
However, the training dataset is alraedy available. Please send a request to sikehata@nii.ac.jp

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## References
[1] Boxin Shi, Zhipeng Mo, Zhe Wu, Dinglong Duan, Sai-Kit Yeung, and Ping Tan, "A Benchmark Dataset and Evaluation for Non-Lambertian and Uncalibrated Photometric Stereo", In IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2018.
