# PS-Transformer-BMVC2021
This is an official Pytorch implementation of PS-Transformer [1] for estimating surface normals from images captured under different known directional lightings.

```
[1] S. Ikehata, "PS-Transformer: Learning Sparse Photometric Stereo Network using Self-Attention Mechanism", BMVC2021
```

[paper](https://arxiv.org/abs/2211.11386)

[supplementary](https://www.dropbox.com/s/zgvhkd24ke9t0xp/319supp.pdf?dl=0)

<p align="center">
<img src="fig/arc.jpg" width="800">  
<img src="fig/examples.png" width="800">
</p>
<p align="center">
(top) Surface normal estimation result from 10 images, (bottom) ground truth.
</p>




## Getting Started

### Prerequisites

- Python3
- torch
- tensorboard
- cv2
- scipy

Tested on:
- Ubuntu 20.04/Windows10, Python 3.7.5, Pytorch 1.6.0, CUDA 10.2
  - GPU: NvidiaQuadroRTX8000 (64GB)

### Running the test
For testing the network on DiLiGenT benchmark by Boxin Shi [2], please download [DiLiGenT dataset (DiLiGenT.zip)](https://sites.google.com/site/photometricstereodata/)  and extract it at [USER_PATH].

Then, please run main.py with the DiLiGenT path as an argument.

```
python main.py --diligent [USER_PATH]/DiLiGenT/pmsData
```

You can change the number of test images (default:10) as 

```
python main.py --diligent [USER_PATH]/DiLiGenT/pmsData --n_testimg 5
```

Please note that the lighting directions are randomly chosen, therefore the results are different every time.

### Pretrained Model
The pretrained model (our "full" configuration) is available at https://www.dropbox.com/s/64i4srb2vue9zrn/pretrained.zip?dl=0.
Please extract it at "PS-Transformer-BMVC2021/pretrained".

### Output
If the program properly works, you will get average angular errors (in degrees) for each dataset.

You can use [TensorBoard](https://www.tensorflow.org/tensorboard?hl=en) for visualizing your output. The log file will be saved at


```
[LOGFILE] = 'Tensorboard/[SESSION_NAME (default:eval)]'
```

Then, please run TensorBoard as

```
tensorboard --logdir [YOURLOGFILE]
```

### Important notice about DiLiGenT datasets

As is commonly known, "bear" dataset in DiLiGenT has problem and the first 20 images in bearPNG are skipped. 

### Running the test on othter datasets (Unsupported)
If you want to run this code on ohter datasets, please allocate your own data just in the same manner with DiLiGenT. The required files are
- images (.png format in default, but you can easily change the code for other formats)
- lights (light_directions.txt, light_intensities.txt)
- normals (normal.txt, if no ground truth surface normal is available, you can simply set all the values by zero)

### Running the training
The training script is NOT supported.
However, the training dataset is [available](https://www.dropbox.com/scl/fo/8w47k2pw6v8scaq3gjtw3/AIi7K11wJt4p4ZOt_RsVBSs?rlkey=0yyvdgr7kuv7duxspyjrt7yna&st=a06sy8f7&dl=0)

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## References
[2] Boxin Shi, Zhipeng Mo, Zhe Wu, Dinglong Duan, Sai-Kit Yeung, and Ping Tan, "A Benchmark Dataset and Evaluation for Non-Lambertian and Uncalibrated Photometric Stereo", In IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2018.

### Comment
Honestly, the major reason why this work was aimed at "sparse" set-up is simply because the model size is huge and I didn't have sufficient gpu resources for training my model on "dense" iamges (though test on dense images using the model trained on sparse images is possible as shown in the paper).  I am confident that this model also benefits the dense photometric stereo task and if you have any ideas to reduce the training cost, they are very appreciated! 
