# MS-PRL
Rethinking PRL: A Multiscale Progressively Residual Learning Network for Inverse Halftoning.

## Abstract

## Network architecture

## Contents
1. [Environment](#env)
2. [Demo](#demo)
3. [Train](#train)
4. [Test and Valid](#test)
5. [Dataset](#data)
6. [Model](#model)
7. [Citation](#cite)
8. [Other](#other)

## Environment <a name="env"></a>
```shell
python=3.8 numpy=1.21.2 opencv-python=4.5.5.64
pillow=8.4.0 numba=0.55.1 scikit-image=0.18.3
pytorch=1.10.0 torchvision=0.11.1 cudatoolkit=11.3
```

## Demo <a name="demo"></a>
demo images in ```demo/halftone/``` folder, and the output images in ```demo/output/``` folder.
```shell
python demo.py
```

## Train <a name="train"></a>
To train MS-PRL , run the command below:
```shell
python main.py --mode train --model_name=MSPRL
```
if you want to train other model, pleace change ```--model_name="your model name"```. The model weights will be saved in ```./checkpoint/model_name/model_name_iterations.pth``` folder.

## Test and Valid <a name="test"></a>
1. run test mode, images will be saved in ```./resutls/model_name/test_name/``` and the log will be saved in ```./logs/model_name/test/test_name/log.txt```.
2. run valid mode, just the log will be saved in ```./logs/model_name/test/test_name/log.txt```

To test MS-PRL , run the command below:
```shell
python main.py --mode test --model_name=MS-PRL
```
To valid MS-PRL , run the command below:
```shell
python main.py --mode valid --model_name=MS-PRL
```
Please pay attention to the **dataset path**, refer to the details of the [dataset](#data).

## Dataset <a name="data"></a>
1. Download [VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/), [Kodak25](http://r0k.us/graphics/kodak/), [Place365](http://places2.csail.mit.edu/) dataset and five standard benchmark datasets. You can also download our dataset in [here]().

2. To generate halftone image using Floyd Steinberg error diffusion, run the command below:
```shell
cd utils
python halftone.py
```

The data folder should be like the format below:
```
dataset
├─ train
│ ├─ data     % 13841 halftone images
│ │ ├─ xxxx.png
│ │ ├─ ......
│ │
│ ├─ target   % 13841 gray images
│ │ ├─ xxxx.png
│ │ ├─ ......
│
├─ valid
│ ├─ data     % 3000 halftone images
│ │ ├─ xxxx.png
│ │ ├─ ......
│ │
│ ├─ target   % 3000 gray images
│ │ ├─ xxxx.png
│ │ ├─ ......
|
├─ test
│ ├─ Class
| │ ├─ data     % halftone images
| │ │ ├─ xxxx.png
│ | │ ├─ ......
│ │
│ | ├─ target   % gray image
│ │ | ├─ xxxx.png
│ │ | ├─ ......
|
│ ├─ Kodak
| | ├─ ......

```

## Model <a name="model"></a>
We provide our all pre-trained models.
- MS-PRL, PRL-dt and other model in [here]().
The data folder should be like the format below:
```
checkpoint
├─ MSPRL
│ ├─ MSPRL_iteration.pth
│ │
├─ DnCNN
│ ├─ DnCNN_iteration.pth
│ │
```

## Citation <a name="cite"></a>
```BibTex

```

## Other <a name="other"></a>
Reference Code: 
1. https://github.com/chosj95/MIMO-UNet
2. https://github.com/swz30/MIRNetv2
3. https://github.com/csbhr/CNLRN
4. DnCNN: https://github.com/cszn/KAIR
