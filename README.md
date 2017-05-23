# Caption-Guided Saliency
This code is released as a supplementary material to ["Top-down Visual Saliency Guided by Captions" (CVPR 2017)][1].


![](https://www.dropbox.com/s/zjyy04up13lp657/video9461.gif?raw=1) ![](https://www.dropbox.com/s/3r2x5fwda4nkatu/video7023.gif?raw=1)

## Getting started

**Clone this repo (including coco-caption as a submodule):**
```bash
$ git clone --recursive git@github.com:VisionLearningGroup/caption-guided-saliency.git
```
**Install dependencies**

The model is implemented using TensorFlow framework, Python 2.7. For TensorFlow installation please refer to the official [Installing TensorFlow](https://www.tensorflow.org/install/) guide or simply:

```bash
$ pip install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.1.0-cp27-none-linux_x86_64.whl
```
*Warning!* The standard version of TensorFlow gives the warnings like:
```
The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
```
It's fine. To get rid of them you'll need to build TensorFlow from sources with `--config=opt`.

List of other required python modules:
```bash
$ pip install tqdm numpy six pillow matplotlib scipy
```

The code also uses `ffmpeg` for data preprocessing. 

**Obtain the dataset you need:**

  * [MSR-VTT](http://ms-multimedia-challenge.com/dataset): 
  [train_val_videos.zip](http://202.38.69.241/static/resource/train_val_videos.zip),
  [train_val_annotation.zip](http://202.38.69.241/static/resource/train_val_annotation.zip), 
  [test_videos.zip](http://202.38.69.241/static/resource/test_videos.zip), 
  [test_videodatainfo.json](http://ms-multimedia-challenge.com/static/resource/test_videodatainfo.json)
  
  * [Flickr30k](http://shannon.cs.illinois.edu/DenotationGraph/): flickr30k.tar.gz, flickr30k-images.tar

and unpack files into their respective directories under ```./DATA/```.

Expected layout so far is:
```
./DATA/
    └───MSR_VTT/
    │   │   test_videodatainfo.json
    │   │   train_val_videodatainfo.json
    │   │
    │   └───TestVideo/
    │   │       ...
    │   │   
    │   └───TrainValVideo/
    │           ...
    └───Flickr30k
        │   results_20130124.token
        │      
        └───flickr30k-images/
                ...
```

**Run data preprocessing**
```bash
$ python preprocessing.py --dataset {MSR-VTT|Flickr30k}
```
This step takes ~30mins for Flickr30k and ~2h for MSR-VTT. 

**Run training**
```bash
$ python run_s2vt.py --dataset {MSR-VTT|Flickr30k} --train
```
We do not finetune CNN part of the model, thus, training on GPU takes only several hours. Training/validation/test splits for Flickr30k are taken from [NeuralTalk](https://github.com/karpathy/neuraltalk). After the training you can run evaluation of the model:

```bash
$ python run_s2vt.py --dataset {MSR-VTT|Flickr30k} --test --checkpoint {number}
```

## Saliency Visualization
After you got the model which was trained to produce captions for MSR-VTT dataset, you can get video with saliency visualization similar to those in the beginning of the readme: 

```bash
$ python visualization.py --dataset MSR-VTT     \
                          --media_id video9461  \
                          --checkpoint {number} \
                          --sentence "A man is driving a car"
```
where *media_id* should belong to the test split of MSR-VTT, *sentence* sets a query phrase.

### What's next

You can change model's parameters (dimensionality of layers, learning rate etc.) directly in cfg.py. Every run of `run_s2vt.py` with `--train` switch will overwrite files in `experiments` directory. 

### References

[1]: https://arxiv.org/abs/1612.07360
    

If you find this useful in your work please consider citing:
```
@inproceedings{Ramanishka2017cvpr,
          title = {Top-down Visual Saliency Guided by Captions},
          author = {Vasili Ramanishka and Abir Das and Jianming Zhang and Kate Saenko},
          booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
          year = {2017}
          }
```
