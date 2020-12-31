# Universal Adversarial Perturbation with virtual data
This is the repository accompanying our CVPR 2020 paper [Understanding Adversarial Examples from the Mutual Influence of Images and Perturbations](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhang_Understanding_Adversarial_Examples_From_the_Mutual_Influence_of_Images_and_CVPR_2020_paper.pdf)

## Setup
You can install the requirements with `pip3 install requirements.txt`.

### Config
Copy the `sample_config.py` to `config.py` (`cp ./config/sample_config.py ./config/config.py`) and edit the paths accordingly.

### Datasets
The code supports training UAPs on ImageNet, MS COCO, PASCAL VOC and Places365

#### ImageNet
The [ImageNet](http://www.image-net.org/) dataset should be preprocessed, such that the validation images are located in labeled subfolders as for the training set. You can have a look at this [bash-script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh) if you did not process your data already. Set the paths in your `config.py`.
```
IMAGENET_PATH = "/path/to/Data/ImageNet"
```

#### COCO
The [COCO](https://cocodataset.org/#home) 2017 images can be downloaded from here for [training](http://images.cocodataset.org/zips/train2017.zip) and [validation](http://images.cocodataset.org/zips/val2017.zip). After downloading and extracting the data update the paths in your `config.py`.
```
COCO_2017_TRAIN_IMGS = "/path/to/COCO/train2017/"			
COCO_2017_TRAIN_ANN = "/path/to/COCO/annotations/instances_train2017.json"
COCO_2017_VAL_IMGS = "/path/to/COCO/val2017/"
COCO_2017_VAL_ANN = "/path/to/instances_val2017.json"
```

#### PASCAL VOC
The training/validation data of the [PASCAL VOC2012 Challenge](http://host.robots.ox.ac.uk/pascal/VOC/) can be downloaded from [here](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar). After downloading and extracting the data update the paths in your `config.py`.
```
VOC_2012_ROOT = "/path/to/Data/VOCdevkit/"
```

#### Places 365
The [Places365](http://places2.csail.mit.edu/index.html) data can be downloaded from [here](http://places2.csail.mit.edu/download.html). After downloading and extracting the data update the paths in your `config.py`.
```
PLACES365_ROOT = "/home/user/Data/places365/"
```

## Run
Run `bash ./run.sh` to generate UAPs for different target models trained on ImageNet using virtual data Places365. The bash script should be easy to adapt to perform different experiments. The jupyter notebook `pcc_analysis.ipynb` is an example for the PCC-analysis discussed in the paper. 

## Citation
```
@inproceedings{zhang2020understanding,
  title={Understanding Adversarial Examples From the Mutual Influence of Images and Perturbations},
  author={Zhang, Chaoning and Benz, Philipp and Imtiaz, Tooba and Kweon, In So},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={14521--14530},
  year={2020}
}
```