# Estimation of Near-Instance-Level Attribute Bottleneck for Zero-Shot Learning (IAB)
Codes of [Estimation of Near-Instance-Level Attribute Bottleneck for Zero-Shot Learning (IJCV 2024)](https://link.springer.com/article/10.1007/s11263-024-02021-x)

## Installation
```shell
$ cd repository
$ pip install -r requirements.txt
```

## Datasets
The splits of dataset and its attributes can be found in [data](https://drive.google.com/file/d/1bCZ28zJZNzsRjlHxH_vh2-9d7Ln1GgjE/view)[1]. Please download the data folder and place it in ./data/.

Set the --root in opt.py as your code path.

Please download CUB, AWA2, SUN, FLO datasets, and set the --image_root in opt.py to the datasets.

Please download pretrained resnet [weights](https://drive.google.com/drive/folders/1zra6v53trkd0x8ZmtzBkWwyJdHbVvy2H)[1] and place it in ./pretrained_models/

The final project folder should contain the following structure:
  ```html
  <IAB_DIR>/                   % IAB root
  ├── pretrained_models/                 % Place the pretrained resnet weights there
  ├── data/                 % dataset spilit for AWA2, CUB, FLO, SUN
  ├── logs/                 %  for saving the training logs
  ├── out/                 %  for saving weights during training
...
  ```

## Test
You can evaluate our [pretrained model](https://drive.google.com/file/d/1DDMeK4S0AMuo2MSPgBzQWJgSyx7XgIwt/view?usp=sharing).

Please specify the --model_path in opt.py and then run:

```shell
python test.py --dataset [CUB/AWA2/FLO/SUN]
```

## Train
If you wish to try training our model from scratch, please run IAB.py, for example:
```shell
python IAB.py --dataset [CUB/AWA2/FLO/SUN]
```
We have provided the following configurations in the /configs directory, which contain parameter settings for different datasets. However, it is important to note that these settings may not be optimal. If you wish to explore other configurations, we suggest adjusting a few of the more crucial ones.
* t        %temperature coefficient
* unfix_low        %determines whether the low-level layers in ResNet are frozen
* unfix_high        %determines whether the high-level layers in ResNet are frozen
* pretrain_epoch        %the number of epochs for the warm-up phase
* random_grouping  %specifies whether random divided groups are used
* Lp1  %the number of groups when utilizing random groups
* alphas %loss weights
  
## Acknowledgment

We are very grateful to the following repos for their great help in constructing our work:

[1] [APN](https://github.com/wenjiaXu/APN-ZSL). Xu W, Xian Y, Wang J, et al. Attribute prototype network for zero-shot learning[J]. Advances in Neural Information Processing Systems, 2020, 33: 21969-21980.

[2] [Softsort](https://github.com/sprillo/softsort). Prillo S, Eisenschlos J. Softsort: A continuous relaxation for the argsort operator[C]//International Conference on Machine Learning. PMLR, 2020: 7793-7802.

[3] [C2AM](https://github.com/CVI-SZU/CCAM). Xie J, Xiang J, Chen J, et al. C2AM: contrastive learning of class-agnostic activation map for weakly supervised object localization and semantic segmentation[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022: 989-998.



