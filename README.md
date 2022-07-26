## Learning Semantic Correspondence with Sparse Annotations  (To be appeared at ECCV'22)
For more information, check out our project [[website](https://shuaiyihuang.github.io/publications/SCorrSAN/)] and paper on [[arXiv](https://arxiv.org/pdf/2208.06974.pdf)].

Pretrained models are to be uploaded soon.


# Method

Our method is illustrated below:
![alt text](/images/METHOD.png)

# Environment Settings
```
git clone https://github.com/ShuaiyiHuang/SCorrSAN
cd SCorrSAN

conda create -n scorrsan python=3.6
conda activate scorrsan

pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install -U scikit-image
pip install git+https://github.com/albumentations-team/albumentations
pip install tensorboardX termcolor timm tqdm requests pandas
```

# Evaluation
- Download pre-trained weights on [Link](https://drive.google.com/drive/folders/todo) (TODO)
- All datasets are automatically downloaded into directory specified by argument `datapath`

Result on SPair-71k: (PCK 55.3%)

      python test.py --pretrained "/path_to_pretrained_model/spair" --benchmark spair

Results on PF-PASCAL: (PCK 81.5%, 93.3%, 96.6%)

      python test.py --pretrained "/path_to_pretrained_model/pfpascal" --benchmark pfpascal

Results on PF-WILLOW, (PCK 54.1%, 80.0%, 89.8%)

      python test.py --pretrained "/path_to_pretrained_model/pfpascal" --benchmark pfpascal

# Training

SPair-71k: (PCK 55.3%)

      sh ./scripts/train_spair.sh
 
PF-PASCAL: (PCK 81.5%, 93.3%, 96.6%)

      sh ./scripts/train_pfpascal.sh

# Acknowledgement <a name="Acknowledgement"></a>

This repository builds on other public projects, mainly [CATs](https://github.com/SunghwanHong/Cost-Aggregation-transformers), [DHPF](https://github.com/juhongm999/dhpf), and [GLU-Net](https://github.com/PruneTruong/GLU-Net). 
### BibTeX
If you find this research useful, please consider citing:
````BibTeX
@inproceedings{huang2022learning,
	title={Learning Semantic Correspondence with Sparse Annotations},
	author={Huang, Shuaiyi and Yang, Luyu and He, Bo and Zhang, Songyang and He, Xuming and Shrivastava, Abhinav},
	booktitle={Proceedings of the European Conference on Computer Vision(ECCV)},
	year={2022}
}
````
