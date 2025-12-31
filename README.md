# Gravityformer

Code for manuscript "A Gravity-informed Spatiotemporal Transformer for Human Activity Intensity Prediction".

# Introduction:
This is a Pytorch implementation of Gravityformer. Our code is based on ASTGCN (https://github.com/guoshnBJTU/ASTGCN-r-pytorch).

# Pre:
Step1: Clone the code of ASTGCN.

Step2: Put Gravityformer.py into model and SF_astgcn.conf into configurations.

Step3: Put SF_r3_d0_w0_astcgn.npz into data/SF/ and also adj_SF.npz into data/SF/.


# Train and Test:
Please refer to ASTGCN's Run and Test (https://github.com/guoshnBJTU/ASTGCN-r-pytorch).


# Datasets:
Due to privacy issues associated with the datasets, please contact Prof. Di Zhu (dizhu@umn.edu) if you require access to the data.


# Reference
```latex
@ARTICLE{11218803,
  title={A Gravity-informed Spatiotemporal Transformer for Human Activity Intensity Prediction},
  author={Wang, Yi and Wang, Zhenghong and Zhang, Fan and Kang, Chaogui and Ruan, Sijie and Zhu, Di and Tang, Chengling
and Ma, Zhongfu and Zhang, Weiyu and Zheng, Yu and Yu, Philip S. and Liu, Yu},
  booktitle={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  pages={1-18},
  year={2025},
  doi={10.1109/TPAMI.2025.3625859}
}
```
