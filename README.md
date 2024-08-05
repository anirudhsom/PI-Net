# PI-Net: A Deep Learning Approach to Extract Topological Persistence Images

[Paper](https://arxiv.org/pdf/1906.01769.pdf)  

This repository contains:

- Sample code to compute persistence images (PIs) using the proposed Image PI-Net and Signal PI-Net models.
  
- The provided pretrained Image PI-Net model was trained using the [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset, and the provided pretrained Signal PI-Net model was trained using the [USC-HAD](https://sipi.usc.edu/had/) dataset. [Scikit-TDA](https://scikit-tda.org/) was used to generate ground-truth PIs.

Note, performance of the provided Signal PI-Net model differs from that described in the paper, since the original model was pretrained using the GENEactiv dataset which is not publicly released.  



## [Image PI-Net](https://github.com/anirudhsom/PI-Net/tree/master/Image%20PI-Net) 

For sample test-set images in CIFAR10, both "main_image_pinet.py" and "main_image_pinet.ipynb" do the following: 

- Load weights from a pretrained Image PI-Net model;

- Compute PIs using the Image PI-Net model;

- Compare the generated PIs to ground-truth PIs obtained using conventional topological data analysis (TDA) tools.

Additionally, the "main.py" file saves the PI comparisons for each sample image in the "Examples folder".



## [Signal PI-Net](https://github.com/anirudhsom/PI-Net/tree/master/Signal%20PI-Net) 

Here, we provide pretrained model weights and sample code to train the Signal PI-Net model using the USC-HAD dataset.

To extract ground-truth persistence images for your time-series data, please refer to the following repository: [Sublevel-Set-TDA](https://github.com/itsmeafra/Sublevel-Set-TDA). 



## [Required Packages](https://github.com/anirudhsom/PI-Net/blob/master/requirements.txt) 

Python 3.8.5 was used to create an environment with the following packages.

- tensorflow == 2.10.0
- keras == 2.10.0
- matplotlib
- numpy
- scikit-learn
- scikit-tda


## Citation

```
@inproceedings{som2020pi,
  title={PI-Net: A Deep Learning Approach to Extract Topological Persistence Images},
  author={Som, Anirudh and Choi, Hongjun and Ramamurthy, Karthikeyan Natesan and Buman, Matthew P and Turaga, Pavan},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition workshops},
  pages={834--835},
  year={2020}
}
```
