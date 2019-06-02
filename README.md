# PI-Net: A deep learning approach to extract topological persistence images

Here we provide sample code to compute persistence images (PIs) using the proposed Image PI-Net model. We load weights from a pre-trained model trained on the CIFAR10 dataset.


## Key Files  

For sample test-set images in CIFAR10, both files first load weights from a pre-trained Image PI-Net model; next, compute PIs using the Image PI-Net model and finally compare the generated PIs to ground-truth PIs obtained using conventional topological data analysis (TDA) tools. In addition, the "main.py" file saves the PI comparisons for each sample image in the "Examples folder". We use [Scikit-TDA](https://scikit-tda.org/) to generate ground-truth PIs.

- main.ipynb 

- main.py

## Required Packages

Please install the following packages to before running the code.

- numpy
- scipy
- matplotlib
- keras (with tensorflow backend)

**Note:** If you have trouble running these codes, we illustrate the generated PIs in the "Examples" folder and for each image compare the generated PIs using the PI-Net model to the ground-truth PIs.
