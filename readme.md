# ResNet Implementation

### Info
For this repository, I implemented the [ResNet paper](https://arxiv.org/pdf/1512.03385.pdf) in PyTorch from scratch without referencing any prior implementations. I trained the model on the full [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset.

### Setup and Dependencies

Before cloning the repository, ensure that you have PyTorch installed, preferably in a `conda` environment (see [here](https://anaconda.org/pytorch/pytorch) for setup details). 

To set up this repository, first clone it on your local machine. Then, navigate to [CIFAR-10 webpage](https://www.cs.toronto.edu/~kriz/cifar.html) and downlaod "CIFAR-10 python version" (163 MB). Put the files `data_batch_*` and `test_batch` into the same folder as `resnet.ipynb`, or alternatively modify the file names in the notebook to properly load these files.

After properly loading the data, you should be able to run all of the blocks in `resnet.ipynb`.
