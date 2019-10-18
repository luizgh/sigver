# Learning representations for Offline Handwritten Signature Verification

This repository contains code to train CNNs for feature extraction for Offline Handwritten Signatures, code to 
train writer-dependent classifiers [1] and code to train meta-learners [3]. It also contains code to train two countermeasures for Adversarial Examples,
as described in [2] (the code to run the experiments from this second paper can be found in this [repository](https://github.com/luizgh/adversarial_signatures)).

[1] Hafemann, Luiz G., Robert Sabourin, and Luiz S. Oliveira. "Learning Features for Offline Handwritten Signature Verification using Deep Convolutional Neural Networks" http://dx.doi.org/10.1016/j.patcog.2017.05.012 ([preprint](https://arxiv.org/abs/1705.05787))

[2] Hafemann, Luiz G., Robert Sabourin, and Luiz S. Oliveira. "Characterizing and evaluating adversarial examples for Offline Handwritten Signature Verification" https://doi.org/10.1109/TIFS.2019.2894031 ([preprint](https://arxiv.org/abs/1901.03398))

[3] Hafemann, Luiz G., Robert Sabourin, and Luiz S. Oliveira. "Meta-learning for fast classifier adaptation to new users of Signature Verification systems" ([preprint](https://arxiv.org/abs/1910.08060))

This code for feature extraction and writer-dependent classifiers is a re-implementation in Pytorch (original code for [1] was written in theano+lasagne: [link](https://github.com/luizgh/sigver_wiwd)). 

# Installation

This package requires python 3. Installation can be done with pip:

```bash
pip install git+https://github.com/luizgh/sigver.git  --process-dependency-links
```

You can also clone this repository and install it with ```pip install -e <path/to/repository>  --process-dependency-links```

# Usage

## Data preprocessing

The functions in this package expect training data to be provided in a single .npz file, with the following components:

* ```x```: Signature images (numpy array of size N x 1 x H x W)
* ```y```: The user that produced the signature (numpy array of size N )
* ```yforg```: Whether the signature is a forgery (1) or genuine (0) (numpy array of size N )

We provide functions to process some commonly used datasets in the script ```sigver.datasets.process_dataset```. 
As an example, the following code pre-process the MCYT dataset with the procedure from [1] (remove background, center in canvas and resize to 170x242)

```bash
python -m sigver.preprocessing.process_dataset --dataset mcyt \
 --path MCYT-ORIGINAL/MCYToffline75original --save-path mcyt_170_242.npz
```

During training a random crop of size 150x220 is taken for each iteration. During test we use the center 150x220 crop.

## Training a CNN for Writer-Independent feature learning

This repository implements the two loss functions defined in [1]: SigNet (learning from genuine signatures only)
and SigNet-F (incorporating knowledge of forgeries). In the training script, the flag
```--users``` is use to define the users that are used for feature learning. In [1],
GPDS users 300-881 were used (```--users 300 881```). 


Training SigNet:

```
python -m sigver.featurelearning.train --model signet --dataset-path  <data.npz> --users [first last]\ 
--model signet --epochs 60 --logdir signet  
```

Training SigNet-F with lambda=0.95:

```
python -m sigver.featurelearning.train --model signet --dataset-path  <data.npz> --users [first last]\ 
--model signet --epochs 60 --forg --lamb 0.95 --logdir signet_f_lamb0.95  
```

For checking all command-line options, use ```python -m sigver.featurelearning.train --help```. 
In particular, the option ```--visdom-port``` allows real-time monitoring using [visdom](https://github.com/facebookresearch/visdom) (start the visdom
server with ```python -m visdom.server -port <port>```).   

## Training WD classifiers and evaluating the result

For training and testing the WD classifiers, use the ```sigver.wd.test``` script. Example:

```bash
python -m sigver.wd.test -m signet --model-path <path/to/trained_model> \
    --data-path <path/to/data> --save-path <path/to/save> \
    --exp-users 0 300 --dev-users 300 881 --gen-for-train 12
```

Where trained_model is a .pth file (trained with the script above, or pre-trained - see the section below).
This script will split the dataset into train/test, train WD classifiers and evaluate then on the test set. This
is performed for K random splits (default 10). The script saves a pickle file containing a list, where each element is the result 
of one random split. Each item contains a dictionary with:

* 'all_metrics': a dictionary containing:
  * 'FRR': false rejection rate
  * 'FAR_random': false acceptance rate for random forgeries
  * 'FAR_skilled': false acceptance rate for skilled forgeries
  * 'mean_AUC': mean Area Under the Curve (average of AUC for each user)
  * 'EER': Equal Error Rate using a global threshold
  * 'EER_userthresholds': Equal Error Rate using user-specific thresholds
  * 'auc_list': the list of AUCs (one per user)
  * 'global_threshold': the optimum global threshold (used in EER)
* 'predictions': a dictionary containing the predictions for all images on the test set:
  * 'genuinePreds': Predictions to genuine signatures
  * 'randomPreds': Predictions to random forgeries
  * 'skilledPreds': Predictions to skilled forgeries


The example above train WD classifiers for the exploitation set (users 0-300) using a development
set (users 300-881), with 12 genuine signatures per user (this is the setup from [1] - refer to 
the paper for more details). For knowing all command-line options, run ```python -m sigver.wd.test -m signet```. 

# Pre-trained models

Pre-trained models can be found here: 
* SigNet ([link](https://drive.google.com/open?id=1l8NFdxSvQSLb2QTv71E6bKcTgvShKPpx))
* SigNet-F lambda 0.95 ([link](https://drive.google.com/open?id=1ifaUiPtP1muMjt8Tkrv7yJj7we8ttncW))


These models contains the weights for the feature extraction layers.

**Important**: These models were trained with pixels ranging from [0, 1]. Besides the pre-processing steps described above, you need to divide each  pixel by 255 to be in the range. This can be done as follows: ```x = x.float().div(255)```. Note that Pytorch does this conversion automatically if you use ```torchvision.transforms.totensor```, which is used during training.

Usage:

```python

import torch
from sigver.featurelearning.models import SigNet

# Load the model
state_dict, classification_layer, forg_layer = torch.load('models/signet.pth')
base_model = SigNet().eval()
base_model.load_state_dict(state_dict)

# Extract features
with torch.no_grad(): # We don't need gradients. Inform torch so it doesn't compute them
    features = base_model(input)

```

See ```example.py``` for a complete example. For a jupyter notebook, see this [interactive example](https://nbviewer.jupyter.org/github/luizgh/sigver/blob/master/interactive_example.ipynb).


# Meta-learning 

To train a meta-learning model, use the `sigver.metalearning.train` script:
```bash
python -m sigver.metalearning.train --dataset-path <path/to/datataset.npz> \
    --pretrain-epochs <number_pretrain_epochs> --num-updates <number_gradient_steps> --num-rf <num_random_forgeries> \
    --epochs <number_epochs> --num-sk-test <number_skilled_in_Dtest> --model <model>
```

Where `num-updates` is the number of gradient descent steps in the classifier adaptation (`K` in the paper), and `num-rf` refers to 
the number of random forgeries in the classifier adaptation steps (set `--num-rf 0` for the one-class formulation). Refer to the
sigver/metalearning/train.py script for a complete list of arguments.

# Citation

If you use our code, please consider citing the following papers:

[1] Hafemann, Luiz G., Robert Sabourin, and Luiz S. Oliveira. "Learning Features for Offline Handwritten Signature Verification using Deep Convolutional Neural Networks" http://dx.doi.org/10.1016/j.patcog.2017.05.012 ([preprint](https://arxiv.org/abs/1705.05787))

[2] Hafemann, Luiz G., Robert Sabourin, and Luiz S. Oliveira. "Characterizing and evaluating adversarial examples for Offline Handwritten Signature Verification" ([preprint](https://arxiv.org/abs/1901.03398))

[3] Hafemann, Luiz G., Robert Sabourin, and Luiz S. Oliveira. "Meta-learning for fast classifier adaptation to new users of Signature Verification systems" ([preprint](https://arxiv.org/abs/1910.08060))

# License

The source code is released under the BSD 3-clause license. Note that the trained models used the GPDS dataset for training, which is restricted for non-commercial use.  

Please do not contact me requesting access to any particular dataset. These requests should be addressed directly to the groups that collected them.
