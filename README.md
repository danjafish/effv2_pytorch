## effv2_pytorch

My implementation of effnetv2 in pytorch. 

Based on original repo: https://github.com/google/automl/tree/master/efficientnetv2

Paper: https://arxiv.org/pdf/2104.00298v3.pdf

This is just a playground project. For better implementation check https://github.com/rwightman/pytorch-image-models/

## Usage

1. git clone https://github.com/danjafish/effv2_pytorch.git
2. Install package with cd effv2_pytorch/ &&  pip install .
3. Use model as simple as from effnetv2_pytorch.effnetv2_model import EffnetV2Model 

Avaliable params are:

	include_top=False
	model_name='efficientnetv2-s'
	n_channels=3
	n_classes=None
	
Avaliable model_name values:
	
	'efficientnetv2-s'
	'efficientnetv2-m'
	'efficientnetv2-l'

See example in test.ipynb. See dummy training examples in effnetv2_simple_train.ipynb

