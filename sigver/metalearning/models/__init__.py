import functools
from . import signet, smallcnn

available_models = {'signet': functools.partial(signet.SigNetModel, normalize=True),
                    'smallcnn': smallcnn.SmallCNN}
