from .signet import SigNet, SigNet_smaller, SigNet_thin

available_models = {'signet': SigNet,
                    'signet_thin': SigNet_thin,
                    'signet_smaller': SigNet_smaller}
