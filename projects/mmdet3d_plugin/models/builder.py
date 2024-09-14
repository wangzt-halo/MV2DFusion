import warnings

from mmcv.cnn import MODELS as MMCV_MODELS
from mmcv.utils import Registry

MODELS = Registry('models', parent=MMCV_MODELS)

QUERY_GENERATORS = MODELS


def build_query_generator(cfg):
    """Build loss."""
    return QUERY_GENERATORS.build(cfg)