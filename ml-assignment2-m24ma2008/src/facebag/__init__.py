"""Decision trees, bagging and SAMME boosting on Olivetti faces."""
from .data import load_olivetti, load_split, make_split, mask_images, IMG_SIDE
from .bagging import BaggedTrees
from .samme import SAMME
from .biasvar import bias_variance, identity_bootstrap
