from .callbacks import get_callbacks
from .hardware import detect_hardware
from .model import get_model, make_model_file, top_2_accuracy, top_5_accuracy
from .optimizer import get_optimizer

DROPOUT = [0.2, 0.2, 0.3, 0.3, 0.4, 0.4, 0.5, 0.5]

MOMENTUM = 0.9
EPSILON = 1
