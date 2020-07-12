from efficientnet.tfkeras import EfficientNetB0, EfficientNetB1, EfficientNetB2
from efficientnet.tfkeras import EfficientNetB3, EfficientNetB4, EfficientNetB5
from efficientnet.tfkeras import EfficientNetB6, EfficientNetB7


def efficientnet(b, weights='imagenet',
                 include_top=False,
                 input_shape=(None, None, 3)):
    if b == 0:
        return EfficientNetB0(weights=weights,
                              include_top=include_top,
                              input_shape=input_shape)
    elif b == 1:
        return EfficientNetB1(weights=weights,
                              include_top=include_top,
                              input_shape=input_shape)
    elif b == 2:
        return EfficientNetB2(weights=weights,
                              include_top=include_top,
                              input_shape=input_shape)
    elif b == 3:
        return EfficientNetB3(weights=weights,
                              include_top=include_top,
                              input_shape=input_shape)
    elif b == 4:
        return EfficientNetB4(weights=weights,
                              include_top=include_top,
                              input_shape=input_shape)
    elif b == 5:
        return EfficientNetB5(weights=weights,
                              include_top=include_top,
                              input_shape=input_shape)
    elif b == 6:
        return EfficientNetB6(weights=weights,
                              include_top=include_top,
                              input_shape=input_shape)
    elif b == 7:
        return EfficientNetB7(weights=weights,
                              include_top=include_top,
                              input_shape=input_shape)
    else:
        raise Exception("Invalid size for EfficientNet")
