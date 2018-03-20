import tensorflow as tf
from keras.layers import Conv2D
from keras.layers import Deconvolution2D
from keras.layers import Conv2DTranspose
from keras.layers import InputLayer
from keras.layers import Input
from keras.layers import merge
from keras.models import Sequential
from keras.models import Model
from layers import ImageRescale
from layers import Conv2DSubPixel


def bicubic(x, scale=3):
    model = Sequential()
    model.add(InputLayer(input_shape=x.shape[-3:]))
    model.add(ImageRescale(scale, method=tf.image.ResizeMethod.BICUBIC))
    return model


def srcnn(x, f=[9, 1, 5], n=[64, 32], scale=3):
    """Build an SRCNN model.

    See https://arxiv.org/abs/1501.00092
    """
    assert len(f) == len(n) + 1
    model = bicubic(x, scale=scale)
    c = x.shape[-1]
    for ni, fi in zip(n, f):
        model.add(Conv2D(ni, fi, padding='same',
                         kernel_initializer='he_normal', activation='relu'))
    model.add(Conv2D(c, f[-1], padding='same',
                     kernel_initializer='he_normal'))
    return model


def fsrcnn(x, d=56, s=12, m=4, scale=3):
    """Build an FSRCNN model.

    See https://arxiv.org/abs/1608.00367
    """
    model = Sequential()
    model.add(InputLayer(input_shape=x.shape[-3:]))
    c = x.shape[-1]
    f = [5, 1] + [3] * m + [1]
    n = [d, s] + [s] * m + [d]
    for ni, fi in zip(n, f):
        model.add(Conv2D(ni, fi, padding='same',
                         kernel_initializer='he_normal', activation='relu'))
    model.add(Conv2DTranspose(c, 9, strides=scale, padding='same',
                              kernel_initializer='he_normal'))
    return model


def espcn(x, f=[5, 3, 3], n=[64, 32], scale=3):
    """Build an ESPCN model.

    See https://arxiv.org/abs/1609.05158
    """
    assert len(f) == len(n) + 1
    model = Sequential()
    model.add(InputLayer(input_shape=x.shape[1:]))
    c = x.shape[-1]
    for ni, fi in zip(n, f):
        model.add(Conv2D(ni, fi, padding='same',
                         kernel_initializer='he_normal', activation='tanh'))
    model.add(Conv2D(c * scale ** 2, f[-1], padding='same',
                     kernel_initializer='he_normal'))
    model.add(Conv2DSubPixel(scale))
    return model


def ees(x, scale=3):
    """Build an EES model.

    See https://arxiv.org/pdf/1607.07680
    """
    c = x.shape[-1]
    input_img = Input(shape=x.shape[1:])
    ees = Conv2D(nb_filter=8, nb_row=3, nb_col=3, init='he_normal', activation='relu', border_mode='same', bias=True)(input_img)
    ees = Deconvolution2D(nb_filter=16, nb_row=14, nb_col=14, output_shape=(None, c * scale, c * scale, 16),
        subsample=(scale, scale), border_mode='same', init='glorot_uniform', activation='relu')(ees)
    output_img = Conv2D(nb_filter=1, nb_row=5, nb_col=5, init='glorot_uniform', activation='relu', border_mode='same')(ees)
    model = Model(input=input_img, output=output_img)
    return model
def eed(x, scale=3):
    """Build an EES model.

    See https://arxiv.org/pdf/1607.07680
    """
    c = x.shape[-1]
    input_img = Input(shape=x.shape[1:])
    # Feature extractor
    feature1 = Conv2D(nb_filter=64, nb_row=3, nb_col=3, init='glorot_uniform', activation='relu', border_mode='same', bias=True)(input_img)
    feature1 = Conv2D(nb_filter=64, nb_row=3, nb_col=3, init='glorot_uniform', activation='relu', border_mode='same', bias=True)(feature1)
    feature2 = Conv2D(nb_filter=64, nb_row=3, nb_col=3, init='glorot_uniform', activation='relu', border_mode='same', bias=True)(feature1)
    feature_out = merge(inputs=[feature1, feature2], mode='sum')
    # Upsampling
    upsamp1 = Conv2D(nb_filter=8, nb_row=1, nb_col=1, init='glorot_uniform', activation='relu', border_mode='same', bias=True)(feature_out)
    upsamp2 = Deconvolution2D(nb_filter=8, nb_row=14, nb_col=14, output_shape=(None, c * scale, c * scale, 8), subsample=(scale, scale), border_mode='same', init='glorot_uniform', activation='relu')(upsamp1)
    upsamp3 = Conv2D(nb_filter=64, nb_row=1, nb_col=1, init='glorot_uniform', activation='relu', border_mode='same', bias=True)(upsamp2)
    # Multi-scale Reconstruction
    Reslayer1 = Conv2D(nb_filter=64, nb_row=3, nb_col=3, init='glorot_uniform', activation='relu', border_mode='same', bias=True)(upsamp3)
    Reslayer2 = Conv2D(nb_filter=64, nb_row=3, nb_col=3, init='glorot_uniform', activation='relu', border_mode='same', bias=True)(Reslayer1)
    block1 = merge(inputs=[Reslayer1, Reslayer2], mode='sum')
    Reslayer3 = Conv2D(nb_filter=64, nb_row=3, nb_col=3, init='glorot_uniform', activation='relu', border_mode='same', bias=True)(block1)
    Reslayer4 = Conv2D(nb_filter=64, nb_row=3, nb_col=3, init='glorot_uniform', activation='relu', border_mode='same', bias=True)(Reslayer3)
    block2 = merge(inputs=[Reslayer3, Reslayer4], mode='sum')
    # ***************//
    Multi_scale1 = Conv2D(nb_filter=16, nb_row=1, nb_col=1, init='glorot_uniform', activation='relu', border_mode='same', bias=True)(block2)
    Multi_scale2a = Conv2D(nb_filter=16, nb_row=1, nb_col=1, init='glorot_uniform', activation='relu', border_mode='same', bias=True)(Multi_scale1)
    Multi_scale2b = Conv2D(nb_filter=16, nb_row=3, nb_col=3, init='glorot_uniform', activation='relu', border_mode='same', bias=True)(Multi_scale1)
    Multi_scale2c = Conv2D(nb_filter=16, nb_row=5, nb_col=5, init='glorot_uniform', activation='relu', border_mode='same', bias=True)(Multi_scale1)
    Multi_scale2d = Conv2D(nb_filter=16, nb_row=7, nb_col=7, init='glorot_uniform', activation='relu', border_mode='same', bias=True)(Multi_scale1)
    Multi_scale2 = merge(inputs=[Multi_scale2a, Multi_scale2b, Multi_scale2c, Multi_scale2d], mode='concat')
    # OUT
    output_img = Conv2D(nb_filter=1, nb_row=1, nb_col=1, init='glorot_uniform',activation='relu', border_mode='same', bias=True)(Multi_scale2)
    model = Model(input=input_img, output=output_img)
    return model
def eeds(x, scale=3):
    """Build an EEDS model.

    See https://arxiv.org/pdf/1607.07680
    """
    _input = Input(shape=x.shape[1:])
    EES = ees(x, scale)(_input)
    EED = eed(x, scale)(_input)
    EEDS = merge(inputs=[EED, EES], mode='sum')
    model = Model(input=_input, output=EEDS)
    return model


def vdsr(x, f=3, n=64, l=19, scale=3):
    """Build an VDSR-CNN model.

    See https://arxiv.org/abs/1511.04587
    """
    input_img = Input(shape=x.shape[-3:])
    input_img_scale = ImageRescale(scale, method=tf.image.ResizeMethod.BICUBIC)(input_img)
    model = Conv2D(n, (f, f), padding='same', kernel_initializer='he_normal', activation='relu')(input_img_scale)
    for i in range(l-1):
        model = Conv2D(n, (f, f), padding='same', kernel_initializer='he_normal', activation='relu')(model)
    model = Conv2D(1, (f, f), padding='same', kernel_initializer='he_normal')(model)
    output_img = merge([model, input_img_scale], mode='sum')
    model = Model(input_img, output_img)
    return model


def get_model(name):
    return globals()[name]

