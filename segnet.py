import caffe
from caffe import layers as L
from caffe import params as P

def conv_2(bottom, num_output=64, kernel_size=3, stride=1, pad=0):
    conv1 = L.Convolution(bottom, num_output=num_output, kernel_size=kernel_size, stride=stride, pad=pad,
                         param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                         weight_filler=dict(type='msra', std=0.01),
                         bias_filler=dict(type='constant', value=0))
    #conv_bn = L.BatchNorm(conv, use_global_stats=False, in_place=True)
    relu1 = L.ReLU(conv1, in_place=True)

    conv2 = L.Convolution(relu1, num_output=num_output, kernel_size=kernel_size, stride=stride, pad=pad,
                          param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                          weight_filler=dict(type='msra', std=0.01),
                          bias_filler=dict(type='constant', value=0))
    # conv_bn = L.BatchNorm(conv, use_global_stats=False, in_place=True)
    relu2 = L.ReLU(conv2, in_place=True)
    return conv2, relu2

def conv_3(bottom, num_output=64, kernel_size=3, stride=1, pad=0):
    conv1 = L.Convolution(bottom, num_output=num_output, kernel_size=kernel_size, stride=stride, pad=pad,
                         param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                         weight_filler=dict(type='msra', std=0.01),
                         bias_filler=dict(type='constant', value=0))
    #conv_bn = L.BatchNorm(conv, use_global_stats=False, in_place=True)
    relu1 = L.ReLU(conv1, in_place=True)

    conv2 = L.Convolution(relu1, num_output=num_output, kernel_size=kernel_size, stride=stride, pad=pad,
                          param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                          weight_filler=dict(type='msra', std=0.01),
                          bias_filler=dict(type='constant', value=0))
    # conv_bn = L.BatchNorm(conv, use_global_stats=False, in_place=True)
    relu2 = L.ReLU(conv2, in_place=True)

    conv3 = L.Convolution(relu2, num_output=num_output, kernel_size=kernel_size, stride=stride, pad=pad,
                          param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                          weight_filler=dict(type='msra', std=0.01),
                          bias_filler=dict(type='constant', value=0))
    # conv_bn = L.BatchNorm(conv, use_global_stats=False, in_place=True)
    relu3 = L.ReLU(conv3, in_place=True)
    return conv3, relu3

def pool_layer(bottom):
    pool = L.Pooling(bottom, kernel_size=3, stride=2, pool=P.Pooling.MAX)
    relu = L.ReLU(pool, in_place=True)
    return relu

def eltwize_relu(bottom1, bottom2):
    residual_eltwise = L.Eltwise(bottom1, bottom2, eltwise_param=dict(operation=1))
    residual_eltwise_relu = L.ReLU(residual_eltwise, in_place=True)
    return residual_eltwise, residual_eltwise_relu

def split_concat(bottom):
    bottom_layers = [bottom]*16
    concat = L.Concat(*bottom_layers)
    return concat