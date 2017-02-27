import caffe
from caffe import layers as L
from caffe import params as P

def conv_1(bottom, num_output=64, kernel_size=5, stride=1, pad=2):
    conv1 = L.Convolution(bottom, num_output=num_output, kernel_size=kernel_size, stride=stride, pad=pad,
                         param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                         weight_filler=dict(type='msra', variance_norm=2),
                         bias_filler=dict(type='constant', value=0))
    #conv_bn = L.BatchNorm(conv, use_global_stats=False, in_place=True)
    return conv1

def conv_2(bottom, num_output=64, kernel_size=5, stride=1, pad=2):
    conv1 = L.Convolution(bottom, num_output=num_output, kernel_size=kernel_size, stride=stride, pad=pad,
                         param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                         weight_filler=dict(type='msra', variance_norm=2),
                         bias_filler=dict(type='constant', value=0))
    #conv_bn = L.BatchNorm(conv, use_global_stats=False, in_place=True)
    return conv1

def conv_3(bottom, num_output=64, kernel_size=5, stride=1, pad=2):
    conv1 = L.Convolution(bottom, num_output=num_output, kernel_size=kernel_size, stride=stride, pad=pad,
                         param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                         weight_filler=dict(type='msra', variance_norm=2),
                         bias_filler=dict(type='constant', value=0))
    #conv_bn = L.BatchNorm(conv, use_global_stats=False, in_place=True)
    relu1 = L.PReLU(conv1, in_place=True)

    conv2 = L.Convolution(relu1, num_output=num_output, kernel_size=kernel_size, stride=stride, pad=pad,
                          param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                          weight_filler=dict(type='msra', variance_norm=2),
                          bias_filler=dict(type='constant', value=0))
    # conv_bn = L.BatchNorm(conv, use_global_stats=False, in_place=True)
    return conv2

def conv_4(bottom, num_output=64, kernel_size=5, stride=1, pad=2):
    conv1 = L.Convolution(bottom, num_output=num_output, kernel_size=kernel_size, stride=stride, pad=pad,
                         param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                         weight_filler=dict(type='msra', variance_norm=2),
                         bias_filler=dict(type='constant', value=0))
    #conv_bn = L.BatchNorm(conv, use_global_stats=False, in_place=True)
    relu1 = L.PReLU(conv1, in_place=True)

    conv2 = L.Convolution(relu1, num_output=num_output, kernel_size=kernel_size, stride=stride, pad=pad,
                          param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                          weight_filler=dict(type='msra', std=0.01),
                          bias_filler=dict(type='constant', value=0))
    # conv_bn = L.BatchNorm(conv, use_global_stats=False, in_place=True)
    relu2 = L.PReLU(conv2, in_place=True)

    conv3 = L.Convolution(relu2, num_output=num_output, kernel_size=kernel_size, stride=stride, pad=pad,
                          param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                          weight_filler=dict(type='msra', variance_norm=2),
                          bias_filler=dict(type='constant', value=0))
    # conv_bn = L.BatchNorm(conv, use_global_stats=False, in_place=True)
    return conv3

def fcn(bottom, num_output=64, kernel_size=1, stride=1, pad=0):
    conv1 = L.Convolution(bottom, num_output=num_output, kernel_size=kernel_size, stride=stride, pad=pad,
                         param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                         weight_filler=dict(type='msra', variance_norm=2),
                         bias_filler=dict(type='constant', value=0))
    #conv_bn = L.BatchNorm(conv, use_global_stats=False, in_place=True)
    return conv1

def down_conv(bottom, num_output):
    conv = L.Convolution(bottom, num_output=num_output, kernel_size=2, stride=2, pad=0,
                         param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                         weight_filler=dict(type='msra', variance_norm=2),
                         bias_filler=dict(type='constant', value=0))
    relu = L.PReLU(conv, in_place=True)
    return relu

def deconv(bottom, num_output):
    conv = L.Deconvolution(bottom,
                           param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                           convolution_param=dict(num_output=num_output, kernel_size=2, stride=2,pad=0,
                                                  weight_filler=dict(type="msra",variance_norm=2),
                                                  bias_filler=dict(type="constant",value=0)))
    relu = L.PReLU(conv, in_place=True)
    return relu

def conv_relu_conv(bottom, num_output):
    conv1 = L.Convolution(bottom, num_output=num_output, kernel_size=2, stride=2, pad=0,
                         param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                         weight_filler=dict(type='msra', variance_norm=2),
                         bias_filler=dict(type='constant', value=0))
    relu = L.PReLU(conv1, in_place=True)
    conv2 = L.Convolution(relu, num_output=num_output, kernel_size=2, stride=2, pad=0,
                         param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                         weight_filler=dict(type='msra', variance_norm=2),
                         bias_filler=dict(type='constant', value=0))
    return conv2

def add_layer(bottom1, bottom2):
    residual_eltwise = L.Eltwise(bottom1, bottom2, eltwise_param=dict(operation=1))
    residual_eltwise_relu = L.PReLU(residual_eltwise, in_place=True)
    return residual_eltwise_relu

def split_concat(bottom):
    bottom_layers = [bottom,bottom,bottom,bottom,bottom,bottom,bottom,bottom,
                     bottom, bottom, bottom, bottom, bottom, bottom, bottom, bottom]
    concat = L.Concat(*bottom_layers)
    return concat


class SegNet(object):

    def layers_proto(self, batch_size, phase='TRAIN'):
        net = caffe.NetSpec()

        net.data, net.label = L.Data(batch_size=batch_size, ntop=2)
        net.conv1 = conv_1(net.data, 16)
        net.concat1 = split_concat(net.data)
        net.block1 = add_layer(net.concat1, net.conv1)
        net.pooling1 = down_conv(net.block1, 32)

        net.conv2 = conv_2(net.pooling1, 32)
        net.block2 = add_layer(net.pooling1,net.conv2)
        net.pooling2 = down_conv(net.block2, 64)

        net.conv3 = conv_3(net.pooling2, 64)
        net.block3 = add_layer(net.pooling2, net.conv3)
        net.pooling3 = down_conv(net.block3, 128)

        net.conv4 = conv_4(net.pooling3, 128)
        net.block4 = add_layer(net.pooling3, net.conv4)
        net.pooling4 = down_conv(net.block4, 256)

        net.conv5 = conv_4(net.pooling4, 256)
        net.block5 = add_layer(net.pooling4, net.conv5)
        net.depooling1 = deconv(net.block5, 128)
        net.concat5 = L.Concat(net.depooling1, net.block4)

        net.conv6 = conv_4(net.concat5, 256)
        net.block6 = add_layer(net.concat5, net.conv6)
        net.depooling2 = deconv(net.block6, 64)
        net.concat6 = L.Concat(net.depooling2, net.block3)

        net.conv7 = conv_3(net.concat6, 128)
        net.block7 = add_layer(net.concat6, net.conv7)
        net.depooling3 = deconv(net.block7, 32)
        net.concat7 = L.Concat(net.depooling3, net.block2)

        net.conv8 = conv_2(net.concat7, 64)
        net.block8 = add_layer(net.concat7, net.conv8)
        net.depooling4 = deconv(net.block8, 16)
        net.concat8 = L.Concat(net.depooling4, net.block1)

        net.conv9 = conv_1(net.concat8, 32)
        net.block9 = add_layer(net.concat8, net.conv9)
        net.output = fcn(net.block9,2)
        #train
        if phase=="TRAIN":
            net.data_flat = L.Reshape(net.output, shape=dict(dim=[0,2,1048576]))
            net.label_flat = L.Reshape(net.label, shape=dict(dim=[0,1,1048576]))
            net.softmax_out = L.Softmax(net.data_flat)
        elif phase=="TEST":
            net.data_flat = L.Reshape(net.output, shape=dict(dim=[1, 2, 1048576]))
            net.softmax_out = L.Softmax(net.data_flat)
            net.labelmap = L.Reshape(net.softmax_out, shape=dict(dim=[1,2,128,128,64]))

        return net.to_proto()

if __name__ == "__main__":
    net = SegNet()
    netstr = net.layers_proto(2,"TRAIN")
    print netstr