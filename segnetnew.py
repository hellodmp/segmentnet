import caffe
from caffe import layers as L
from caffe import params as P

def conv(bottom, num_output=16, kernel_size=3, stride=1, pad=1):
    conv = L.Convolution(bottom, num_output=num_output, kernel_size=kernel_size, stride=stride, pad=pad,
                         param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                         weight_filler=dict(type='msra', variance_norm=2),
                         bias_filler=dict(type='constant', value=0))
    #conv_bn = L.BatchNorm(conv, use_global_stats=False, in_place=True)
    return conv

def resblock(bottom, num_output=16, kernel_size=3, stride=1, pad=1):
    conv1 = L.Convolution(bottom, num_output=num_output, kernel_size=kernel_size, stride=stride, pad=pad,
                         param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                         weight_filler=dict(type='msra', variance_norm=2),
                         bias_filler=dict(type='constant', value=0))
    bn1 = L.BatchNorm(conv1, use_global_stats=False, in_place=True)
    relu1 = L.PReLU(bn1, in_place=True)
    conv2 = L.Convolution(relu1, num_output=num_output, kernel_size=kernel_size, stride=stride, pad=pad,
                          param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                          weight_filler=dict(type='msra', variance_norm=2),
                          bias_filler=dict(type='constant', value=0))
    bn2 = L.BatchNorm(conv2, use_global_stats=False, in_place=True)
    eltwise = L.Eltwise(bottom, bn2, eltwise_param=dict(operation=1))
    relu2 = L.PReLU(eltwise, in_place=True)
    return conv1

def down_conv(bottom, num_output):
    conv = L.Convolution(bottom, num_output=num_output, kernel_size=2, stride=2, pad=0,
                         param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                         weight_filler=dict(type='msra', variance_norm=2),
                         bias_filler=dict(type='constant', value=0))
    return conv

def deconv(bottom, num_output):
    conv = L.Deconvolution(bottom,
                           param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                           convolution_param=dict(num_output=num_output, kernel_size=2, stride=2,pad=0,
                                                  weight_filler=dict(type="msra",variance_norm=2),
                                                  bias_filler=dict(type="constant",value=0)))
    #relu = L.PReLU(conv, in_place=True)
    return conv

def add_layer(bottom1, bottom2):
    eltwise = L.Eltwise(bottom1, bottom2, eltwise_param=dict(operation=1))
    return eltwise

def fcn(bottom, num_output=2, kernel_size=1, stride=1, pad=0):
    conv1 = L.Convolution(bottom, num_output=num_output, kernel_size=kernel_size, stride=stride, pad=pad,
                         param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                         weight_filler=dict(type='msra', variance_norm=2),
                         bias_filler=dict(type='constant', value=0))
    #conv_bn = L.BatchNorm(conv, use_global_stats=False, in_place=True)
    return conv1

class SegNet(object):

    def layers_proto(self, batch_size, phase='TRAIN', chan_num=32):
        net = caffe.NetSpec()

        net.data, net.label = L.Data(batch_size=batch_size, ntop=2)
        net.conv1 = conv(net.data, chan_num)
        chan_num = chan_num * 2
        net.pooling1 = down_conv(net.conv1, chan_num)
        net.res1 = resblock(net.pooling1, chan_num)

        chan_num = chan_num * 2
        net.pooling2 = down_conv(net.res1, chan_num)
        net.res2 = resblock(net.pooling2, chan_num)

        chan_num = chan_num * 2
        net.pooling3 = down_conv(net.res2, chan_num)
        net.res3 = resblock(net.pooling3, chan_num)

        chan_num = chan_num * 2
        net.pooling4 = down_conv(net.res3, chan_num)
        net.res4 = resblock(net.pooling4, chan_num)

        chan_num = chan_num * 2
        net.pooling5 = down_conv(net.res4, chan_num)
        net.res5 = resblock(net.pooling5, chan_num)

        chan_num = chan_num / 2
        net.depooling4 = deconv(net.res5, chan_num)
        net.deadd4 = add_layer(net.depooling4,net.res4)
        net.deres4 = resblock(net.deadd4, chan_num)

        chan_num = chan_num / 2
        net.depooling3 = deconv(net.deres4, chan_num)
        net.deadd3 = add_layer(net.depooling3,net.res3)
        net.deres3 = resblock(net.deadd3, chan_num)

        chan_num = chan_num / 2
        net.depooling2 = deconv(net.deres3, chan_num)
        net.deadd2 = add_layer(net.depooling2,net.res2)
        net.deres2 = resblock(net.deadd2, chan_num)

        chan_num = chan_num / 2
        net.depooling1 = deconv(net.deres2, chan_num)
        net.deadd1 = add_layer(net.depooling1, net.res1)
        net.deres1 = resblock(net.deadd1, chan_num)

        chan_num = chan_num / 2
        net.depooling0 = deconv(net.deres1, chan_num)
        net.deadd0 = add_layer(net.depooling0, net.conv1)
        net.deres0 = resblock(net.deadd0, chan_num)
        net.output = fcn(net.deres0,2)
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
    chan_num = 64
    netstr = net.layers_proto(2,"TRAIN", chan_num)
    print netstr