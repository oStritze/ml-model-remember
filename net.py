import lasagne
from lasagne.layers import (InputLayer, Conv2DLayer, DenseLayer, ExpressionLayer, PadLayer,
                            GlobalPoolLayer, ElemwiseSumLayer,
                            NonlinearityLayer, batch_norm)
from lasagne.nonlinearities import rectify, softmax
#from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import MaxPool2DLayer as PoolLayer
try:
    from lasagne.layers.dnn import BatchNormDNNLayer as BatchNormLayer
except ImportError:
    from lasagne.layers import BatchNormLayer


def build_resnet(input_var=None, input_shape=(None, 3, 50, 50), n=5, classes=10, final_act=softmax, increase=True, project=True):
    # create a residual learning building block with two stacked 3x3 convlayers as in paper, inherited from
    # https://github.com/Lasagne/Recipes/blob/master/papers/deep_residual_learning/Deep_Residual_Learning_CIFAR-10.py
    def residual_block(l, increase_dim=False, projection=True):
        input_num_filters = l.output_shape[1]
        if increase_dim:
            first_stride = (2, 2)
            out_num_filters = input_num_filters * 2
        else:
            first_stride = (1, 1)
            out_num_filters = input_num_filters

        stack_1 = batch_norm(
            Conv2DLayer(l, num_filters=out_num_filters, filter_size=(3, 3), stride=first_stride, nonlinearity=rectify,
                        pad='same', W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))
        stack_2 = batch_norm(
            Conv2DLayer(stack_1, num_filters=out_num_filters, filter_size=(3, 3), stride=(1, 1), nonlinearity=None,
                        pad='same', W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))

        # add shortcut connections
        if increase_dim:
            if projection:
                # projection shortcut, as option B in paper
                projection = batch_norm(
                    Conv2DLayer(l, num_filters=out_num_filters, filter_size=(1, 1), stride=(2, 2), nonlinearity=None,
                                pad='same', b=None, flip_filters=False))
                block = NonlinearityLayer(ElemwiseSumLayer([stack_2, projection]), nonlinearity=rectify)
            else:
                # identity shortcut, as option A in paper
                identity = ExpressionLayer(l, lambda X: X[:, :, ::2, ::2], lambda s: (s[0], s[1], s[2] // 2, s[3] // 2))
                padding = PadLayer(identity, [out_num_filters // 4, 0, 0], batch_ndim=1)
                block = NonlinearityLayer(ElemwiseSumLayer([stack_2, padding]), nonlinearity=rectify)
        else:
            block = NonlinearityLayer(ElemwiseSumLayer([stack_2, l]), nonlinearity=rectify)

        return block

    # Building the network
    l_in = InputLayer(shape=input_shape, input_var=input_var)

    # first layer, output is 16 x 32 x 32
    l = batch_norm(Conv2DLayer(l_in, num_filters=16, filter_size=(3, 3), stride=(1, 1), nonlinearity=rectify, pad='same',
                               W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))

    # first stack of residual blocks, output is 16 x 32 x 32
    for _ in range(n):
        l = residual_block(l, projection=project)

    # second stack of residual blocks, output is 32 x 16 x 16
    l = residual_block(l, increase_dim=True, projection=project)
    for _ in range(1, n):
        l = residual_block(l)

    # third stack of residual blocks, output is 64 x 8 x 8
    l = residual_block(l, increase_dim=True, projection=project)
    for _ in range(1, n):
        l = residual_block(l)

    # average pooling
    l = GlobalPoolLayer(l)

    # fully connected layer
    network = DenseLayer(
        l, num_units=classes,
        W=lasagne.init.HeNormal(),
        nonlinearity=final_act)

    print(network, type(network))
    return network

def build_cnn(input_var=None, input_shape=(None, 3, 50, 50), n=1, classes=10, final_act=softmax):
    """
    Just re-use Lasagne stuff to reduce complexity and code copy-paste efforts.
    We just build a resnet but without projection == convolutional, as commented above in the code
    """
    #net = build_resnet(input_var, input_shape, n, classes, final_act, False) # just build the resnet but without projection this time 
    
    """
    We use a 5-layer CNN for gender classification on the LFW
    dataset. The first three layers are convolution layers (32 filters in
    the first layer, 64 in the second, 128 in the third) followed by a maxpooling
    operation which reduces the size of convolved features by
    half. Each filter in the convolution layer is 3x3. The convolution
    output is connected to a fully-connected layer with 256 units. The
    latter layer connects to the output layer which predicts gender.
    """
    # Build network as proposed in paper
    
    l_in = InputLayer(shape=input_shape, input_var=input_var)
    l = Conv2DLayer(l_in,
                    num_filters=32,
                    filter_size=3,
                    stride=1,
                    flip_filters=False)
    l = PoolLayer(l,
                    pool_size=2,
                    stride=2,
                    ignore_border=False)
    l = Conv2DLayer(l,
                    num_filters=64,
                    filter_size=3,
                    stride=1,
                    flip_filters=False)
    l = PoolLayer(l,
                    pool_size=2,
                    stride=2,
                    ignore_border=False)                            
    l = Conv2DLayer(l,
                    num_filters=128,
                    filter_size=3,
                    stride=1,
                    flip_filters=False)
    l = PoolLayer(l,
                    pool_size=2,
                    stride=2,
                    ignore_border=False)
    l = DenseLayer(l, num_units=256)
    net = DenseLayer(l, num_units=classes, nonlinearity=softmax)
    #net['probabilites'] = NonlinearityLayer(net['fc5'], softmax) #softmax non-linearity layer

    return net