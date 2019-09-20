import mxnet as mx

def lenet():
	data = mx.sym.var('data')
	# first conv layer
	conv1 = mx.sym.Convolution(data=data, kernel=(5,5), num_filter=20,name='conv1')
	tanh1 = mx.sym.Activation(data=conv1, act_type="tanh")
	pool1 = mx.sym.Pooling(data=tanh1, pool_type="max", kernel=(2,2), stride=(2,2))
	# second conv layer
	conv2 = mx.sym.Convolution(data=pool1, kernel=(5,5), num_filter=50,name='conv2')
	tanh2 = mx.sym.Activation(data=conv2, act_type="tanh")
	pool2 = mx.sym.Pooling(data=tanh2, pool_type="max", kernel=(2,2), stride=(2,2))
	# first fullc layer
	flatten = mx.sym.flatten(data=pool2)
	fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=120,name='fc1')
	tanh3 = mx.sym.Activation(data=fc1, act_type="tanh")
	return tanh3