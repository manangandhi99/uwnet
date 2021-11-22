from uwnet import *
def conv_net():
    l = [   make_convolutional_layer(32, 32, 3, 8, 3, 2),
            make_activation_layer(RELU),
            make_maxpool_layer(16, 16, 8, 3, 2),
            make_convolutional_layer(8, 8, 8, 16, 3, 1),
            make_activation_layer(RELU),
            make_maxpool_layer(8, 8, 16, 3, 2),
            make_convolutional_layer(4, 4, 16, 32, 3, 1),
            make_activation_layer(RELU),
            make_connected_layer(512, 10),
            make_activation_layer(SOFTMAX)]
    return make_net(l)

def conv_net_batchnorm():
    l = [   make_convolutional_layer(32, 32, 3, 8, 3, 2),
            make_batchnorm_layer(8),
            make_activation_layer(RELU),
            make_maxpool_layer(16, 16, 8, 3, 2),
            make_convolutional_layer(8, 8, 8, 16, 3, 1),
            make_batchnorm_layer(16),
            make_activation_layer(RELU),
            make_maxpool_layer(8, 8, 16, 3, 2),
            make_convolutional_layer(4, 4, 16, 32, 3, 1),
            make_batchnorm_layer(32),
            make_activation_layer(RELU),
            make_connected_layer(512, 10),
            # make_batchnorm_layer(10),
            make_activation_layer(SOFTMAX)]
    return make_net(l)

print("loading data...")
train = load_image_classification_data("cifar/cifar.train", "cifar/cifar.labels")
test  = load_image_classification_data("cifar/cifar.test",  "cifar/cifar.labels")
print("done")
print

print("making model...")
batch = 128
iters = 500
rate = .02
momentum = .9
decay = .005

m = conv_net_batchnorm()
print("training...")
train_image_classifier(m, train, batch, iters, rate, momentum, decay)
print("done")
print

print("evaluating model...")
print("training accuracy: %f", accuracy_net(m, train))
print("test accuracy:     %f", accuracy_net(m, test))

# 7.6 Question: What do you notice about training the convnet with/without batch normalization? How does it affect convergence? How does it affect what magnitude of learning rate you can use? Write down any observations from your experiments:
# TODO: Your answer

'''
conv_net: batch = 128, iters = 500, rate = .1
training accuracy: %f 0.10000000149011612
test accuracy:     %f 0.10000000149011612

conv_net_batchnorm: batch = 128, iters = 500, rate = .1
training accuracy: %f 0.5382999777793884
test accuracy:     %f 0.5349000096321106

conv_net: batch = 128, iters = 500, rate = .05
training accuracy: %f 0.46173998713493347
test accuracy:     %f 0.46070000529289246

conv_net_batchnorm: batch = 128, iters = 500, rate = .05
training accuracy: %f 0.5490800142288208
test accuracy:     %f 0.5357000231742859

conv_net: batch = 128, iters = 500, rate = .02
training accuracy: %f 0.4451200067996979
test accuracy:     %f 0.44519999623298645

conv_net_batchnorm: batch = 128, iters = 500, rate = .02
training accuracy: %f 0.546180009841919
test accuracy:     %f 0.5338000059127808

conv_net: batch = 128, iters = 500, rate = .01
training accuracy: %f 0.392659991979599
test accuracy:     %f 0.3970000147819519

conv_net_batchnorm: batch = 128, iters = 500, rate = .01
training accuracy: %f 0.5578399896621704
test accuracy:     %f 0.545199990272522
''' 

'''
Through the sample runs on the CIFAR dataset above, we see that the convolutional nets with batch
normalization always outperms those without, with higher train/test accuracies. We also see that
for runs with higher learning rates, the convolutional nets with batch normalization perform much
compared to those with lower learning rates. Through this, we can conclude that batch normalization
is better suited to handle higher learning rates. With the same number of iterations taken per run, 
we also see that batch normalization allows for quicker convergences to the minima compared to 
runs without batch normalization.
'''
