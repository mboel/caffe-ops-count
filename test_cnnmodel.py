#import matplotlib as mpl
#mpl.use('pgf')
import matplotlib.pyplot as plt
import cnnmodel as model
from matplotlib2tikz import save as tikz_save

import numpy as np

#plt.style.use(['seaborn-pastel'])
plt.style.use(['ggplot'])


#DEPLOY_FILE ='/Users/boel/caffe/models/bvlc_alexnet/deploy.prototxt'
#MODEL = 'data/AlexNet_cpu_.csv'

#DEPLOY_FILE = 'deploy/VGG_ILSVRC_19_layers_deploy.prototxt'
#MODEL = 'data/VGG_ILSVRC_19_layers_cpu_.csv'

#DEPLOY_FILE = 'deploy/VGG_ILSVRC_16_layers_deploy.prototxt'
#MODEL = 'data/VGG_ILSVRC_16_layers_cpu_.csv'


#DEPLOY_FILE = 'deploy/SqueezeNet1_0_deploy.prototxt'
#MODEL = 'data/SqueezeNet1_0_gpu_.csv'


DEPLOY_FILE = 'deploy/SqueezeNet1_1_deploy.prototxt'
MODEL = 'data/SqueezeNet1_1_gpu_.csv'

#DEPLOY_FILE = 'deploy/GoogleNet_deploy.prototxt'
#MODEL = 'data/GoogleNet_gpu_.csv'


#DEPLOY_FILE = 'deploy/ResNet-50-deploy.prototxt'
#MODEL = 'data/ResNet-50_cpu_.csv'

#DEPLOY_FILE = 'deploy/ResNet-101-deploy.prototxt'
#MODEL = 'data/ResNet-101_cpu_.csv'

#DEPLOY_FILE = 'deploy/ResNet-152-deploy.prototxt'
#MODEL = 'data/ResNet-152_cpu_.csv'

#DEPLOY_FILE = 'deploy/nin_deploy.prototxt'
#DEPLOY_FILE = 'deploy/inception_resnet_v2_train_test.prototxt'
#DEPLOY_FILE = 'deploy/inception_v4_train_test.prototxt'


#DEPLOY_FILE = 'deploy/resnet_v2_deploy.prototxt'

#DEPLOY_FILE = 'deploy_vvg_reduced.prototxt'

#DEPLOY_FILE = 'nin_crelu.prototxt'

DEPLOY_FILE = 'deploy/SqueezeNet1_1_deploy.prototxt'
#DEPLOY_FILE = 'models/SqueezeNet_deep_residual_v3/deploy.prototxt'


#DEPLOY_FILE = 'test.prototxt'

cnn = model.cnnmodel()
cnn.open(DEPLOY_FILE)
cnn.set_input_size(227,227,3,1)
cnn.run(True)

data = cnn.get_ops()
names = cnn.get_names()
types = cnn.layer_type

print "Total ops: ", sum(data)/1e9, "GFLOP"
print "Total mac ", sum(cnn.mac)/1e9, "M"
#print "Conv ops ", sum(cnn.conv_ops)/1e9, "M"
print "Parameters: ", cnn.parameters/1e6, " M"
print "Activation maps:", sum(cnn.activation)/1e6, " M"
print len(cnn.get_names())
print cnn.conv_and_fc_layers

exit(0)
#print map(lambda x: str(x), names)


#cnn.calibate(MODEL,True)

labels = ['$28 \\times 28$', '$56\\times 56$', '$84\\times 84$', '$112\\times 112$', '$140\\times 140$', '$168\\times 168$',
          '$196\\times 196$', '$224\\times 224$','$252\\times 252$','$280\\times 280$', '$336\\times 336$']


ticks = [28, 56, 84, 112, 140 ,168, 196, 224, 252, 280, 308, 336]

ops = []
input_size = range(min(ticks),max(ticks))
for i in range(len(input_size)):
    cnn.set_input_size(input_size[i], input_size[i], 3, 1)
    cnn.run()
    ops.append(sum(cnn.get_ops()))

fig = plt.figure(1)
ax = fig.add_subplot(111)
plt.plot(input_size,ops,'.')
plt.xlabel('Image input size')
plt.ylabel('Operations')
#plt.xticks(x, labels, rotation=40)
plt.xticks(ticks, labels,rotation=45)


tikz_save('plots/input_size_vs_complexity.tikz',
          figureheight='\\figureheight',
          figurewidth='\\figurewidth')

plt.show()