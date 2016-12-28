# Example of how to use
import cnnmodel as model
import numpy as np

DEPLOY_FILE = 'deploy/SqueezeNet1_1_deploy.prototxt'

cnn = model.cnnmodel()
cnn.open(DEPLOY_FILE)
cnn.set_input_size(227,227,3,1)
cnn.run(True)

data = cnn.get_ops()
names = cnn.get_names()
types = cnn.layer_type

print "Total ops: ", sum(data)/1e9, "GFLOP"
print "Parameters: ", cnn.parameters/1e6, " M"
print "Activation maps:", sum(cnn.activation)/1e6, " M"
