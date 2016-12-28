import caffe_pb2  # you created this module with the protoc command
from google.protobuf.text_format import Merge
import matplotlib.pyplot as plt
import numpy as np
import math


# http://dgschwend.github.io/netscope/quickstart.html
# https://github.com/dgschwend/netscope/blob/gh-pages/src/analyzer.coffee

"""
Definitions
    W1 : input volume width
    W2 : input volume hight
    D1 : input volume depth

    W1 : output volume width
    W2 : output volume hight
    D1 : output volume depth

    Sv : vertical stride
    Sh : horizontal stride

    Pv  : Zero padding vertical
    Ph  : Zero padding horizontal

    K  : Number of filters
    Fw : Filter width
    Fh : Filter height

    B  : Batch size


    A layer takes input through bottom connections and makes output through top connections.

    A split layer is created by copying the bottom Blob into multiple top Blobs
    to be used by multiple consuming layers.
"""

class cnnmodel:
    def __init__(self):
        self.net = []
        self.height = 0
        self.width = 0
        self.dimension = 0
        self.batch = 0


        self.matrix_dim = []
        self.dim_size = []
        self.time = 0

        # Network parameters
        self.parameters = 0
        self.layer_type = []
        self.ops_list = []
        self.name_list = []
        self.add = []
        self.mac = []
        self.activation = []
        self.conv_ops = [] # Some networks only calculate ops or MACs in conv layers
        self.conv_and_fc_layers = 0


        # Calibration parameters
        self.alpha = 1              # convolution
        self.beta = 1               # Fully connected layers
        self.gamma = 1              # Relu
        self.delta = 1              # LRN
        self.epsilon = 1            # Pooling
        self.zeta = 1               # Batchnorm
        self.eta = 1                # scale
        self.theta = 1              # dropout
        self.iota = 1               # eltwise
        self.kappa = 1              # Concatenate
        self.mu = 1                 # Split

        self.flops = 1 #4*1.2*1e3      # Number of cores * frequency * ops per cycle


    def __get_type(self,type):
        if str(type).lower() == "convolution" or type == 4:
            return "convolution"

        elif str(type).lower() == "relu" or type == 18:
            return "relu"

        elif str(type).lower() == "pooling" or type == 17:
            return "pooling"

        elif str(type).lower() == "softmax" or type == 20:
            return "softmax"

        elif str(type).lower() == "lrn" or type == 15:
            return "lrn"

        elif str(type).lower() == "innerproduct" or type == 14:
            return "innerproduct"

        elif str(type).lower() == "dropout" or type == 6:
            return "dropout"

        elif str(type).lower() == "concat" or type == 3:
            return "concat"

        elif str(type).lower() == "batchnorm":
            return "batchnorm"

        elif str(type).lower() == "eltwise" or type == 25:
            return "eltwise"

        elif str(type).lower() == 'scale':
            return 'scale'

        elif str(type).lower() == 'power':
            return 'power'

        else:
            return type



    def open(self,path):
        self.net = caffe_pb2.NetParameter()
        Merge((open(path, 'r').read()), self.net)

    # Not implemented!
    def dropout(self):
       return 0


    def set_input_size(self,H,W,D,B=1):
        self.height = H
        self.width = W
        self.dimension = D
        self.batch = B

    def get_names(self):
        return self.name_list

    def get_ops(self):
        return self.ops_list

    def run(self, print_dim=False):
        H1 = H2 =  0
        W1 = W2 =  0
        D1 = D2 = 0
        B = 0
        params = []
        self.parameters = 0
        self.layer_type = []
        self.name_list = []
        self.ops_list = []
        self.ops_conv_fc = []
        self.activation = []
        self.mac = []
        self.conv_and_fc_layers = 0

        # Check if we have new or old format
        if self.net.layer:
            layers = self.net.layer
            if str(layers[0].type).lower() != 'input':
                if self.height == 0 | self.width == 0 | self.dimension == 0 | self.batch == 0:
                    print "Input not specified!"
                    exit(1)

                else:
                    H1 = self.height
                    W1 = self.width
                    D1 = self.dimension
                    B = self.batch
                    self.layer_type.append('input')
                    self.name_list.append("data")
                    self.ops_list.append(10)
                    self.mac.append(0)
                    self.activation.append(B * W1 * H1 * D1)
                    params.append(0)


        else:
            layers = self.net.layers
            self.ops_list.append(10)
            self.layer_type.append('input')
            self.name_list.append("data")
            if self.height == 0 | self.width == 0 | self.dimension == 0 | self.batch == 0:
                print "Input not specified!"
                exit(1)
            else:
                H1 = self.height
                W1 = self.width
                D1 = self.dimension
                B = self.batch
                self.activation.append(B * W1 * H1 * D1)
                params.append(0)

        bottoms = np.array([])

        # Store dimensions for each layer!
        dict = {}
        dict['data'] = [H1, W1, D1]
        dict['input'] = [H1, W1, D1]
        bottom_layers = []

        # Save bottom (inouts) to check for split layers
        for layer in layers:
            if str(layer.top[0]) != 'data': # To support the old and new format.
                if str(layer.top) != str(layer.bottom):
                    bottoms = np.append(bottoms, layer.bottom[0])

        for layer in layers:

            if str(layer.top[0]) != 'data':  # To support the old and new format.
                # TODO -> Not the best solution!
                if layer.top[0] == layer.bottom[0]:
                    if self.layer_type[-1] == 'split':
                        self.layer_type.pop()
                        self.name_list.pop()
                        self.ops_list.pop()
                        self.mac.pop()
                        self.activation.append(0)
                        params.append(0)


            self.layer_type.append(self.__get_type(layer.type))
            self.name_list.append(layer.name)

            #print "(%f, %f, %f, %f)" % (B, D1, H1, W1)
            #print layer.name

            if str(layer.type).lower() == 'input':
                input_dim = layer.input_param.shape[0].dim._values
                self.ops_list = []
                self.layer_type = []
                self.name_list = []
                B = float(input_dim[0])
                D1 = float(input_dim[1])
                H1 = float(input_dim[2])
                W1 = float(input_dim[3])
                self.ops_list.append(0)
                self.mac.append(0)
                self.layer_type.append('input')
                self.name_list.append(layer.name)
                self.activation.append(B * W1 * H1 * D1)
                params.append(0)


            elif self.__get_type(layer.type) == 'convolution':
                param = layer.convolution_param

                # Get padding
                pad_h = param.pad_h if param.pad_h else (param.pad._values[0] if param.pad._values else 0)
                pad_w = param.pad_w if param.pad_w else (param.pad._values[0] if param.pad._values else 0)

                # Get stride
                stride_h = param.stride_h if param.stride_h else (param.stride._values[0] if param.stride._values else 1)
                stride_w = param.stride_w if param.stride_w else (param.stride._values[0] if param.stride._values else 1)


                # Check group
                #g = param.group if param.group else 1
                g = 1

                print layer.bottom[0]
                #print split
                self.conv_and_fc_layers += 1

                kernel_h = param.kernel_h if param.kernel_h else (param.kernel_size[0] if param.kernel_size else 0)
                kernel_w = param.kernel_w if param.kernel_w else (param.kernel_size[0] if param.kernel_size else 0)

                # Number of filters
                K = param.num_output

                # get bottom input size
                [H1, W1, D1] = dict[str(layer.bottom[0])]

                # Caffe rounds down in the conv layers
                H2 = math.floor((float(H1) - float(kernel_h) + 2.0 * float(pad_h)) / float(stride_h) + 1.0)
                W2 = math.floor((float(W1) - float(kernel_w) + 2.0 * float(pad_w)) / float(stride_w) + 1.0)
                D2 = K

                # computation
                # matrix mult
                # M = K
                N = D1 * kernel_w * kernel_h
                L = B * H2 * W2
                flops = 2 * K * N * L - K * L
                self.ops_list.append((flops/g))  # bias not included
                self.mac.append(B * kernel_w * kernel_h * W2 * H2 * D1 * D2)
                self.conv_ops.append(flops/g)

                # memory
                self.parameters += (kernel_w * kernel_h * D1 * D2)
                params.append(kernel_w * kernel_h * D1 * D2)
                self.activation.append(B * W2 * H2 * D2)

                # Layer count
                #self.conv_and_fc_layers += 1

                # New output sizes
                H1 = H2
                W1 = W2
                D1 = D2

            # Relu and dropout has the same complexity
            elif self.__get_type(layer.type) == 'relu' or self.__get_type(layer.type) == 'dropout':
                # get bottom input size
                [H1, W1, D1] = dict[str(layer.bottom[0])]

                # computation
                self.ops_list.append(B * H1 * W1 * D1)
                self.mac.append(0)
                # memory
                self.activation.append(B * H1 * W1 * D1)
                params.append(0)



            elif self.__get_type(layer.type) == 'pooling':
                param = layer.pooling_param

                # Get stride
                stride_h = param.stride_h if param.stride_h else (param.stride if param.stride else 1)
                stride_w = param.stride_w if param.stride_w else (param.stride if param.stride else 1)

                # Get padding
                pad_h = param.pad_h if param.pad_h else (param.pad if param.pad else 0)
                pad_w = param.pad_w if param.pad_w else (param.pad if param.pad else 0)

                # Get kernel size
                kernel_h = param.kernel_h if param.kernel_h else param.kernel_size
                kernel_w = param.kernel_w if param.kernel_w else param.kernel_size

                # get bottom input size
                [H1, W1, D1] = dict[str(layer.bottom[0])]
                #print "Layer:", layer.name
                #print "Bottom Layer: ", dict[str(layer.bottom[0])]

                H2 = np.ceil((float(H1) - float(kernel_h) + 2.0 * float(pad_h)) / float(stride_h) + 1.0)
                W2 = np.ceil((float(W1) - float(kernel_w) + 2.0 * float(pad_w)) / float(stride_w) + 1.0)
                #print D1
                if param.global_pooling:
                    # TODO CHECK THIS!!!!
                    W1 = H1 = 1
                    self.ops_list.append(B * W1 * H1 * D1)
                    self.mac.append(0)
                else:
                    self.ops_list.append(D1 * H2 * W2 * B * kernel_h * kernel_w)  # max operation not included
                    self.mac.append(0)
                    H1 = H2
                    W1 = W2

                # memory
                self.activation.append(B * H1 * W1 * D1)
                params.append(0)


            elif self.__get_type(layer.type) == 'lrn':
                # Parmaeters:
                # Local size : l
                # alpha      :
                # beta       :
                param = layer.lrn_param
                l = param.local_size

                # get bottom input size
                [H1, W1, D1] = dict[str(layer.bottom[0])]

                # ACROSS_CHANNELS = 0
                # WITHIN_CHANNEL = 1
                print param.norm_region

                # Computations
                mac = B * H1 * W1 * D1 * l
                additions = B * H1 * W1 * D1
                divisions = B * H1 * W1 * D1 * 2
                exp = B * H1 * W1 * D1
                self.mac.append(mac)
                self.ops_list.append(2*mac + additions + divisions + exp)

                # Memory
                self.parameters += 2
                params.append(2)
                self.activation.append(B * H1 * W1 * D1)


            elif self.__get_type(layer.type) == 'innerproduct':
                # Get the parameters
                out_dim = layer.inner_product_param.num_output

                # get bottom input size
                [H1, W1, D1] = dict[str(layer.bottom[0])]

                # dimensions
                H2 = 1
                W2 = 1
                D2 = out_dim

                # computation
                self.ops_list.append(2*W1 * H1 * D1 * D2)
                self.mac.append(W1 * H1 * D1 * D2)
                # memory
                self.parameters += W1 * H1 * D1 * D2
                params.append(W1 * H1 * D1 * D2)
                self.activation.append(B * W2 * H2 * D2)

                # Layer count

                self.conv_and_fc_layers += 1

                H1 = H2
                W1 = W2
                D1 = D2



            elif self.__get_type(layer.type) == 'concat':
                #print len(layer.bottom)
                #print layer.bottom
                if len(layer.bottom) > 1:
                    D1 = 0
                    ops = 0
                    for i in range(len(layer.bottom)):
                        [tmp_h, tmp_w, tmp_d] = dict[str(layer.bottom[i])]
                        ops += tmp_h * tmp_w * tmp_d
                        #print tmp_d
                        D1 += tmp_d

                        if len(layer.bottom) > 1:
                            if i == 0:
                                old_h = tmp_h
                                old_w = tmp_w
                                old_d = tmp_d
                            else:
                                if old_h != tmp_h or old_w != tmp_w:
                                    print "Error!!!! dim does not fit in concat!"
                                    print self.name_list[-1]
                                    print dict[layer.bottom[i]]
                                    print dict[layer.bottom[0]]
                                    #exit(0)
                                else:
                                    old_h = tmp_h
                                    old_w = tmp_w

                # Ops ???
                self.ops_list.append(ops) # TODO fix Dummy ops!!!
                self.mac.append(0)
                # Memory
                self.activation.append(W1 * H1 * D1)
                params.append(0)

            elif self.__get_type(layer.type) == "softmax" or self.__get_type(
                    layer.type) == "softmaxwithloss" or self.__get_type(layer.type) == "softmax_loss":
                # get bottom input size
                [H1, W1, D1] = dict[str(layer.bottom[0])]

                # Computations
                exp = B * W1 * H1 * D1
                additions = B * W1 * H1 * D1
                divisions = B * W1 * H1 * D1
                self.ops_list.append(exp + additions + divisions)
                self.mac.append(0)
                # memory
                self.activation.append(B * H1 * W1 * D1)
                params.append(0)

            elif self.__get_type(layer.type) == 'batchnorm':
                # get bottom input size
                [H1, W1, D1] = dict[str(layer.bottom[0])]

                additions = W1 * H1 * D1
                divisions = W1 * H1 * D1
                self.ops_list.append(additions+divisions)
                self.mac.append(0)
                params.append(0)

                # memory
                self.parameters += D1 * 2
                self.activation.append(B * W1 * H1 * D1)

            elif self.__get_type(layer.type) == 'eltwise':
                param = layer.eltwise_param

                # get bottom input size
                [H1, W1, D1] = dict[str(layer.bottom[0])]

                # Check for errors
                for i in range(len(layer.bottom)):
                    [tmp_h, tmp_w, tmp_d] = dict[str(layer.bottom[i])]

                    if len(layer.bottom) > 1:
                        if i == 0:
                            old_h = tmp_h
                            old_w = tmp_w
                            old_d = tmp_d
                        else:
                            if old_h != tmp_h or old_w != tmp_w or old_d != tmp_d:
                                print "Error!!!! dim does not fit in eltwise!"
                                print self.name_list[-1]
                                print dict[layer.bottom[i]]
                                print dict[layer.bottom[0]]
                                #exit(0)
                            else:
                                old_h = tmp_h
                                old_w = tmp_w
                                old_d = tmp_d



                # dimensions

                # TODO check input dimensions

                # computation
                #ops_type = str(param.operation).lower()

                #if ops_type == 'sum':
                #    ops = W1 * H1 * D1 # additions
                #elif ops_type == 'max':
                #    ops = W1 * H1 * D1 # computations
                #elif ops_type == 'prod':
                #    ops = W1 * H1 * D1 # MAC opperations
                ops = W1 * H1 * D1
                self.ops_list.append(ops)
                self.mac.append(0)
                self.activation.append(W1 * H1 * D1)
                params.append(0)

            elif self.__get_type(layer.type) == 'scale':
                # scale layer use activation memory and does multiplies
                # We have two parameters alpha and beta

                # get bottom input size
                [H1, W1, D1] = dict[str(layer.bottom[0])]

                # computation: scale = multiplication
                # Bias ?? TODO check!
                self.ops_list.append(W1 * H1 * D1 * B)
                # memory
                self.activation.append(W1 * H1 * D1 * B)
                params.append(0) # TODO fix

            elif self.__get_type(layer.type) == 'power':
                # Power layer
                param = layer.power_param

                # get bottom input size
                [H1, W1, D1] = dict[str(layer.bottom[0])]


                # void caffe_add_scalar(const int N, const float alpha, float* Y) {
                # for (int i = 0; i < N; ++i) {
                # Y[i] += alpha;
                # }
                if param.scale:
                    adds = W1 * H1 * D1 * B
                else:
                    adds = 0

                if param.power:
                    mult = param.power * W1 * H1 * D1 * B
                else:
                    mult = 0

                if param.scale:
                    mult_s = W1 * H1 * D1 * B
                else:
                    mult_s = 0


                self.ops_list.append(adds + mult + mult_s)
                self.mac.append(0)
                # memory
                self.activation.append(W1 * H1 * D1)
                params.append(0) # TODO fix

            else:
                print "WARNING! Unsupported layer: ", self.__get_type(layer.type)
                self.ops_list.append(0)
                self.activation.append(0)
                self.mac.append(0)
                params.append(0)


            # Check for split layers
            # TODO This might not work all the time!
            check = np.where(bottoms == layer.top[0])
            if len(check[0]) >= 2: # and layer.top[0] == layer.bottom[0]:
                self.layer_type.append("split")
                self.name_list.append(str(layer.top[0]) + '_' + layer.name + '_0_split')
                self.ops_list.append(1)
                self.mac.append(0)
                #print "(%f, %f, %f, %f)" % (B, D1, H1, W1)
                dict[str(layer.top[0]) + '_' + layer.name + '_0_split'] = [H1, W1, D1]

            # Save layer dim D1
            dict[str(layer.name)] = [H1, W1, D1]
            # Just in case the name differs
            dict[str(layer.top[0])] = [H1, W1, D1]

        # Calculate total time
        self.time = np.array(np.array(self.ops_list) / (self.flops))


        if print_dim:
            for i in range(len(self.name_list)):
                print self.name_list[i], "\t-\t", dict[str(self.name_list[i])], "\t-\t", self.ops_list[i]/1e6, "G ops\t-\t", params[i], "Parameters"




    def calibate(self,path_to_meass,visualize = False):

        raw_data = open(path_to_meass)
        dataset = np.loadtxt(raw_data, delimiter=",")

        sample_len = len(dataset[1,:])

        if sample_len != len(self.ops_list):
            print "Model data and real data does not have the same size!"
            print "Size model: ", len(self.ops_list)
            print "Size data: ", sample_len
            return

        # Get the different layer types
        types = np.array(self.layer_type)
        # We don't want duplicates
        all_types = np.unique(types)


        for t in range(len(all_types)):
            # Get the indices where we have the type
            index = np.where(types == all_types[t])
            real_time = np.array([])
            ops = np.array([])



            for i in range(len(index[0])):
                #  b index[0][i]
                avg_time =  sum(dataset[:, index[0][i]])/dataset[:, index[0][i]].size

                real_time = np.append(real_time,avg_time)
                ops = np.append(ops,self.ops_list[index[0][i]])

            if all_types[t] == 'relu':
                if len(ops) >= 2:
                    z = np.polyfit(ops, real_time, 1)
                    self.gamma = np.poly1d(z)
                    if visualize:
                        x = np.linspace(min(ops), max(ops))
                        plt.plot(x, self.gamma(x))


            # This assumption may not be correct!
            if all_types[t] == 'lrn':
                if len(ops) >= 2:
                    z = np.polyfit(ops, real_time, 1)
                    self.delta = np.poly1d(z)
                    if visualize:
                        x = np.linspace(min(ops), max(ops))
                        plt.plot(x, self.delta(x))

            if all_types[t] == 'convolution':
                if len(ops) >= 2:
                    z = np.polyfit(ops, real_time, 1)
                    self.aplha = np.poly1d(z)
                    if visualize:
                        x = np.linspace(min(ops), max(ops))
                        plt.plot(x, self.aplha(x))

            if all_types[t] == 'innerproduct':
                if len(ops) >= 2:
                    z = np.polyfit(ops, real_time, 1)
                    self.beta = np.poly1d(z)
                    if visualize:
                        x = np.linspace(min(ops), max(ops))
                        plt.plot(x, self.beta(x))

            if all_types[t] == 'pooling':
                if len(ops) >= 2:
                    z = np.polyfit(ops, real_time, 1)
                    self.epsilon = np.poly1d(z)
                    if visualize:
                        x = np.linspace(min(ops), max(ops))
                        plt.plot(x, self.epsilon(x))

            if visualize:
                plt.plot(ops, real_time, 'ro')
                plt.title(all_types[t])
                # plt.set_xlabel('Operations')
                # plt.set_ylabel('Time')
                plt.show()


        sum_time = dataset.sum(axis=1)
        self.flops = sum(self.ops_list)/(sum_time.sum()/sum_time.size)
        #print self.flops

        #print self.flops
        #print "total time: ", sum_time.sum()/sum_time.size
        #print "Total ops: ", sum(self.ops_list)

    def calibate_full(self, types, ops_list,time, visualize=False):


        # We don't want duplicates
        all_types = np.unique(types)

        for t in range(len(all_types)):
            # Get the indices where we have the type
            index = np.where(types == all_types[t])
            real_time = np.array([])
            ops = np.array([])

            for i in range(len(index[0])):
                #  b index[0][i]
                avg_time = sum(time[index[0][i],:]) / time[index[0][i], :].size

                real_time = np.append(real_time, avg_time)
                ops = np.append(ops, ops_list[index[0][i]])
            print str(all_types[t]).lower()
            if str(all_types[t]).lower() == 'relu':
                if len(ops) >= 2:
                    z = np.polyfit(ops, real_time, 1)
                    self.gamma = np.poly1d(z)
                    if visualize:
                        x = np.linspace(min(ops), max(ops))
                        plt.plot(x, self.gamma(x))

            # This assumption may not be correct!
            elif str(all_types[t]).lower() == 'lrn':
                if len(ops) >= 2:
                    z = np.polyfit(ops, real_time, 1)
                    self.delta = np.poly1d(z)
                    if visualize:
                        x = np.linspace(min(ops), max(ops))
                        plt.plot(x, self.delta(x))

            elif str(all_types[t]).lower() == 'convolution':
                if len(ops) >= 2:
                    z = np.polyfit(ops, real_time, 4)
                    self.aplha = np.poly1d(z)
                    if visualize:
                        x = np.linspace(min(ops), max(ops))
                        plt.plot(x, self.aplha(x))

            elif str(all_types[t]).lower() == 'innerproduct':
                if len(ops) >= 2:
                    z = np.polyfit(ops, real_time, 1)
                    self.beta = np.poly1d(z)
                    if visualize:
                        x = np.linspace(min(ops), max(ops))
                        plt.plot(x, self.beta(x))

            elif str(all_types[t]).lower() == 'pooling':
                if len(ops) >= 2:
                    z = np.polyfit(ops, real_time, 1)
                    self.epsilon = np.poly1d(z)
                    if visualize:
                        x = np.linspace(min(ops), max(ops))
                        plt.plot(x, self.epsilon(x))

            elif str(all_types[t]).lower() == 'batchnorm':
                if len(ops) >= 2:
                    z = np.polyfit(ops, real_time, 1)
                    self.zeta = np.poly1d(z)
                    if visualize:
                        x = np.linspace(min(ops), max(ops))
                        plt.plot(x, self.zeta(x))

            elif str(all_types[t]).lower() == 'scale':
                if len(ops) >= 2:
                    z = np.polyfit(ops, real_time, 1)
                    self.eta = np.poly1d(z)
                    if visualize:
                        x = np.linspace(min(ops), max(ops))
                        plt.plot(x, self.eta(x))

            elif str(all_types[t]).lower() == 'dropout':
                if len(ops) >= 2:
                    z = np.polyfit(ops, real_time, 1)
                    self.theta = np.poly1d(z)
                    if visualize:
                        x = np.linspace(min(ops), max(ops))
                        plt.plot(x, self.theta(x))

            elif str(all_types[t]).lower() == 'eltwise':
                if len(ops) >= 2:
                    z = np.polyfit(ops, real_time, 1)
                    self.iota = np.poly1d(z)
                    if visualize:
                        x = np.linspace(min(ops), max(ops))
                        plt.plot(x, self.iota(x))

            elif str(all_types[t]).lower() == 'concat':
                if len(ops) >= 2:
                    z = np.polyfit(ops, real_time, 1)
                    self.kappa = np.poly1d(z)
                    if visualize:
                        x = np.linspace(min(ops), max(ops))
                        plt.plot(x, self.kappa(x))

            elif str(all_types[t]).lower() == 'split':
                if len(ops) >= 2:
                    self.mu = real_time.mean()
                    if visualize:
                        plt.plot([0,1,2], [self.mu,self.mu,self.mu])

            else:
                print "print unsupported layer: ", str(all_types[t]).lower()



            if visualize:
                plt.plot(ops, real_time, 'ro')
                plt.title(all_types[t])
                # plt.set_xlabel('Operations')
                # plt.set_ylabel('Time')
                plt.show()

        sum_time = time.mean(axis=1)
        print "Time shape: ", time.shape
        print "sum_time shape: ", sum_time.shape
        print "Total time: ", sum_time.sum()
        self.flops = sum(self.ops_list) / sum_time.sum()
        print "Flops: ", self.flops

        # print self.flops
        # print "total time: ", sum_time.sum()/sum_time.size
        # print "Total ops: ", sum(self.ops_list)


    def get_time(self):
        time = []
        for i in range(len(self.name_list)):
            #print self.net.layer[i].type
            if self.layer_type[i] == 'relu':
                time.append((self.gamma(self.ops_list[i])))

            elif self.layer_type[i] == 'lrn':
                time.append(self.delta(self.ops_list[i]))

            elif self.layer_type[i] == 'convolution':
                time.append(self.aplha(self.ops_list[i]))

            elif self.layer_type[i] == 'innerproduct':
                time.append(self.beta(self.ops_list[i]))

            elif self.layer_type[i] == 'pooling':
                time.append(self.epsilon(self.ops_list[i]))

            elif self.layer_type[i] == 'batchnorm':
                time.append(self.zeta(self.ops_list[i]))

            elif self.layer_type[i] == 'scale':
                time.append(self.eta(self.ops_list[i]))

            elif self.layer_type[i] == 'dropout':
                time.append(self.theta(self.ops_list[i]))

            elif self.layer_type[i] == 'eltwise':
                time.append(self.iota(self.ops_list[i]))

            elif self.layer_type[i] == 'concat':
                time.append(self.kappa(self.ops_list[i]))

            elif self.layer_type[i] == 'split':
                time.append(self.ops_list[i]*self.mu)

            else:
                time.append(self.ops_list[i]/self.flops)

        time = np.array(time)
        return time
