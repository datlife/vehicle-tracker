import time
import json

from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from yolo.net import TFNet

# ##########
# FOR CNN
# ##########
# Image Size
HEIGHT = 720
WIDTH = 1280
CHANNELS = 3
# output dimension from RNN
OUTPUT_DIM = 1  # Steer, throttle, brake
# HYPER-PARAMETER
LOAD_WEIGHT = True
LEARN_RATE = 0.001
KEEP_PROP = 0.4
EPOCHS = 1
BATCH_SIZE = 4
INIT = 'he_uniform'


class SupportVectorMachineClassifier(object):

    def __init__(self):
        self.svc = Pipeline([('scaling', StandardScaler()), ('classification', LinearSVC(loss='hinge')),])

    def train(self, x_train, y_train):
        print("\nStarting to train vehicle detection classifier.")
        start = time.time()
        self.svc.fit(x_train, y_train)
        print("Completed training in {:5f} seconds.\n".format(time.time() - start))

    def score(self, x_test, y_test):
        print("Testing accuracy:")
        scores = self.svc.score(x_test, y_test)
        print("Accuracy {:3f}%".format(scores))
        return scores

    def predict(self, feature):
        return self.svc.predict(feature)

    def decision_function(self, feature):
        return self.svc.decision_function(feature)


class YOLOV2(object):
    def __init__(self, cfg_path="cfg/tiny-yolo-voc.cfg", weight_path="bin/tiny-yolo-voc.weights"):
        option = {"model": cfg_path, "load": weight_path, "threshold": 0.1}
        self.model = TFNet(option) # From DarkFlow repo

    def train(self, img):
        raise NotImplemented

    def predict(self, img):
        result = self.model.return_predict(img)
        boxes = self._convert_json_to_points(result)
        return boxes

    def _convert_json_to_points(self, jfile):
        boxes = []
        for i in jfile:
            i = str(i).replace("\'", "\"")
            data = json.loads(i)
            top = (int(data['topleft']['x']), int(data['topleft']['y']))
            bot = (int(data['bottomright']['x']), int(data['bottomright']['y']))
            boxes.append((top, bot))
        return boxes


from keras.models import Model
from keras.layers import Convolution2D, Dense, Dropout, BatchNormalization, AveragePooling2D
from keras.layers import Input, Flatten, merge, Lambda, Activation
from keras.optimizers import Adam
from keras.regularizers import l2


class ResNet(object):
    def __init__(self, input_shape=(720, 1280, 3), pre_trained_weights=None):
        self.model = self.build_model(input_shape=input_shape)
        if pre_trained_weights is not None:
            print("Loading pre-trained weights...")
            self.model.load_weights(pre_trained_weights, by_name=True)
            print("Loaded pre-trained weights")

    def train(self, x, y, epochs=EPOCHS, batch_size=BATCH_SIZE, learning_rate=LEARN_RATE):
        model = self.model
        model.compile(loss='binary_crossentropy', optimizer=Adam(lr=learning_rate), metrics=['accuracy'])
        model.fit(x, y, batch_size=batch_size, nb_epoch=epochs, shuffle=True)

    def predict(self, x):
        self.model.predict(x, batch_size=1)

    def _fit_generator(self):
        '''
        Generate Image
        '''
    def build_model(self, input_shape,
                    layer1_params=(5, 32, 2),
                    res_layers_params=(3, 16, 3),
                    init='he_uniform', reg=0.01):
        '''
        Return a ResNet Pre-Activation Model. An Implementation of He et al in this paper:
        https://arxiv.org/pdf/1603.05027.pdf
        '''
        #  Filter Config.
        # ##################################################################################
        sz_L1_filters, nb_L1_filters, stride_L1 = layer1_params
        sz_res_filters, nb_res_filters, nb_res_stages = res_layers_params
        sz_pool_fin = (input_shape[0]) / stride_L1

        #  INPUT LAYERS
        # ###################################################################################
        frame = Input(shape=(HEIGHT, WIDTH, CHANNELS), name='cifar')

        # VISION MODEL - USING CNN
        # ####################################################################################
        x = Lambda(lambda image: image / 255.0 - 0.5, input_shape=(HEIGHT, WIDTH, CHANNELS))(frame)
        x = Convolution2D(nb_L1_filters, sz_L1_filters, sz_L1_filters, border_mode='same',
                          subsample=(stride_L1, stride_L1),
                          init=init, W_regularizer=l2(reg), bias=False, name='conv0')(x)
        x = BatchNormalization(axis=1, name='bn0', mode=2)(x)
        x = Activation('relu', name='relu0')(x)
        x = Dropout(KEEP_PROP)(x)

        # Bottle Neck Layers
        for stage in range(1, nb_res_stages + 1):
            x = self._bottleneck_layer(x, (nb_L1_filters, nb_res_filters), sz_res_filters, stage, init=init, reg=reg)

        x = BatchNormalization(axis=1, name='bnF', mode=2)(x)
        x = Activation('relu', name='reluF')(x)
        x = Dropout(KEEP_PROP)(x)
        x = AveragePooling2D((sz_pool_fin, sz_pool_fin), name='avg_pool')(x)
        x = Flatten(name='flat')(x)

        x = Dense(1024, name='fc1', activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(512, name='fc2', activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(256, name='fc3', activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(OUTPUT_DIM, name='output_tracker', activation='sigmoid')(x) ## Because this is binary classification

        model = Model(input=frame, output=x)
        model.summary()
        return model

    def _bottleneck_layer(self, input_tensor, nb_filters, filter_sz, stage,
                          init='glorot_normal', reg=0.0, use_shortcuts=True):
        '''

        :param input_tensor:
        :param nb_filters:   number of filters in Conv2D
        :param filter_sz:    filter size for Conv2D
        :param stage:        current position of the block (used a loop when get called)
        :param init:         initialization type
        :param reg:          regularization type
        :param use_shortcuts:
        :return:
        '''
        nb_in_filters, nb_bottleneck_filters = nb_filters

        bn_name = 'bn' + str(stage)
        conv_name = 'conv' + str(stage)
        relu_name = 'relu' + str(stage)
        merge_name = '+' + str(stage)

        # batch-norm-relu-conv, from nb_in_filters to nb_bottleneck_filters via 1x1 conv
        if stage > 1:  # first activation is just after conv1
            x = BatchNormalization(axis=1, name=bn_name + 'a', mode=2)(input_tensor)
            x = Activation('relu', name=relu_name + 'a')(x)
            x = Dropout(KEEP_PROP)(x)
        else:
            x = input_tensor

        x = Convolution2D(nb_bottleneck_filters, 1, 1,
                          init=init, W_regularizer=l2(reg), border_mode='valid',
                          bias=False, name=conv_name + 'a')(x)

        # batch-norm-relu-conv, from nb_bottleneck_filters to nb_bottleneck_filters via FxF conv
        x = BatchNormalization(axis=1, name=bn_name + 'b', mode=2)(x)
        x = Activation('relu', name=relu_name + 'b')(x)
        x = Convolution2D(nb_bottleneck_filters, filter_sz, filter_sz, border_mode='same',
                          init=init, W_regularizer=l2(reg), bias=False, name=conv_name + 'b')(x)

        # batch-norm-relu-conv, from nb_in_filters to nb_bottleneck_filters via 1x1 conv
        x = BatchNormalization(axis=1, name=bn_name + 'c', mode=2)(x)
        x = Activation('relu', name=relu_name + 'c')(x)
        x = Dropout(KEEP_PROP)(x)

        x = Convolution2D(nb_in_filters, 1, 1,
                          init=init, W_regularizer=l2(reg),
                          name=conv_name + 'c')(x)
        # merge
        if use_shortcuts:
            x = merge([x, input_tensor], mode='sum', name=merge_name)

        return x