import keras
from keras import backend as K

from keras.layers import Dense
from keras.models import Sequential
from support import *

def get_model(data_type):
    # Model parameters and other things based upon the type of data
    op_dims = 4
    op_activ = 'softmax'
    krnl_init = keras.initializers.RandomNormal(mean = 0.0, stddev = 0.05, seed = 121)
    bias_init = keras.initializers.Zeros()
    if (data_type == 'encoded'):
        # NOT IN USE!!!!
        ip_dims = 6
        l1_units = 200
        l2_units = 200
        activ = 'relu'
        lrn_rt = 1.0e-06
    
        #https://stackoverflow.com/questions/59737875/keras-change-learning-rate
        opt = keras.optimizers.Adam(lr = lrn_rt)
    elif (data_type == 'sparse'):
        ip_dims = 70 
        l1_units = 200
        l2_units = 20
        activ = 'relu'
        lrn_rt = 1.0e-6
    
        # RMSprop optimizer
        #https://stackoverflow.com/questions/59737875/keras-change-learning-rate
        #opt = keras.optimizers.RMSprop(
        #    learning_rate = lrn_rt, rho = 0.9, momentum = 0.0, epsilon = 1e-07, 
        #    centered = False, name = 'RMSprop')
    
        # Adadelta optimizer
        #opt = keras.optimizers.Adadelta(
        #    lr = lrn_rt, rho = 0.95, epsilon = 1e-07)

        # Adam w/wo amsgrad
        # opt = keras.optimizers.Adam(lr = lrn_rt, amsgrad=True)

        # Adagrad
        #opt = keras.optimizers.Adagrad(
        #    learning_rate=lrn_rt, initial_accumulator_value=0.1, epsilon=1e-07,
        #    name='Adagrad')

        # Nadam
        opt = keras.optimizers.Nadam(
            lr=lrn_rt, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name = 'nadam')
    else:
        import sys
        sys.exit(0)

    # creates a generic neural network architecture
    model = Sequential()
    
    model.add(Dense(units = l1_units,
                    input_dim = ip_dims, 
                    activation = activ,
                    kernel_initializer = krnl_init, 
                    bias_initializer = bias_init))
    model.add(Dense(units = l2_units, 
                    activation= activ,
                    kernel_initializer = krnl_init,  
                    bias_initializer = bias_init))
    
    # output layer
    model.add(Dense(units = op_dims, 
                    activation = op_activ,
                    kernel_initializer = krnl_init,  
                    bias_initializer = bias_init))
    
    # compile the model using traditional Machine Learning losses and optimizers
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model


def get_pre_proc_info(data_type):
    if (data_type == 'encoded'):
        get_features = get_pos
        from sklearn.preprocessing import RobustScaler
        pre_proc_features = RobustScaler() 
    else:
        get_features = get_same
        from sklearn.preprocessing import StandardScaler
        pre_proc_features = StandardScaler(with_mean = False)
    return get_features, pre_proc_features
