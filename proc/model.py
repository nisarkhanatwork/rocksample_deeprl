import keras
from keras import backend as K

from keras.layers import Dense
from keras.models import Sequential
from support import *
import keras.losses
INPUT_SIZE = 56

def get_model(data_type):
    # Model parameters and other things based upon the type of data
    op_dims = 4
    op_activ = 'softmax'
    # krnl_init = keras.initializers.RandomNormal(mean = 0.0, stddev = 0.05, seed = 11211)
    krnl_init = keras.initializers.GlorotUniform(seed = 11211)
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
        ip_dims = INPUT_SIZE
        l1_units = 400
        l2_units = 90
        activ = 'relu'
        lrn_rt = 1.0e-04
    
        # Adagrad
        #opt = keras.optimizers.Adagrad(
        #    learning_rate=lrn_rt, initial_accumulator_value=0.1, epsilon=1e-07,
        #    name='Adagrad')

        # Adadelta optimizer
        #opt = keras.optimizers.Adadelta(
        #    lr = lrn_rt, rho = 0.95, epsilon = 1e-07)

        # RMSprop optimizer
        #https://stackoverflow.com/questions/59737875/keras-change-learning-rate
        # opt = keras.optimizers.RMSprop(
        #    learning_rate = lrn_rt, rho = 0.9, momentum = 0.0, epsilon = 1e-07, 
        #    centered = False, name = 'RMSprop')
    

        # Adam w/wo amsgrad
        opt = keras.optimizers.Adam(lr = lrn_rt, amsgrad=True)


        # Nadam
        # opt = keras.optimizers.Nadam(
        #    lr=lrn_rt, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name = 'nadam')
        
        #opt = keras.optimizers.SGD(
        #       learning_rate=lrn_rt, momentum=0.001, nesterov=True, name='SGD'
        #    )


        # Loss
        # loss = keras.losses.KLDivergence()

        loss = keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error") 

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
    model.add(Dense(units = l1_units,
                    input_dim = ip_dims, 
                    activation = activ,
                    kernel_initializer = krnl_init, 
                    bias_initializer = bias_init))
    model.add(Dense(units = l2_units, 
                    activation= activ,
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
    model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])

    return model


def get_pre_proc_info(data_type):
    if (data_type == 'encoded'):
        get_features = get_pos
        from sklearn.preprocessing import RobustScaler
        pre_proc_features = RobustScaler() 
    else:
        get_features = get_same

        from sklearn.preprocessing import OrdinalEncoder
        pre_proc_features = OrdinalEncoder()
        from sklearn.preprocessing import MultiLabelBinarizer
        pre_proc_features = MultiLabelBinarizer()
        from sklearn.preprocessing import PolynomialFeatures
        pre_proc_features = PolynomialFeatures(4)
        from sklearn.preprocessing import RobustScaler
        pre_proc_features = RobustScaler() 
        from sklearn.preprocessing import StandardScaler
        pre_proc_features = StandardScaler()
    return get_features, pre_proc_features
