import scipy.io as sio
import matplotlib.pyplot as plt
import pickle, os

from pi_net import *

class DoDict(dict):
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self

###########################
## Variables
seed_value_list = [100]
param = DoDict()
param.batch_size = 128
param.nb_epochs = [300,300,400]
param.learning_rate = [1e-3,1e-4,1e-5]
        
###########################
## Load train and test data
f = open('USC-HAD_processed.pckl','rb')
x_train,y_train,x_test,y_test = pickle.load(f)
f.close()

for seed_value in seed_value_list:

    tensorflow.keras.backend.clear_session()

    model_path = os.getcwd() + f'/Signal2PI_model/Seed-{seed_value}'
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    model_checkpoint_path = model_path + "/cp.ckpt.weights.h5"
    model_checkpoint_dir = os.path.dirname(model_checkpoint_path)

    model_cp_callback = tensorflow.keras.callbacks.ModelCheckpoint(
        model_checkpoint_path, verbose=0, save_weights_only=True)
        
    tensorflow.keras.utils.set_random_seed(seed_value)
    model = Signal_PINet()

    total_epochs = 0
    for training_segment in range(len(param.nb_epochs)):
        total_epochs+=param.nb_epochs[training_segment]

        opt1 = optimizers.Adam(learning_rate=param.learning_rate[training_segment], beta_1=0.99, epsilon=1e-1)

        model.compile(optimizer=opt1, loss='mean_squared_error')

        history = model.fit(x_train, y_train, batch_size=param.batch_size, epochs=param.nb_epochs[training_segment], verbose=1, validation_data=(x_test,y_test), callbacks = [model_cp_callback])

        model.save(model_path + f'/Model_{total_epochs}_epochs.h5')