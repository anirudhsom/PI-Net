import scipy.io as sio
import matplotlib.pyplot as plt
import pickle, os

from pi_net import *

class DoDict(dict):
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self

###########################
## Load train and test data
f = open('USC-HAD_processed.pckl','rb')
x_train,y_train,x_test,y_test = pickle.load(f)
f.close()

###########################
## Evaluation
opt1 = optimizers.Adam(learning_rate=1e-3, beta_1=0.99, epsilon=1e-1)

model = Signal_PINet()
model.compile(optimizer=opt1, loss='mean_squared_error')
model.load_weights(f'Signal2PI_model/Seed-100/Model_1000_epochs.h5')

print(f"Train MSE: {model.evaluate(x_train,y_train)}")
print(f"Test MSE: {model.evaluate(x_test,y_test)}")