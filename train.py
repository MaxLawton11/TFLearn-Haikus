from Model import *
import os
import os.path
import sys

path = 'haiku_dataset.text'
n_epoch = 10

print('--------- Training ---------')
print(f'# Running {n_epoch} epoch(s)')

# create model
m = Model(f'{path}')

# load model if there is one
if os.path.isfile('model_instance.tflearn.index') and os.path.isfile('model_instance.tflearn.meta') :
    m.model.load('model_instance.tflearn')
    print('# Loaded pre-existing model')
else :
    print('# No model found - Creating new model')
   
# train and save
m.train(n_epoch)
m.save('model_instance.tflearn')
del m

print('--- Model Traind & Saved ---')
