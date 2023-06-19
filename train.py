from model_class import *
import os
import os.path

ds_path = 'haiku_dataset.text'
model_path = 'model_instance.tflearn' #Note: this file don't exist. It is in both model_instance.tflearn.index and model_instance.tflearn.meta
n_epoch = 1

print('--------- Training ---------')
print(f'# Running {n_epoch} epoch(s)')

# create model
m = Model(ds_path)

# load model if there is one
if os.path.isfile(f'{model_path}.index') and os.path.isfile(f'{model_path}.meta') :
    m.model.load(model_path)
    print(f'# Loaded pre-existing model ({model_path})')
else :
    print('# No model found - Creating new model')
   
# train and save
m.train(n_epoch)
m.save(model_path)
del m

print('--- Model Traind & Saved ---')
