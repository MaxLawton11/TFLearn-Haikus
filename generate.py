from model_class import *
import os
import os.path

ds_path = 'haiku_dataset.text'
model_path = 'model_instance.tflearn'
seed = "test of faith / today i"
temp = 0.01
n_chars = 100

print('--------- Generating ---------')
# define the model
m = Model(ds_path)

# test for vaild model
if os.path.isfile(f'{model_path}.index') and os.path.isfile(f'{model_path}.meta') :
    m.model.load('model_instance.tflearn')
    print(f'# Loaded  model ({model_path})')
else :
    print('# No model found. Please train before generating.')

# load model
m.model.load(model_path)

# make text from seed
text = m.generate(seed, n_chars, temp)
print(f'Seed: "{seed}"')
print(f'n_chars: {n_chars}')
print('Text: \n', text)
del m

print('--------- Success ---------')
