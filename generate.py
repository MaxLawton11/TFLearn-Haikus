from model_class import *
import os
import os.path

path = 'haiku_dataset.text'
seed = "the world is finite"
temp = 0.5
n_chars = 100

print('--------- Generating ---------')
# define the model
m = Model(path)

# test for vaild model
if os.path.isfile('model_instance.tflearn.index') and os.path.isfile('model_instance.tflearn.meta') :
    m.model.load('model_instance.tflearn')
    print('# Loaded  model')
else :
    print('# No model found. Please train before generating.')

# load model
m.model.load('model_instance.tflearn')

# make text from seed
text = m.generate(seed, n_chars, temp)
print(f'Seed: "{seed}"')
print(f'n_chars: "{n_chars}"')
print('Text: ', text)
del m

print('--------- Success ---------')
