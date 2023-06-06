from model_class import *
import os
import os.path
import random

# vars
ds_path = 'haiku_dataset.text'
model_path = 'model_instance.tflearn'
temp = .01
n_chars = 300

# enter any seed under 25 characters long, any more will be cut off!
# if empty, will randomly pull from dataset
seed = ''

#test for empty string
if seed == '' :
    # get random line from file and pick one
    file_lines = open(ds_path).read().splitlines()
    seed = random.choice(file_lines)
    del file_lines

seed = seed[0:25] # cut off extra seed

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
del m

# format and print text
print(f'Seed: "{seed}"')
print(f'n_chars: {n_chars}')
print('Text: ')

text = text.replace('$','/').split('/')
print(text)

print('--------- Success ---------')
