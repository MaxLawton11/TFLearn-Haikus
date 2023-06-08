from model_class import *
import os
import os.path
import random

# vars
ds_path = 'haiku_dataset.text'
model_path = 'model_instance.tflearn'
live_display = True
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
text = m.generate(seed, n_chars, temp, live_display)
del m

# format and print text
print(f'Seed: "{seed}"')
print(f'n_chars: {n_chars}')
print(f'Temp: {temp}')

print('\nRaw Text: ')
print(text)

print('\nProcessed Text: ')

# this makes sure that we end and an '$'.
# if there is none, then all is passed
if '$' in text :
    text = text.split('$')
    text.pop()
    text = ''.join(text)

text = ' '.join(text.split()) # turn everythin to single space
text = text.replace('\n','/').replace('$','/') # make new lines from / and $
text = text.split('/') # slit at /

print(text)

for line in text :
    # idk for some reson this wants to be here
    if len(line) < 1 :
        break
       
    # remove space if there is one at the start
    if line[0] == ' ' :
        print(line[1:])
    else :
        print(line)

print('--------- Success ---------') # we done here
