import math
from pyteomics import mgf, mass
import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras as K
from tensorflow.keras import layers
from utils import *
import matplotlib.pyplot as plt
import random
import sys
import pickle
import tensorflow.keras as k
from tensorflow.keras import backend as K

physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass
  
parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='example.tsv')
parser.add_argument('--model', type=str, default='pretrained_model')
parser.add_argument('--output', type=str, default='seq2ms_prediction.msp')

args = parser.parse_args()

data = pd.read_csv(args.input, sep='\t')
data = data[data['Sequence'].str.contains('X') == False]
data = data[data['Sequence'].str.contains('U') == False]
data = data[data['Sequence'].str.contains('O') == False]
print(data.head())
print('Length of input:',len(data))
    
model_name = args.model
pm = k.models.load_model(model_name,custom_objects={'masked_spectral_distance': masked_spectral_distance})
pm.compile(optimizer=k.optimizers.Adam(lr=0.0003), loss=masked_spectral_distance, metrics=[tf.keras.metrics.CosineSimilarity(axis=1), masked_spectral_distance])

def matrix_cosine(x, y):
  val =  np.einsum('ij,ij->i', x, y) / (
              np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1)
    )
  return val
  
if 'peptide' in data:
  embedded_data = asnp32([embed_pdeep(data.iloc[i]) for i in range(len(data))])
else:
  embedded_data = asnp32([embed_maxquant(data.iloc[i]) for i in range(len(data))])

# generator for inputs
class input_generator(k.utils.Sequence):
    def __init__(self, spectra, data, batch_size):
        self.spectra = spectra
        self.batch_size = batch_size
        self.data = data

    def __len__(self):
        return math.ceil(len(self.spectra) / self.batch_size)

    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.spectra))

        return (self.spectra[start_idx: end_idx], self.data[start_idx: end_idx])

types = (tf.float16)
shapes = ((MAX_PEPTIDE_LENGTH+2, encoding_dimension))
#test_dataset = tf.data.Dataset.from_generator(testgen, output_types=types, output_shapes=shapes)
#test_dataset = tf.data.Dataset.from_generator(input_generator(embedded_data,batch_size), output_types=types, output_shapes=shapes)
#test_dataset = test_dataset.batch(batch_size)
generator = input_generator(embedded_data,data,batch_size)

if args.output == 'seq2ms_prediction.msp':
	output_filename = args.input.split('.')[0] + '.msp'
else:
	output_filename = args.output
with open(output_filename, 'w') as f:
  for i in generator:
      label, data = i
      pred_spectra = pm.predict(label,verbose=2)
      write_msp(f, pred_spectra, data)

f.close()

