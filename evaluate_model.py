import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras as k
from utils import *

physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass
   
   
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='hcd_testingset.mgf')
parser.add_argument('--model', type=str, default='pretrained_model')

args = parser.parse_args()

model_name = args.model
print('Model used: ', model_name)
pm = k.models.load_model(model_name,custom_objects={'masked_spectral_distance': masked_spectral_distance})
pm.compile(optimizer=k.optimizers.Adam(lr=0.0003), loss=masked_spectral_distance,  metrics=[tf.keras.metrics.CosineSimilarity(axis=1), masked_spectral_distance])


def matrix_cosine(x, y):
  val =  np.einsum('ij,ij->i', x, y) / (
              np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1)
    )
  return val

test_data = readmgf(args.data)

test_spectras = asnp32([spectrum2vector(i['mz'], i['it'], BIN_SIZE, i['Charge']) for i in test_data])
test_labels = asnp32([embed_maxquant(seq, augment=False) for seq in test_data])


test_pred = pm.predict(test_labels)
test_sim = matrix_cosine(test_pred,test_spectras)
print('Self calculated mean similarity: ', np.mean(test_sim))
print("Keras Evaluate: ")
print(pm.evaluate(x=test_labels, y=test_spectras, verbose=2))

