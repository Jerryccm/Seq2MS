import numpy as np
import tensorflow as tf
from tensorflow import keras as k
from tensorflow.keras import layers
from utils import *
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv1D, Dense, Add, Activation, BatchNormalization
from tensorflow.keras import Input
    
print('Loading data...')
train_data = readmgf("ProteomeTools.mgf")

print('Data set size:', len(train_data))

class DataGen():
    def __init__(self, data , batch_size=64, shuffle=True):
        self.data = np.array(data)
        self.indexs = np.arange(self.data.shape[0])
        self.batch_size = batch_size
    
    def __len__(self):
        return self.data.shape[0]//self.batch_size
    
    def get_item(self, i):
        embed = asnp32(embed_maxquant(self.data[i], fixedaugment=False, key='N', havelong=True))
        spectra = asnp32(spectrum2vector(self.data[i]['mz'], self.data[i]['it'], BIN_SIZE, self.data[i]['Charge']))
        return embed, spectra
    
    def __call__(self):
        for i in self.indexs:
            yield self.get_item(i)
            
batch_size = 64
train_gen = DataGen(train_data)
types = (tf.float32, tf.float16)
shapes = ((MAX_PEPTIDE_LENGTH+2, encoding_dimension), (20000))
dataset = tf.data.Dataset.from_generator(train_gen, output_types=types, output_shapes=shapes)
dataset = dataset.batch(batch_size)

print('Made dataset')

physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass  

# model component functions
def cb(x, channel, kernel, padding='same'):
    x = Conv1D(channel, kernel_size=kernel, padding=padding)(x)
    x = BatchNormalization(gamma_initializer='zeros')(x)
    return x


def res(x, layers, kernel=(3,), act='relu', se=0,cbam = 0,rf = 16, **kws):
    normalizer = BatchNormalization

    ConvLayer = k.layers.Conv1D
    MaxPoolingLayer = k.layers.MaxPooling1D
    AvePoolingLayer = k.layers.AveragePooling1D
    GlobalPoolingLayer = k.layers.GlobalAveragePooling1D
    GlobalMaxLayer = k.layers.GlobalMaxPooling1D
    assert K.ndim(x) == 3

    raw_x = x  # backup input

    x = ConvLayer(layers, kernel_size=kernel, padding='same', **kws)(x)
    x = normalizer(gamma_initializer='zeros')(x)
    
    #convolution attention block module
    if cbam == 1:
        #channel part
        glomax = GlobalMaxLayer()(x)
        gloavg = GlobalPoolingLayer()(x)
        shared1 = Dense(max(4, layers // rf), activation='relu')
        shared2 = Dense(layers)
        
        glomax = shared1(glomax)
        glomax = shared2(glomax)
        
        gloavg = shared1(gloavg)
        gloavg = shared2(gloavg)
        
        c_attention = Add()([glomax, gloavg])
        c_attention = Activation('sigmoid')(c_attention)
        c_attention = k.layers.Reshape((1, -1))(c_attention)
        x = k.layers.Multiply()([x, c_attention])
        
        #spatial part
        #k_size = 1
        #avg_pool = Lambda(lambda x: K.mean(x, axis=2, keepdims=True))(x)
        #max_pool = Lambda(lambda x: K.max(x, axis=2, keepdims=True))(x)
        #concat = k.layers.Concatenate(axis=2)([avg_pool, max_pool])
        #s_attention = ConvLayer(filters = 1,kernel_size=k_size,strides=1,padding='same',activation='sigmoid',kernel_initializer='he_normal',use_bias=False)(concat)
        #x = k.layers.Multiply()([x, s_attention])
	
    #squeeze and excitation block
    if se == 1:
        x2 = GlobalPoolingLayer()(x)
        x2 = Dense(max(4, layers // rf), activation='relu')(x2)
        x2 = Dense(layers, activation='sigmoid')(x2)
        x2 = k.layers.Reshape((1, -1))(x2)

        x = k.layers.Multiply()([x, x2])

    if K.int_shape(x)[-1] != layers:
        raw_x = ConvLayer(layers, kernel_size=1, padding='same')(raw_x)
        raw_x = normalizer()(raw_x)

    x = Add()([raw_x, x])

    return Activation(act)(x)  # final activation

def mlp(x, hidden_units, dropout_rate):
      # for units in hidden_units:
      for i in range(2):
          x = layers.Dense(hidden_units, activation=tf.nn.gelu)(x)
          x = layers.Dropout(dropout_rate)(x)
      return x

#transformer block
def encode(x, head=1):
      # Layer normalization 1.
      back_up = x
      x = layers.LayerNormalization(epsilon=1e-6)(x)
      # Create a multi-head attention layer.
      attention_output = layers.MultiHeadAttention(
          num_heads=head, key_dim=512, dropout=0.1
      )(x, x)
      # Skip connection 1.
      x2 = layers.Add()([attention_output, x])
      # Layer normalization 2.
      x2 = layers.LayerNormalization(epsilon=1e-6)(x2)
      # MLP.
      x3 = mlp(x2, hidden_units=512, dropout_rate=0.1)
      # Skip connection 2.
      x = layers.Add()([x3, x2])

      return x


#function to build model
def build(act='relu'):
    inp = Input(shape=(MAX_PEPTIDE_LENGTH + 2, encoding_dimension))
    x = inp

    features = k.layers.Concatenate(axis=-1)([cb(x, 64, i) for i in range(2, 18)])

    x = Conv1D(1024, kernel_size=1, padding='same')(x)
    x = BatchNormalization()(x)

    x = Add()([x, features])
    x = Activation(act)(x)

    for i in range(8):
        x = res(x, 1024, 3, act=act, se=0,cbam=1,rf=2)

    for i in range(3):
        x = res(x, 1024, 1, se=0, act=act)
        

    x = k.layers.Conv1D(SPECTRA_DIMENSION, kernel_size=1, padding='valid')(x)
    x = Activation('sigmoid')(x)
    x = k.layers.GlobalAveragePooling1D(name='spectrum')(x)
    
    return k.models.Model(inputs=inp, outputs=x)


#instantiate model
pm = build()

    
pm.compile(optimizer=k.optimizers.Adam(lr=0.0003), loss=masked_spectral_distance,  metrics=[tf.keras.metrics.CosineSimilarity(axis=1), masked_spectral_distance])
print(pm.summary())


checkpoint_filepath = './checkpoint'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='loss',
    mode='min',
    save_best_only=True)
    
def scheduler(epoch, lr):
    if epoch < 15:
      return lr
    elif epoch > 25:
      return lr
    else:
      return lr * tf.math.exp(-0.05)

callback = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=2)

test_data = readmgf('hcd_testingset.mgf')

test_spectras = asnp32([spectrum2vector(i['mz'], i['it'], BIN_SIZE, i['Charge']) for i in test_data])
test_labels = asnp32([embed_maxquant(seq, augment=False) for seq in test_data])
 
class TestCallback(tf.keras.callbacks.Callback):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def on_epoch_end(self, epoch, logs):
        print("hcd testingset evaluation:")
        print(self.model.evaluate(x=self.x, y=self.y, verbose=2))

print('Start model training')
pm.fit(x=dataset, epochs=30, batch_size=32, callbacks=[callback, model_checkpoint_callback, TestCallback(test_labels,test_spectras)], verbose = 2)

pm.save('trained_model')  

