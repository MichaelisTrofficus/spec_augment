import tensorflow as tf
from tensorflow.keras.models import Sequential
import kapre
import librosa
import matplotlib.pyplot as plt
import librosa.display
from spec_augment.spec_augment import SpecAugment

filename = librosa.ex('trumpet')
y, sr = librosa.load(filename)

audio_tensor = tf.reshape(tf.cast(y, tf.float32), (1, -1, 1))
input_shape = y.reshape(-1, 1).shape

melgram = kapre.composed.get_melspectrogram_layer(input_shape=input_shape, n_fft=1024, return_decibel=True,
                                                  n_mels=256, input_data_format='channels_last',
                                                  output_data_format='channels_last')
spec_augment = SpecAugment(20, 20, 3, 2)

model = Sequential()
model.add(melgram)
model.add(spec_augment)
