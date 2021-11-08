import tensorflow as tf
from tensorflow.keras.layers import Layer


class SpecAugment(Layer):
    """
    Implementation of a layer that contains the SpecAugment Transformation
    """

    def __init__(self,
                 freq_mask_param: int,
                 time_mask_param: int,
                 n_freq_mask: int = 1,
                 n_time_mask: int = 1,
                 mask_value: float = 0.
                 ):
        """
        :param freq_mask_param: Frequency Mask Parameter (F in the paper)
        :param time_mask_param: Time Mask Parameter (T in the paper)
        :param n_freq_mask: Number of frequency masks to apply (mF in the paper). By default is 1.
        :param n_time_mask: Number of time masks to apply (mT in the paper). By default is 1.
        :param mask_value: Imputation value. By default is zero.
        """
        super(SpecAugment, self).__init__(name="SpecAugment")
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.n_freq_mask = n_freq_mask
        self.n_time_mask = n_time_mask
        self.mask_value = tf.cast(mask_value, tf.float32)

    def _frequency_mask_single(self, input_mel_spectrogram: tf.Tensor) -> tf.Tensor:
        """
        :param input_mel_spectrogram:
        :return:
        """
        n_mels = tf.cast(tf.shape(input_mel_spectrogram)[1], tf.float32)
        freq_indices = tf.reshape(tf.cast(tf.range(n_mels), tf.int32), (1, -1, 1))

        # We use the paper's notation
        f = tf.cast(tf.random.uniform(shape=(), maxval=self.freq_mask_param), tf.int32)
        f0 = tf.cast(tf.random.uniform(shape=(), maxval=n_mels - tf.cast(f, tf.float32)), tf.int32)

        condition = tf.logical_and(freq_indices >= f0, freq_indices <= f0 + f)
        return tf.cast(condition, tf.float32)

    def _frequency_masks(self, input_mel_spectrogram: tf.Tensor) -> tf.Tensor:
        """
        :param input_mel_spectrogram:
        :return:
        """
        mel_repeated = tf.repeat(tf.expand_dims(input_mel_spectrogram, 0), self.n_freq_mask, axis=0)
        masks = tf.cast(tf.map_fn(elems=mel_repeated, fn=self._frequency_mask_single), tf.bool)
        mask = tf.math.reduce_any(masks, 0)
        return tf.where(mask, self.mask_value, input_mel_spectrogram)

    def _time_mask_single(self, input_mel_spectrogram: tf.Tensor) -> tf.Tensor:
        """
        :param input_mel_spectrogram:
        :return:
        """
        n_steps = tf.cast(tf.shape(input_mel_spectrogram)[0], tf.float32)
        time_indices = tf.reshape(tf.cast(tf.range(n_steps), tf.int32), (-1, 1, 1))

        # We use the paper's notation
        t = tf.cast(tf.random.uniform(shape=(), maxval=self.time_mask_param), tf.int32)
        t0 = tf.cast(tf.random.uniform(shape=(), maxval=n_steps - tf.cast(t, tf.float32)), tf.int32)

        condition = tf.logical_and(time_indices >= t0, time_indices <= t0 + t)
        return tf.cast(condition, tf.float32)

    def _time_masks(self, input_mel_spectrogram: tf.Tensor) -> tf.Tensor:
        """
        :param input_mel_spectrogram:
        :return:
        """
        mel_repeated = tf.repeat(tf.expand_dims(input_mel_spectrogram, 0), self.n_time_mask, axis=0)
        masks = tf.cast(tf.map_fn(elems=mel_repeated, fn=self._time_mask_single), tf.bool)
        mask = tf.math.reduce_any(masks, 0)
        return tf.where(mask, self.mask_value, input_mel_spectrogram)

    def _apply_spec_augment(self, input_mel_spectrogram: tf.Tensor) -> tf.Tensor:
        """
        :param input_mel_spectrogram:
        :return:
        """
        if self.n_freq_mask >= 1:
            input_mel_spectrogram = self._frequency_masks(input_mel_spectrogram)
        if self.n_time_mask >= 1:
            input_mel_spectrogram = self._time_masks(input_mel_spectrogram)
        return input_mel_spectrogram

    def call(self, inputs: tf.Tensor, training=None, **kwargs):
        """
        Applies the SpecAugment operation to the input Mel Spectrogram
        :param inputs: The input mel spectrogram
        :param training: If True then it will be applied
        :return: A mel spectrogram after the time and frequency are applied
        """
        if training:
            inputs_masked = tf.map_fn(elems=inputs, fn=self._apply_spec_augment)
            return inputs_masked
        return inputs

    def get_config(self):
        """
        Generates a description of the parameters selected. It uses the notation in the paper
        :return:
        """
        config = {
            "F": self.freq_mask_param,
            "T": self.time_mask_param,
            "mF": self.n_freq_mask,
            "mT": self.n_time_mask,
            "mask_value": self.input_mask_value
        }
        return config
