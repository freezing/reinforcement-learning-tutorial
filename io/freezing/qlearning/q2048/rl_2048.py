"""Reinforcement learning using Q-Learning algorithm for 2048 Game."""

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import RMSprop

from freezing.g2048.game_2048 import Game2048, DEFAULT_BOARD_SIZE, NUMBER_OF_ACTIONS

INPUT_SIZE = DEFAULT_BOARD_SIZE ** 2
OUTPUT_SIZE = NUMBER_OF_ACTIONS

ACTIVATION = Activation('relu')
OUTPUT_ACTIVATION = Activation('linear')


class ReinforcementLearning2048(object):
    """Implementation of the Neural Network Q-learning algorithm for 2048 Game."""

    @staticmethod
    def default_model(activation, output_activation, input_size, layers_sizes, output_size):
        """Creates Neural Network with the given activation functions for all layers and output layer.

            :arg activation: Default activation function used between all layers, except the output layer.
            :arg output_activation: Activation function used for output layer.
            :arg input_size: Size of the input layer.
            :arg layers_sizes: List of sizes for fully-connected layers between the input and the output layers.
            :arg output_size: Size of the output layer.
        """

        model = Sequential()
        # TODO: How to add convolutional layer

        for layer_size in layers_sizes:
            model.add(Dense(layer_size, kernel_initializer='lecun_uniform', input_shape=(input_size,)))
            model.add(activation)

        model.add(Dense(output_size, kernel_initializer='lecun_uniform'))
        model.add(output_activation)

        rms = RMSprop()
        model.compile(loss='mse', optimizer=rms)

        return model



