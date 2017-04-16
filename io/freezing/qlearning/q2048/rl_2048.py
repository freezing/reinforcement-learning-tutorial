"""Reinforcement learning using Q-Learning algorithm for 2048 Game."""

from keras.models import Sequential, load_model
from keras.layers.core import Dense, Activation
from keras.optimizers import RMSprop

from freezing.g2048.game_2048 import Game2048, DEFAULT_BOARD_SIZE, NUMBER_OF_ACTIONS
from freezing.g2048.strategy_2048 import *
from freezing.g2048.player_2048 import *

from collections import deque
import random

INPUT_SIZE = DEFAULT_BOARD_SIZE ** 2
OUTPUT_SIZE = NUMBER_OF_ACTIONS

DEFAULT_LAYERS_SIZES = [256, 256]

ACTIVATION = Activation('relu')
OUTPUT_ACTIVATION = Activation('linear')


# Default RL parameters
DEFAULT_EPOCHS = 100

DEFAULT_LEARNING_RATE = 0.7
DEFAULT_DISCOUNT_FACTOR = 0.9

DEFAULT_START_EPSILON = 1.0
DEFAULT_EPSILON_DECREMENT = 1.0 / DEFAULT_EPOCHS
DEFAULT_MIN_EPSILON = 0.1

DEFAULT_BATCH_SIZE = 40


MEMORY_CAPACITY = 80


GAME_OVER_REWARD = -100


class ReinforcementLearning2048(object):
    """Implementation of the Neural Network Q-learning algorithm for 2048 Game."""

    @staticmethod
    def train_model(game_generator,
                    model=None,
                    learning_rate=DEFAULT_LEARNING_RATE,
                    discount_factor=DEFAULT_DISCOUNT_FACTOR,
                    start_epsilon=DEFAULT_START_EPSILON,
                    epsilon_decrement=DEFAULT_EPSILON_DECREMENT,
                    min_epsilon=DEFAULT_MIN_EPSILON,
                    batch_size=DEFAULT_BATCH_SIZE,
                    verbose=False):
        """Trains model Neural Network using Q-Learning algorithm by running 2048 Game simulations.

            :arg game_generator: Generator of the Game2048 instances to run each epoch against.
                This is instead of the number of epochs to run the training against.
            :arg model: Optional Neural Network to be trained. If None, creates default model.
            :arg learning_rate: Q-Learning algorithm learning rate.
            :arg discount_factor: Q-Learning algorithm discount factor.
            :arg start_epsilon: Start probability for the epsilon_greedy_strategy.
            :arg epsilon_decrement: Decrement probability after each epoch.
            :arg min_epsilon: Minimum probability for the epsilon_greedy_strategy.
            :arg batch_size: Size of the batch data set to use when fitting the model.
            :arg verbose: If True, prints detailed information about training.

            :return Trained Neural Network model using Q-Learning algorithm.
        """

        if model is None:
            model = ReinforcementLearning2048.create_sequential_nn(ACTIVATION, OUTPUT_ACTIVATION, INPUT_SIZE,
                                                                   DEFAULT_LAYERS_SIZES, OUTPUT_SIZE)

        # TODO: Create Memory class to keep these
        memory = []   # Queue of experiences
        memory_idx = 0

        epsilon = start_epsilon
        for game in game_generator:
            q_values_provider = ReinforcementLearning2048.make_q_values_provider(model)
            strategy = make_epsilon_greedy_strategy(q_values_provider, epsilon)
            _, experiences = play(strategy, game, verbose=verbose, allow_unavailable_actions=False)

            # Add all experiences in the memory
            for exp in experiences:
                if memory_idx < MEMORY_CAPACITY:
                    memory.append(exp)
                    memory_idx += 1
                else:
                    if memory_idx == MEMORY_CAPACITY:
                        memory_idx = 0
                    memory[memory_idx] = exp
                    memory_idx += 1

                # TODO: Should I do it here?
                # Pick random samples from memory
                samples = random.sample(memory, min(len(memory), batch_size))
                ReinforcementLearning2048.batch_train(model, samples, learning_rate, discount_factor)

            epsilon = max(min_epsilon, epsilon - epsilon_decrement)

        return model

    @staticmethod
    # TODO: This is where we can use batch prediction
    def batch_train(model, experiences_batch, learning_rate, discount_factor):
        # TODO: add allow_unavailable_actions
        """Fit the model with the given experiences_batch, skipping invalid move actions."""

        state_batch = np.array([exp.state.flatten() for exp in experiences_batch])
        all_q_vals = model.predict(state_batch, batch_size=len(state_batch))

        new_state_batch = np.array([exp.next_state.flatten() for exp in experiences_batch])
        new_all_q_vals = model.predict(new_state_batch, batch_size=len(new_state_batch))

        X_train = []
        Y_train = []
        for idx, experience in enumerate(experiences_batch):
            if experience.is_not_available:
                continue

            reward = experience.reward
            if experience.is_game_over:
                reward = GAME_OVER_REWARD

            # q_vals = model.predict(experience.state.reshape(1, INPUT_SIZE), batch_size=1)
            # new_q_vals = model.predict(experience.next_state.reshape(1, INPUT_SIZE), batch_size=1)

            q_vals = np.array([all_q_vals[idx]])
            new_q_vals = np.array([new_all_q_vals[idx]])

            y = q_vals.copy()
            update = y[0][experience.action]

            if not experience.is_game_over:
                sorted_actions = np.argsort(new_q_vals[0])
                next_action = [a for a in sorted_actions if a in experience.next_state_available_actions][-1]
                new_max_q = new_q_vals[0][next_action]

                update += learning_rate * (reward + discount_factor * new_max_q - update)
            else:
                update = reward

            y[0][experience.action] = update
            X_train.append(experience.state.flatten())
            Y_train.append(y.flatten())

        X_train = np.array(X_train)
        Y_train = np.array(Y_train)
        model.fit(X_train, Y_train, batch_size=len(Y_train), epochs=1, verbose=0)

    @staticmethod
    def make_q_values_provider(model):
        """Creates q_values_provider.

            :arg model: Neural Network model used as basis for q_values_provider.
        """

        # TODO: Implement through batch q values provider once it's done
        def q_values_provider(state):
            """Returns Q-values for the given state based on the model.

                :arg state: A state to calculate Q-values for.
            """

            q_values = model.predict(state.reshape(1, INPUT_SIZE), batch_size=1)
            return q_values[0]

        return q_values_provider

    @staticmethod
    def create_sequential_nn(activation, output_activation, input_size, layers_sizes, output_size):
        """Creates Neural Network with the given activation functions for all layers and output layer.

            :arg activation: Default activation function used between all layers, except the output layer.
            :arg output_activation: Activation function used for output layer.
            :arg input_size: Size of the input layer.
            :arg layers_sizes: List of sizes for fully-connected layers between the input and the output layers.
            :arg output_size: Size of the output layer.
        """

        model = Sequential()
        # TODO: How to add convolutional layer

        for idx, layer_size in enumerate(layers_sizes):
            name = "Dense_" + str(idx) + "_" + str(layer_size)
            model.add(Dense(layer_size, kernel_initializer='lecun_uniform', input_shape=(input_size,), name=name))
            activation = Activation(activation='relu', name='Activation_Relu_' + str(idx))
            model.add(activation)

        output_name = "Dense_output_" + str(output_size)
        model.add(Dense(output_size, kernel_initializer='lecun_uniform', name=output_name))
        output_activation = Activation(activation='linear', name='Activation_Output_Linear')
        model.add(output_activation)

        rms = RMSprop()
        model.compile(loss='mse', optimizer=rms)

        return model


def game_generator():
    for i in range(10000):
        print("Running simulation: {:d}".format(i))
        yield Game2048()

model = ReinforcementLearning2048.train_model(game_generator(), verbose=False)
model.save("/Users/freezing/Projects/reinforcement-learning/resources/model.h5")
# model = load_model("/Users/freezing/Projects/reinforcement-learning/resources/model.h5")


game = Game2048(seed=1)
q_values_provider = ReinforcementLearning2048.make_q_values_provider(model)
strategy = make_greedy_strategy(q_values_provider)
score, experiences = play(strategy, game, verbose=True, allow_unavailable_actions=False)