# -*- coding: utf-8 -*-

from freezing.gridworld.state import State

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import RMSprop

import numpy as np
import random

def epsilon_greedy_action(state, qvals, epsilon):
    # TODO: Refactor this later to make it more generic. At the moment state is not used and it should return available actions.
    """Return random action with probability of epsilon, or the best action with probability of (1 - epsilon).
    """
    if np.random.random() < epsilon:
        return np.random.randint(0, 4)
    else:
        return np.argmax(qvals)

def mini_batch_update(model, batch, alpha, gamma):
    X_train = []
    Y_train = []

    for memory in batch:
        old_state, action, reward, new_state = memory
        old_qvals = model.predict(old_state.as_vector, batch_size=1)

        new_q = model.predict(new_state.as_vector, batch_size=1)
        max_q = np.max(new_q)

        y = np.copy(old_qvals)

        update = y[0][action]
        if not new_state.is_terminal():
            update = update + alpha * (reward + gamma * max_q - update)
        else:
            update = reward

        y[0][action] = update

        X_train.append(old_state.as_vector.reshape(64,))
        Y_train.append(y.reshape(4,))

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)

    model.fit(X_train, Y_train, batch_size=batchSize, nb_epoch=1, verbose=0)


def run_epoch(model, alpha, gamma, epsilon, buffer_size, batch_size):
    state = State()

    replay = []
    replay_idx = 0
    while not state.is_terminal():
        q_vals = model.predict(state.as_vector, batch_size=1)
        action = epsilon_greedy_action(state, q_vals, epsilon)

        new_state = state.run_action(action)
        reward = new_state.reward()

        if len(replay) < buffer_size:
            replay.append((state, action, reward, new_state))
        else:
            if replay_idx < buffer_size - 1:
                replay_idx += 1
            else:
                replay_idx = 0

            replay[replay_idx] = (state, action, reward, new_state)

            mini_batch = random.sample(replay, batch_size)
            mini_batch_update(model, mini_batch, alpha, gamma)

        state = new_state


model = Sequential()

model.add(Dense(164, kernel_initializer='lecun_uniform', input_shape=(64,)))
model.add(Activation('relu'))

model.add(Dense(150, kernel_initializer='lecun_uniform'))
model.add(Activation('relu'))

model.add(Dense(50, kernel_initializer='lecun_uniform'))
model.add(Activation('relu'))

model.add(Dense(4, kernel_initializer='lecun_uniform'))
model.add(Activation('linear'))

rms = RMSprop()
model.compile(loss='mse', optimizer=rms)

# state = State()
# prediction = model.predict(state.as_vector, batch_size = 1)

tests = 15
epochs = 3000
epsilon = 1.0
alpha = 0.7
gamma = 0.9
bufferSize = 80
batchSize = 40

for epochId in range(0, epochs):
    print("Running epoch: " + str(epochId))
    run_epoch(model, alpha, gamma, epsilon, bufferSize, batchSize)

    if epsilon > 0.1:
        epsilon -= (1 / epochs)


for i in range(0, tests):
    print("NEW TRY: " + str(i))
    state = State()

    while True:
        print(state.display_grid())
        print()

        if state.is_terminal():
            break

        q_vals = model.predict(state.as_vector, batch_size=1)
        action = np.argmax(q_vals)
        state = state.run_action(action)
        print("Action: " + str(action))

    print()
    print()
    print()


# for _ in range(0, 2):
#     s = State()
#     print(s.display_grid())
#     print()
#
#     s1 = s.run_action(0)
#     print(s1.display_grid())
#     print()
#
#     s2 = s1.run_action(2)
#     print(s2.display_grid())
#     print()
#
#     s3 = s2.run_action(0)
#     print(s3.display_grid())
#     print()
#
#
#     print()
#     print()