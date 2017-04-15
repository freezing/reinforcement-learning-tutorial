# -*- coding: utf-8 -*-

from freezing.gridworld.state import State

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import RMSprop

import numpy as np


def epsilon_greedy_action(state, qvals, epsilon):
    # TODO: Refactor this later to make it more generic. At the moment state is not used and it should return available actions.
    """Return random action with probability of epsilon, or the best action with probability of (1 - epsilon).
    """
    if np.random.random() < epsilon:
        return np.random.randint(0, 4)
    else:
        return np.argmax(qvals)


def run_epoch(model, alpha, gamma, epsilon):
    state = State()

    while not state.is_terminal():
        q_vals = model.predict(state.as_vector, batch_size=1)
        action = epsilon_greedy_action(state, q_vals, epsilon)

        new_state = state.run_action(action)
        reward = new_state.reward()

        new_q = model.predict(new_state.as_vector, batch_size=1)
        max_q = np.max(new_q)

        y = np.copy(q_vals)

        # TODO: Play with formula, use alpha (learning rate and expected value instead of max?)
        update = y[0][action];
        if not new_state.is_terminal():
            update = update + alpha * (reward + gamma * max_q - update)
        else:
            update = reward

        y[0][action] = update

        model.fit(state.as_vector, y, batch_size=1, nb_epoch=1, verbose=0)
        state = new_state


model = Sequential()

model.add(Dense(264, kernel_initializer='lecun_uniform', input_shape=(64,)))
model.add(Activation('relu'))

model.add(Dense(250, kernel_initializer='lecun_uniform'))
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

for epochId in range(0, epochs):
    print("Running epoch: " + str(epochId))
    run_epoch(model, alpha, gamma, epsilon)

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