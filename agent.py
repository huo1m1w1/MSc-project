import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

import numpy as np
import random
from collections import deque

class Agent:
    def __init__(self, action_space, state, is_eval=False, model_name=""):
        self.__inventory = []
        self.__total_profit = 0
        self.action_history = []
        self.entry_money = 10000 # in GBP
        self.action_space = action_space
        self.action_size = 3  # hold, buy, sell
        self.memory = deque(maxlen=1000)
        self.model_name = model_name
        self.is_eval = is_eval
        self.action_space = action_space
        self.gamma = 0.98
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.state_size = len(state)
        self.model = load_model("models/" + model_name) if is_eval else self.create_model()

    # Create deep learning model structure
    def create_model(self):
        model = Sequential()
        model.add(Dense(units=32, input_dim=self.state_size, activation="relu"))
        model.add(Dense(units=8, activation="relu"))
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(learning_rate=0.01))
        
        return model

    def reset(self):
        self.__inventory = []
        self.__total_profit = 0
        self.action_history = []
        self.entry_money = 10000
        self.action_space=  [0,1,-np.inf]
    def act(self, state, df, index, done):
        self.df = df.reset_index()
        self.index = index
        self.state_size = len(state)
        if not self.is_eval and np.random.rand() <= self.epsilon:
            # select a random action in available action space

            action = random.choice(self.action_space)
            while action == -np.inf:
                action = random.choice(self.action_space)
        else:
            # Predict what would be the possible action for a given state
            options = self.model.predict(state.to_numpy().reshape(-1,4)) # predicted q-value of the current state

            # pick the action with highest probability within available action space
            if self.action_space[options[0].tolist().index(max(options[0].tolist()))] == -np.inf:
                action = self.action_space[options[0].tolist().index(sorted(options[0].tolist())[-2])]# choose second largest if the largest value not available
            else:
                action = np.argmax(options[0]) # otherwise select the highest value
        if done:
            if self.action_space == [0, -np.inf, 2] and enter_rate is not None:
                action = 2

        enter_rate = None
        if action == 0:  # Do nothing!
            print(".", end='', flush=True) # mark as '.' 
            self.action_history.append(action)
        elif action == 1:  # buy
            enter_rate = self.df.askclose[self.index]
            self.buy(enter_rate)
            self.action_history.append(action)
            self.action_space = [0, -np.inf, 2]
        elif action == 2 and self.has_inventory():  # sell
            sell_rate = self.df.bidclose[self.index]  # bid price is the price that trader is willing to buy, 
            self.sell(sell_rate)
            self.action_history.append(action)
            self.action_space = [0, 1, -np.inf]
            enter_rate = None

        else: 
            self.action_history.append(0)
            
        return action, enter_rate

    def batch_learning(self, batch_size):
        mini_batch = []
        l = len(self.memory)
        for i in range(l - batch_size + 1, l):
            mini_batch.append(self.memory[i])

        for state, action, reward, next_state, done in mini_batch:
            if done:
                target = reward
            else:
                # updated q_value = reward + gamma * [max_a' Q(s',a')]
                next_q_values = self.model.predict(next_state.to_numpy().reshape(-1, 4))[0] # this is Q(s', a') for all possible a'
                next_q_values[self.action_space.index(-np.inf)]=-np.inf # assign -inf value on action not available in action space
                target = reward + self.gamma * np.amax(next_q_values) # update target q_value using Bellman equation

            predicted_target = self.model.predict(state.to_numpy().reshape(-1, 4)) # predict q_value for current state
            # Update the action values
            predicted_target[0][action] = target #  Substitue target q_value to the predicted value
            # Train the model with updated action values
            self.model.fit(state.to_numpy().reshape(-1,4), predicted_target, epochs=1, verbose=0) # train the model with new q_value

        # Make epsilon smaller over time, so do more exploitation than exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def buy(self, enter_rate):
        self.__inventory.append(enter_rate)
        print("Buy: {0}".format(enter_rate))

    def sell(self, sell_rate):
        enter_rate = self.__inventory.pop(0)
        current_money = self.entry_money * enter_rate/self.df.bidclose[self.index]
        profit = current_money - self.entry_money -1 # here minors £1 as comission fee
        print("\n")
        print("Sell: {0} | Profit: {1} | Starting Balance: {2}".format(sell_rate, self.format_price(profit), self.format_price(self.entry_money)))
        
        self.entry_money = current_money-1
        return enter_rate

    def get_total_profit(self):
        self.__total_profit = self.entry_money - 10000
        return self.format_price(self.__total_profit)

    def has_inventory(self):
        return len(self.__inventory) > 0

    @staticmethod
    def format_price(n):
        return ("-£" if n < 0 else "£") + "{0:.4f}".format(abs(n))
