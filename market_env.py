import numpy as np

class Market():
    def __init__(self, df):
        # self.data = df
        self.states = df[['feature1', 'feature2', 'RSI', 'SAR']].reset_index().iloc[:, 1: ]
        self.enter_rate = None    
        self.entry_money = 10000
        self.data = df[:4122*3][['Close']]
        self.data = self.data.reset_index()
        self.index = -1
        self.last_data_index = len(self.data) - 1




    def reset(self):
        self.index = -1      
        return self.states.iloc[0], self.data.Close[0], 0

    def get_next_state_reward(self, action, df, enter_rate):
        self.index += 1
        if self.index > self.last_data_index:
            self.index = 0
        next_state = self.states.iloc[self.index + 1]
        next_price_data = df.Close[self.index + 1]

        price_data = df.Close[self.index]
        reward = 0
        if action==2 and self.enter_rate is not None:
            current_money = self.entry_money * self.enter_rate/df.bidclose[self.index]
            reward = current_money - self.entry_money -1 # here minors £1 as comission fee
            # self.entry_money = current_money
            # self.enter_rate = None
        
        
        done = True if self.index == self.last_data_index - 1 else False

        return next_state, next_price_data, reward, self.index, done
