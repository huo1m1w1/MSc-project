import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd
from agent import Agent
from market_env import Market
import numpy as np
import matplotlib
from datetime import datetime
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
import sys
# import time
import time, datetime


def main():
    with open('output3.txt', 'w') as f: # to save the result in a file.
        sys.stdout = f
        df = pd.read_csv('prepared_data.csv')
        test = df[1000:]
        test = test.reset_index()
        model_name = "model_ep29"
        steps = 4122*3
        model = load_model("models/" + model_name)
        action_space = [0,1,-np.inf]
        market = Market(test)
        state, price_data, index= market.reset()
        agent = Agent(action_space, state, True, model_name)
        

        state, price_data, index = market.reset() # Start from an initial state

        for t in range(market.last_data_index):

            # Check the action to get reward and observe next state
            action, enter_rate = agent.act(state, test, index, False)
            
            # get next state
            next_state, next_price_data, reward, index, done = market.get_next_state_reward(action, df, enter_rate)
            state = next_state
            price_data = next_price_data

            if done:
                print("--------------------------------")
                print("{0} Total Profit: ".format(agent.get_total_profit()))
                print("--------------------------------")

        plot_action_profit(test, agent.action_history, agent.get_total_profit(), steps)

def plot_action_profit(data, action_data, profit, steps):
    #data['date'] = pd.to_datetime(data['date'], format="%Y/%m/%d %H:%M:%S")

    fig = pd.DataFrame({'date': np.array([datetime.datetime.strptime(i, '%Y-%m-%d %H:%M:%S') for i in data.iloc[:steps]['date']]),
                  'rate': data.iloc[:steps]['Close']})
    plt.plot(range(steps), data.Close[:steps])
    # plt.plot(fig.date, fig.rate, linewidth=1)
    plt.xlabel("date")
    plt.ylabel("rate")
    buy, sel = False, False
    for d in range(steps-1):
        if action_data[d] == 1:  # buy
            # buy, = plt.plot(fig['date'][d], data.Close[d], 'g^')
            buy, = plt.plot(d, data.Close[d], 'g^')
        elif action_data[d] == 2:  # sell
            # sel, = plt.plot(fig['date'][d], data.Close[d], 'rv')
            sel, = plt.plot(d, data.Close[d], 'rv')
    if buy and sel:
        plt.legend([buy, sel], ["Buy", "Sell"])

    pos = [int(steps/5)*i for i in range(6)]
    ticklabels = [fig['date'][int(steps/5)*i].strftime("%Y-%m-%d") for i in range(6)]
    # locs, labels = plt.xticks()
    plt.xticks(pos, ticklabels) 
    # fig.set_xticks(fig['date'].tolist())

    # plt.xticks(xticks, xticklabels)
    # plt.autoscale(True, axis='x', tight=True)
        

    plt.title("Total Profit: {0}".format(profit))
    plt.savefig("buy_sell.png")
    plt.show()

if __name__=="__main__":
    main()