from agent import Agent
from market_env import Market
import pandas as pd
import numpy as np
import os
import sys
import time

def main():
    with open('output.txt', 'w') as f:
        sys.stdout = f

        episode_count = 200
        df = pd.read_csv('prepared_data.csv')
        train = df[20:420]
        
        action_space = [0,1,-np.inf] # initial position only for buy or hold, we cannot sell. 
        
        market = Market(train)
        state, price_data, index= market.reset()
        batch_size = 32
        agent = Agent(action_space, state)
        start_time = time.time()
        pnl = [] # profit and loss
        for e in range(episode_count):
            print("Episode " + str(e+1) + "/" + str(episode_count))

            agent.reset()
            state, price_data, index= market.reset() # get the initial state
            done = False
            for t in range(market.last_data_index):
                # get the action of the agent
                action, enter_rate = agent.act(state, train, index, done) # Call the act() method of the agent considering the current state

                # get the next state of the stock
                # Get the next available state from market data
                next_state, next_price_data, reward, index, done = market.get_next_state_reward(action, df, enter_rate)

                # add the transaction to the memory
                agent.memory.append((state, action, reward, next_state, done))
                # learn from the history
                if len(agent.memory) > batch_size:
                    agent.batch_learning(batch_size)

                state = next_state
                price_data = next_price_data

                if done:
                    if action_space == [0, -np.inf, 2] and enter_rate is not None:
                        _, _, reward, done =  market.get_next_state_reward(2, df, enter_rate)
                    pnl.append(agent.entry_money)
                    print("--------------------------------")
                    print("Total Profit: {0}".format(agent.get_total_profit()))
                    print("--------------------------------")

                if e % 10 == 9:
                    if not os.path.exists("models"):
                        os.mkdir("models")
                    agent.model.save("models/model_ep" + str(e))
                pnl.append( agent.get_total_profit())
        end_time = time.time()
        training_time = round(end_time - start_time)
        print("Training took {0} seconds.".format(training_time))
        A = []
        for i, x in enumerate(pnl):
            if i/2 == i//2:
                A.append(x)

        print(A)
        pd.DataFrame(A).plot()


if __name__ == "__main__":
    main()





