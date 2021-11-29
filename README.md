# DQN-Algorithm Trader

An implementation of Deep Q-Network applied to foreign exchange trading. The model uses the difference of 5 and 10 days moving average of closing prices and normalised volumns to determine if the best action to take at a given time is to buy, sell or hold.

As a result, the model is not very good at making decisions over the relatively resting market, but produces excellent performance in turmoil market.

The model has been trained with GBP/USD data of 400 time steps and 300 episodes in January 2020,  and tested with the rest of year of 2020.
## data processing
please check 'preparing_data' file

## Running the Code
modify train.py line 15 to select data and change market.py line 9 to "self.df = df" for training. To avoid nan value, select starting row above 20. 

```
python train.py 
```

Then when training finishes you can evaluate with the test dataset :
Modify eveluate.py line 21 to select test dataset range, and change market line 9 accordingly.To avoid nan value and training set, enter starting row above 20 + training rows. model choose "model_ep29" as "model_ep39" was not saved properly. then run 
```
python evaluate.py 
```
