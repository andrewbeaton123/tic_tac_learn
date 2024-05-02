# Tic Tac Toe Game with Reinforcement Learning

This repository contains a Tic Tac Toe game that uses reinforcement learning techniques to train the game player. The ML approach uses Q-values, epsilon greedy selections, and multi-threading to learn and improve its gameplay.

## Reinforcement Learning

The code learns by playing the game multiple times and updating the Q-values based on the outcomes of the games. An epsilon greedy selection strategy is used to balance exploration and exploitation during the learning process. The learning process is multi-threaded to speed up the training.

## Configuration

You can run the learning process by changing the `ConfigClass` in the main file. Here is an example of how to do this:

In this example, the learning process will be run on 4 cores, with 1000 steps per run, for a total of 10000 runs. The resulting model will be tested with 9508 games. The learning rate is set to 0.95.

```
    config = ConfigClass(4,# cores
        1000,#steps(games) per run
        10000, # total games to create a model from
        9508,#How many games to test with
        [0.95],# learning rate starting values
        "Run Name Here"
        )
```
## Current Status

Please note that this code is a work in progress and is currently not in active development.