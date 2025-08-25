# Tic Tac Toe Game with Reinforcement Learning


![Alt Text](https://i.makeagif.com/media/4-24-2016/N2q-9R.gif)

This repository contains a Tic Tac Toe game that uses reinforcement learning techniques to train the game player. The ML approach uses Q-values, epsilon greedy selections, and multi-threading to learn and improve its gameplay.

## Reinforcement Learning

The code learns by playing the game multiple times and updating the Q-values based on the outcomes of the games. An epsilon greedy selection strategy is used to balance exploration and exploitation during the learning process. The learning process is multi-threaded to speed up the training.

## Configuration - 'Its always DNS'

### tic_tac_learn_0.1.1 1 Billion game training session.
The aim is to create a training session that trains a 1 billion  game model  using the git 0.1.1 version of tic tac learn. 

The session will include a warmup session configuration and  a 1 billion parameter game with expected good training results. 

#### Warmup session 

``` Python
conf = Config_2_MC()
    conf.total_games = int(1e6)
    level = "WARMUP"
    conf.experiment_name= "Tic Tac Learn 0.1.1"
    conf.steps = 4
    
    conf.cores= 8
    conf.learning_rate_start= 0.8
    conf.learning_rate_min = 0.001
    conf.learning_rate_scaling = 1
    conf.test_games_per_step = 30000
    conf.learning_rate_flat_games = conf.total_games* 0.2


    conf.run_name = f"One Billion Games 4 Steps - {level} - {str(conf.total_games)}"
    conf.custom_model_name = f"{conf.run_name}_2mc"
```

```bash
docker build -t tic_tac_learn_0.1.1:warmupsession .
```

```bash
docker tag tic_tac_learn_0.1.1:warmupsession  homelab.docker.general/tic_tac_learn_0.1.1:warmupsession

docker push homelab.docker.general:5000/tic_tac_learn_0.1.1:warmupsession

docker run homelab.docker.general:5000/tic_tac_learn_0.1.1:warmupsession
```


###  Debug Session
```Python
conf = Config_2_MC()
    conf.total_games = int(1e4)
    level = "DEBUG"
    conf.experiment_name= "Tic Tac Learn 0.1.1"
    conf.steps = 4
    
    conf.cores= 8
    conf.learning_rate_start= 0.8
    conf.learning_rate_min = 0.001
    conf.learning_rate_scaling = 1
    conf.test_games_per_step = 30000
    conf.learning_rate_flat_games = conf.total_games* 0.2


    conf.run_name = f"One Billion Games 4 Steps - {level} - {str(conf.total_games)}"
    conf.custom_model_name = f"{conf.run_name}_2mc
```


```
docker build -t homelab.docker.general:5000/tic_tac_learn_0.1.1:debug .
```

## Current Status

Please note that this code is a work in progress.
