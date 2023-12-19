# Tic Tac Learn

## Overview 
This repo is a developmental space where different models that learn how to play tic tac toe can be created. 

## Methods
### Reinforcement Learning
This is currently implemented through montecarlo based methods. These methods generate q values fir each game state. These q values are then used to pick th emost optimal move to win the game. 
The code base is iterating quickly at this early stage and so the mc_[num] can be used to track the latest methods being trialed. At writing this is mc_simplified_multiprocess.py  and mc_6_refactor.py

In addition to the python there is a developmental - non working verison in Julia to compare speed at a later date