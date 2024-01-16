### DDQN_Deep-Double_Q-Learning-for-solving-OpenAi-Gym-LunarLander-v2-in-python
***reinforcement learning Double Deep Q Learning (DDQN) method to solve OpenAi Gym "LunarLander-v2" by usnig Double Deep NeuralNetworks
Solving Openai gym LunarLanderV2 by using Double DQN***  
Naive TensorFlow (using Keras) implementation of paper:  Deep Reinforcement Learning with Double Q-learning
https://arxiv.org/abs/1509.06461   

[feel free to ask any question in Issues or just email me]  
Mohammad.E.Asadolahi@gmail.com

#### How to install requirements
The `requirements.txt` file should list all Python libraries that the project depend on, and they will be installed using:
```
pip install -r Requirements.txt
```
I keep updating the project to be compatible with new versions of libraries. If there was any problem with the diffrent versions of the required libraries let me know in the "Issues" section, so i can resolve them.  
  
  
**this code is implemented with TensorFlow using Keras! i will add Pytorch version soon!!**  
#### to do:  
* deploy the ReplayBuffer code   [***done***]
* add imports file   [***done***]
* deploy Q approximation neural network [***done***]
* deploy the Agnet in pytorch  [***done***]
* deploy the Environmetn class (using OpenAI Gym library to import an environment) [***done***]
* deploy the main learning loop [***done***]
* deploy the test loop [***done***]
* run a standard RL evaluation
* refactor the project in Pytorch


#### Sample Results: 
![Average Rewards](https://github.com/MohammadAsadolahi/DDQN_Deep-Double_Q-Learning-for-solving-OpenAi-Gym-LunarLander-v2-in-python/blob/main/Results/Average%20Rewards.png)
![Total Rewards](https://github.com/MohammadAsadolahi/DDQN_Deep-Double_Q-Learning-for-solving-OpenAi-Gym-LunarLander-v2-in-python/blob/main/Results/Total%20Rewards.png)
