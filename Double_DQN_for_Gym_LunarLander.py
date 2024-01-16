
env = gym.make('LunarLander-v2')

agent=Agent(stateShape=env.observation_space.shape[0],actionShape=env.action_space.n\
            ,exploreRate=1.0,exploreRateDecay=0.9995,minimumExploreRate=0.01,gamma=0.99,copyNetsCycle=100)



# train loop for 200 diffrent episodeds

# to load pretrained model to continue training uncomment the following line (the pretrained wheights are included in the github directory "pretrained_model")
# agent.loadModel("DoubleDQN_LunarLanderV2.h")
averageRewards=[]
totalRewards=[]
for i in range(1,200):
  done = False
  truncuated = False
  state,_=env.reset()
  rewards=0
  while (not done) and (not truncuated):
    action=agent.getAction(state)
    nextState,reward,done,truncuated,_=env.step(action)
    agent.memory.save(state,action,reward,nextState,int(done))
    rewards+=reward
    state=nextState
    agent.learn(batchSize=64)
  totalRewards.append(rewards)
  averageRewards.append(sum(totalRewards)/len(totalRewards))
  print(f"episode: {i+1}   reward: {rewards}  avg so far:{averageRewards[-1]} exploreRate:{agent.exploreRate}")

plt.title(f'Total Rewards')
# plt.yscale('symlog')
plt.plot(totalRewards)
plt.savefig("Total Rewards",dpi=200)


plt.clf()
plt.title(f'Average Rewards')
# plt.yscale('symlog')
plt.plot(averageRewards)
plt.savefig("Average Rewards",dpi=200)




# test loop for 10 diffrent episodeds

# to load pretrained model for evaluation uncomment the following line (the pretrained wheights are included in the github directory "pretrained_model")
# agent.loadModel("DoubleDQN_LunarLanderV2.h")
sum_evaluation_rewards=0
for i in range(0,20):
  done = False
  truncuated = False
  state,_=env.reset()
  rewards=0
  while (not done) and (not truncuated):
    action=agent.getAction(state,True)
    nextState,reward,done,truncuated,_=env.step(action)
    rewards+=reward
    state=nextState
    # no learning we want to test the model
  print(f" Test episode: {i+1}   gained reward: {rewards}")
  sum_evaluation_rewards+=rewards
print(f"average of evaluation rewards: {sum_evaluation_rewards/20}")




