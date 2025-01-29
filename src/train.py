from unityagents import UnityEnvironment
import numpy as np
from src.agent import Agent
from collections import deque
import torch
import torch.nn.functional as F
import torch.optim as optim
import time

#set the environment
env = UnityEnvironment(file_name="./env/Reacher_Windows_x86_64/Reacher.exe")

#set the brain name
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

#set the agent
agent = Agent(state_size=33, action_size=4, random_seed=2)

#set the number of agents
num_agents = len(env_info.agents)

#set the ddpg function
def ddpg(n_episodes=2000, max_t=1000):

    print("Enter ddpg...\n")
    scores_deque = deque(maxlen=100)
    scores = []
    best_score = 0
    best_average_score = 0
    for i_episode in range(1, n_episodes + 1):

        avg_score = 0
        # reset the environment
        env_info = env.reset(train_mode=True)[brain_name]
        # get the number of agents
        num_agents = len(env_info.agents)
        # get the states vector
        states = env_info.vector_observations
        # init score agents
        scores_agents = np.zeros(num_agents)
        score = 0
        agent.reset()
        for t in range(max_t):
            # choose actions
            actions = agent.act(states)
            # send the actions to the environment
            env_info = env.step(actions)[brain_name]
            # get the next states
            next_states = env_info.vector_observations
            # get the rewards
            rewards = env_info.rewards
            # see if episode has finished
            dones = env_info.local_done
            agent.step(states, actions, rewards, next_states, dones)
            states = next_states
            scores_agents += rewards
            if np.any(dones):
                break
        # mean score of 20 agents in this episode
        score = np.mean(scores_agents)
        scores_deque.append(score)
        #
        avg_score = np.mean(scores_deque)
        scores.append(score)
        # refresh the best agent score
        if score > best_score:
            best_score = score
        # refresh the best average score
        if avg_score > best_average_score:
            best_average_score = avg_score

        # print current episode
        print(
            "Episode:{}, Score:{:.2f}, Best Score:{:.2f}, Average Score:{:.2f}, Best Avg Score:{:.2f}".format(
                i_episode, score, best_score, avg_score, best_average_score
            )
        )
        if avg_score >= 32:
            torch.save(agent.actor_local.state_dict(), "train_data/actor_solved.pth")
            torch.save(agent.critic_local.state_dict(), "train_data/critic_solved.pth")
            break
    return scores


start = time.time()
scores = ddpg()
end = time.time()
print("\nTotal training time = {:.1f} min".format((end - start) / 60))
