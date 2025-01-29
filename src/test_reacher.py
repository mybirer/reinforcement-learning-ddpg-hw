from unityagents import UnityEnvironment
import numpy as np
import torch
from ddpg_agent import Agent
import time

env = None
try:
    print("Starting Unity Environment for testing...")
    env = UnityEnvironment(
        file_name='./env/Reacher_Windows_x86_64/Reacher.exe',
        no_graphics=False,  # Görsel modda çalıştıralım
        worker_id=0
    )
    
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    
    # create DDPG agent
    agent = Agent(state_size=33, action_size=4, random_seed=0)
    
    # load the trained weights
    agent.actor_local.load_state_dict(torch.load('./models/actor_solved.pth'))
    agent.critic_local.load_state_dict(torch.load('./models/critic_solved.pth'))
    
    print("Starting test episodes...")
    
    num_episodes = 5  # Test için episode sayısı
    for i_episode in range(num_episodes):
        env_info = env.reset(train_mode=False)[brain_name]  # train_mode=False
        states = env_info.vector_observations
        scores = np.zeros(len(env_info.agents))
        
        while True:
            actions = agent.act(states, add_noise=False)  # Test için noise kapalı
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            scores += rewards
            states = next_states
            
            if np.any(dones):
                break
        
        print(f'Episode {i_episode+1} Score: {np.mean(scores):.2f}')
        time.sleep(0.5)  # Her episode arasında biraz bekleyelim
        
    env.close()
except Exception as e:
    print(f"\nError occurred:")
    print(f"Type: {type(e).__name__}")
    print(f"Message: {str(e)}")
    raise
    
finally:
    print("\nTest completed.") 