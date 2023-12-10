# begin Python q-learning tutorial: 
# the goal is to compose a Q-Table, a memoization table
import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("MountainCar-v0")  # q-learners should perform in other environments also
env.reset()
# print(env.observation_space.high)
# print(env.observation_space.low)
# print(env.action_space.n)
num_wins = 0  # win counter

# define the q-learning parameters (ADJUST THESE)
LEARNING_RATE = 0.15  # 0.1
DISCOUNT = 0.95  # reducing this value increases sensitivity to distant rewards
EPISODES = 10000  # 
SHOW_EVERY = 1000  # 
NUM_OPTIONS = 8  # number of adjacent move options presented by all adjacent pixels, drawn out as a square around the central point (state).

DISCRETE_OBS_SIZE = [NUM_OPTIONS] * len(env.observation_space.high)  # number of choices * extent of the observation_space
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OBS_SIZE  # descretizes the state space to manifest the entire cube
# print(discrete_os_win_size)

# define the epsilon control parameter
epsilon = 0.5  # controls agent stochasticity, decreases by 'epsilon_decay_value' for the first 'END_EPSILON_DECAYING' episodes
START_EPSILON_DECAYING = 1  # offset by one to avoid division by zero
DIV_FACTOR = 30  # try 2,3,4,...  larger value == larger delta_epsilon "chunks"
END_EPSILON_DECAYING = EPISODES // DIV_FACTOR  # determines size of the denominator in 'epsilon_decay_value'
# the next line forms the delta_epsilon term
epsilon_decay_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)  # normalized epsilon decay value

q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OBS_SIZE + [env.action_space.n]))  # Q-table with states in columns and (state,action) cominations in rows
# print(q_table.shape)
print(q_table)  # q-table: rows represent potential states the agent can go to, columns represent the actions leading the agent to that state.

ep_rewards = []  # episode rewards
aggr_ep_rewards = {'ep': [], 'avg': [], 'min': [], 'max': []}  # 'avg' should contain a pooled average

# get_discrete_state() takes an agents current state and maps it onto the discretized game environment.
def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size  # normalized relative state value
    return tuple(discrete_state.astype(np.int64))  # returns as a tuple that will eventually comprise the (state,action) pair


for episode in range(EPISODES):
    
    episode_reward = 0  # keep track of sequences of events that scored higher
    if episode % SHOW_EVERY == 0:  # give update every SHOW_EVERY episodes
        print(episode)
        render = True
    
    else: 
        render = False
        
    discrete_state = get_discrete_state(env.reset())  # collect the 1st discrete state for the new board
    done = False
    while not done:
        # eploration/exploitation trade-off occurs here
        
        if np.random.random() <= epsilon:  # if the random value is less than epsilon
            action = np.random.randint(0, env.action_space.n)  # explore
        
        else:  # otherwise...
            action = np.argmax(q_table[discrete_state])  # exploit the best move option at the current state as shown by the current q_table

        new_state, reward, done, _ = env.step(action)  # selected action generates 'new_state', 'reward', and 'done'(T/F)
        episode_reward += reward  # rewards accumulated during the episode so-far.
        new_discrete_state = get_discrete_state(new_state)  # descretise the continuous (default) new_state value 
        
        if render:  # screen is meant to render a session of game play every SHOW_EVERY episodes. (usually starved by the OS)
            env.render()
            # print(reward, new_state)
        
        if not done: 
            max_future_q = np.max(q_table[new_discrete_state])  # find the best option at state s'
            current_q = q_table[discrete_state + (action, )]  # collect the memoized q-table value
            # Now perform the q-update using collected values
            new_q = ((1-LEARNING_RATE) * current_q) + (LEARNING_RATE * (reward + (DISCOUNT * max_future_q)))  # 
            q_table[discrete_state + (action, )] = new_q  # update the current q-value (max determines policy) of the (state,action) pair q-table

        elif new_state[0] >= env.goal_position:  # Agent made it to the flag
            print(f"agent made it to the flag on episode: {episode}")
            num_wins += 1
            q_table[discrete_state + (action, )] = 0  # assign a "reward" of 0 (recall that cost of living is -1)
    
        discrete_state = new_discrete_state  # 
    
    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value  # update epsilon
        
    ep_rewards.append(episode_reward)  # append the episode rewards to the table
    if episode % SHOW_EVERY == 0:
        average_reward = sum(ep_rewards[-SHOW_EVERY:])/len(ep_rewards[-SHOW_EVERY:])
        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['avg'].append(average_reward)
        aggr_ep_rewards['min'].append(min(ep_rewards[-SHOW_EVERY:]))
        aggr_ep_rewards['max'].append(max(ep_rewards[-SHOW_EVERY:]))
        
        print(f"Episode: {episode}, avg: {average_reward}, min: {min(ep_rewards[-SHOW_EVERY:])}, max: {max(ep_rewards[-SHOW_EVERY:])}")
        
print(f"Agent won {num_wins} out of {EPISODES} games")
print("epsilon decay value used: ", epsilon_decay_value, "division factor: ", DIV_FACTOR)
env.close()

plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label="average rewards")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label="minimum rewards")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label="maximum rewards")
plt.legend(loc=4)
plt.show()