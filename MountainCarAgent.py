import gymnasium as gym 
import numpy as np
import matplotlib.pyplot as plt
import pickle  # used to save q-table between training runs. "pickling" converts a Python object hierarchy into a byte stream, whereas "unpickling" reverses this operation.

# The MountainCar-v0 world is a continuous state MDP model. To handle this, we need to descretize each action the agent takes in the state-space graph
# the 'is_training' flag decides if the model is in training, or pre-trained. 'render' controls the display
def run(num_episodes, is_training=True, render=False):
    
    # Initiate the environment
    env = gym.make('MountainCar-v0', render_mode='human' if render else None)
    NUM_BINS = 26  # 16 bins is optimal discretization for the MountainCar-v0 state space graph
    
    # TUNING PARAMETERS
    alpha = 0.15  # learning rate
    discount_factor = 0.95  # 
    epsilon = 0.5  # 50% random to start
    
    DIV_FACTOR = 30  # try 2,3,4,...  larger value == larger delta_epsilon "chunks"
    END_EPSILON_DECAYING = num_episodes // DIV_FACTOR  # determines size of the denominator in 'epsilon_decay_value'
    
    epsilon_decay_rate = epsilon/((END_EPSILON_DECAYING) - 1)  # normalized epsilon decay value
    # epsilon_decay_rate = 2/num_episodes  # alternative method
    
    # np.linspace(range_low, range_high, #_of_points) returns evenly spaced numbers covering a specified interval. A handy descretization tool.
    pos_space = np.linspace(env.observation_space.low[0], env.observation_space.high[0], NUM_BINS)  # descretized position val, range: [-1.2, 0.6]
    vel_space = np.linspace(env.observation_space.low[1], env.observation_space.high[1], NUM_BINS)  # descretized velocity val, range: [-0.07, 0.07]
    
    # Initialize the q_table, a memoization table used to exploit the best move option
    if(is_training):  # creating a new q-table
        q_table = np.zeros((len(pos_space), len(vel_space), env.action_space.n))  # pos_space * vel_space * NUM_BINS
    else: 
        file = open('mountain_car.pkl', 'rb')
        q_table = pickle.load(file)  # load the trained q-table
        file.close()
        
    random_number_generator = np.random.default_rng() 
    rewards_per_episode = np.zeros(num_episodes)  # table that stores rewards accumulated in each episode of gameplay
    
    
    for i in range(num_episodes):
        # Start a new game (episode)
        state = env.reset()[0]  # generate the starting state and initiate the model environment
        state_p = np.digitize(state[0], pos_space)  # np.digitize(val, linspace) returns the bin that 'val' belongs in, relative to linspace
        state_v = np.digitize(state[1], vel_space)
        
        terminated = False
        rewards = 0
        # Play the game until it ends
        while(not terminated and rewards>-1000):  # -1 living reward
            
            if is_training and random_number_generator.random() < epsilon:  # explore
                action = env.action_space.sample()  # choose a random action: [0: go-left, 1: neutral, 2: go-right]
            else:  # exploit
                action = np.argmax(q_table[state_p, state_v, :])  # find the best option given current position and velocity
            
            new_state, reward, terminated, _, _ = env.step(action)  # execute the action to receive reward and new state
            new_state_p = np.digitize(new_state[0], pos_space)  # find bin
            new_state_v = np.digitize(new_state[1], vel_space)
            
            if is_training:  # update the q_table [there are several ways you can do this], only updated during training sessions
                q_table[state_p, state_v, action] = q_table[state_p, state_v, action] + alpha*(reward + discount_factor*np.max(q_table[new_state_p, new_state_v, :]) - q_table[state_p, state_v, action])
            state = new_state  # format => (position, velocity)
            state_p = new_state_p
            state_v = new_state_v
            
            rewards += reward  # update accumulated rewards
        
        epsilon = max(epsilon - epsilon_decay_rate, 0)  # ensure that epsilon is never negative to avoid undefined behavior
        rewards_per_episode[i] = rewards  # 
        
    env.close()
    # Training run is over, now we tabulate the results
    
    if is_training:
        f = open('mountain_car.pkl', 'wb')
        pickle.dump(q_table, f)  # place the q_table in the file
        f.close()
    
    mean_rewards = np.zeros(num_episodes)  # initialize the mean_rewards table
    for game in range(num_episodes):
        # mean_rewards = np.mean(rewards_per_episode[game])  # tablulate every game
        mean_rewards[game] = np.mean(rewards_per_episode[max(0, game-100):(game+1)])  # tabulate every 100th game
        
    # Include the following in saved figure: NUM_BINS, alpha, discount_factor, epsilon, epsilon_decay_rate, 
    plt.plot(mean_rewards)
    plt.xlabel("Game Episodes")
    plt.ylabel("Accrued Reward ()")
    plt.savefig(f'mountain_car.png')
    
    
if __name__ == '__main__':
    run(10000, is_training=False, render=False)