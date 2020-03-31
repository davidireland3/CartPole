import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from collections import deque
import pickle

epsilon_max = 0.5
epsilon_min = 0.01
exploration_decay = 0.999
learning_rate = 0.001
batch_size = 20
gamma = 0.99
max_memory = 1000000
update_freq = 1000


class DQN:

    def __init__(self, observation_space, action_space):
        self.exploration_rate = epsilon_max
        self.observation_space = observation_space

        self.action_space = action_space

        self.memory = deque(maxlen=max_memory)
        
        self.model = self.make_model()
        self.target_network = self.make_model()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        self.exploration_rate *= exploration_decay
        self.exploration_rate = max(epsilon_min, self.exploration_rate)
        if np.random.uniform() < self.exploration_rate:
            return np.random.randint(0, self.action_space)
        q_values = self.model.predict(state)
        return np.argmax(q_values)

    def experience_replay(self):
        if len(self.memory) < batch_size:
            return
        indexes = np.random.choice([i for i in range(len(self.memory))], batch_size)
        batch = [self.memory[i] for i in indexes]
        for state, action, reward, state_next, terminal in batch:
            q_update = reward
            if not terminal:
                q_update = (reward + gamma * np.amax(self.target_network.predict(state_next)[0]))
            q_values = self.target_network.predict(state)
            q_values[0][action] = q_update
            self.model.fit(state, q_values, verbose=0)
            
    def make_model(self):
        model = Sequential()
        model.add(Dense(24, input_shape=(self.observation_space,), activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_space, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=learning_rate))
        return model
    
    def update_target(self):
        self.target_network.set_weights(self.model.load_weights())

def reward_giver(reward,terminal,steps):
    if terminal and steps < 200:
        return -1
    else:
        return reward


def cartpole(no_eps):
    score = []
    update_checker = 0
    env = gym.make("CartPole-v0")
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    dqn = DQN(observation_space, action_space)
    run = 0
    while run < no_eps:
        a = 0
        run += 1
        print("Episode number {}".format(run))
        state = env.reset()
        state = np.reshape(state, [1, observation_space])
        while True:
            update_checker += 1
            if update_checker % 1000 == 0:
                dqn.update_target()
            a += 1
            action = dqn.act(state)
            state_next, reward, terminal, info = env.step(action)
            # env.render()
            reward = reward_giver(reward,terminal,a)
            state_next = np.reshape(state_next, [1, observation_space])
            dqn.remember(state, action, reward, state_next, terminal)
            state = state_next
            if terminal:
                break
            dqn.experience_replay()
        score.append(a)
    return dqn,score


final_dqn,score = cartpole(300)

with open("/Users/Documents/cartpole_dqn","wb") as cartpole_file:
    pickle.dump(final_dqn,cartpole_file)

plt.plot(score)
file_path = "/Users/Documents/cartpole_score.pdf"
plt.savefig(file_path)
plt.show()

