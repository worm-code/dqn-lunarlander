import random
import numpy as np
import tensorflow as tf
import tensorflow.keras
import gym
from collections import namedtuple, deque
import matplotlib.pyplot as plt
import time


class DQNConfig:
    def __init__(self, env, alpha, gamma, NUM_STEPS_FOR_UPDATE, memory_size, epsilon, MINIBATCH_SIZE, TAU):
        self.state_size = env.observation_space.shape
        self.num_actions = env.action_space.n
        self.alpha = alpha
        self.gamma = gamma
        self.update = NUM_STEPS_FOR_UPDATE
        self.memory_size = memory_size
        self.epsilon = epsilon
        self.batch = MINIBATCH_SIZE
        self.tau = TAU

class DQNAgent:
    def __init__(self, configuration):
        self.config = configuration
        self.q_network = self.Build_Q_network()
        self.Target_q_network = self.Build_Q_network()
        self.update_target_network()  # 初始化两个网络的参数结构，使其相同
        self.memory_buffer = deque(maxlen=self.config.memory_size)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.alpha)

    def Build_Q_network(self):
        model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=self.config.state_size),
        tf.keras.layers.Dense(64, 'relu'),
        tf.keras.layers.Dense(64, 'relu'),
        tf.keras.layers.Dense(self.config.num_actions, 'linear')
        ])
        return model

    def update_target_network(self):
        self.Target_q_network.set_weights(self.q_network.get_weights())

    def compile(self, optimizer, loss_fn):
        self.q_network.compile(optimizer=optimizer, loss=loss_fn)

    def get_action(self, q_values):
        if random.random() > self.config.epsilon:
            return np.argmax(q_values.numpy()[0])
        else:
            return random.choice(np.arange(self.config.num_actions))

    def update_epsilon(self):
        self.config.epsilon = max(0.01, 0.995 * self.config.epsilon)


    # @tf.function
    def train(self, experiences):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(experiences)
        gradients = tape.gradient(loss, self.q_network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.q_network.trainable_variables))
        # self.update_target_q_network()
        self.update_target_network1()

    # @tf.function
    def compute_loss(self, experiences):
        states, actions, rewards, next_states, done_vals = experiences
        max_q_value = tf.reduce_max(self.Target_q_network(next_states), -1)
        y_target = rewards + self.config.gamma * max_q_value * (1-done_vals)
        q_values = self.q_network(states)
        q_values = tf.gather_nd(q_values, tf.stack([tf.range(q_values.shape[0]),
                                                    tf.cast(actions, tf.int32)], axis=1))  # gather_nd其主要功能是根据indices描述的索引，提取params上的元素
        loss = tf.keras.losses.MSE(y_target, q_values)
        return loss


    def update_target_q_network(self):
        # store_before = self.Target_q_network.weights
        # for target_weights, q_weights in zip(self.Target_q_network.weights, self.q_network.weights):
        #     target_weights.assign(self.config.tau * q_weights + (1.0 - self.config.tau) * target_weights)
        # print(f"更改后,Target_q_network{self.Target_q_network.weights}")
        # print("True" if store_before == store_after else "False")
        target_weights = []
        layer_num = 0
        for target_layer, q_layer in zip(self.Target_q_network.layers, self.q_network.layers):
            target_weights.extend([(1.0 - self.config.tau) * target_weight + self.config.tau * q_weight
                                   for target_weight, q_weight in
                                   zip(target_layer.get_weights(), q_layer.get_weights())])
            self.Target_q_network.layers[layer_num].set_weights(target_weights)
            layer_num += 1
            target_weights.clear()

        # layer_num = 0
        # for target_layer, q_layer in zip(self.Target_q_network.layers, self.q_network.layers):
        #     Target_weights = (1.0 - self.config.tau) * target_layer.weights[0].numpy() + self.config.tau * q_layer.weights[0].numpy()
        #     Target_bias = (1.0 - self.config.tau) * target_layer.weights[1].numpy() + self.config.tau * q_layer.weights[1].numpy()
        #     self.Target_q_network.layers[layer_num].set_weights([Target_weights, Target_bias])
        #     layer_num += 1

    def save_model(self):
        self.q_network.save('lunar_lander_model.h5')

    def load_model(self):
        dqn_model = tf.keras.models.load_model('lunar_lander_model.h5')
        return dqn_model

    # @tf.function
    def update_target_network1(self):
        # 后来发现了assign就能直接更改网络参数中的值有点和pytorch中的copy_差不多，会直接改变相应位置的值
        for target_weights, q_net_weights in zip(self.Target_q_network.weights, self.q_network.weights):
            target_weights.assign(TAU * q_net_weights + (1.0 - TAU) * target_weights)

if __name__ == '__main__':
    tf.random.set_seed(0)
    t1 = time.time()
    env = gym.make('LunarLander-v2', render_mode="rgb_array")

    ALPHA = 0.001
    gamma = 0.995
    NUM_STEPS_FOR_UPDATE = 4
    memory_size = 100000
    epsilon = 1.0
    num_episode = 1000
    max_num_timesteps = 1000
    MINIBATCH_SIZE = 64
    TAU = 1e-3
    num_av = 100


    experience = namedtuple("experience", "state action next_state reward done")

    config = DQNConfig(env, ALPHA, gamma, NUM_STEPS_FOR_UPDATE, memory_size, epsilon, MINIBATCH_SIZE, TAU)
    dqn = DQNAgent(config)
    points_history = []
    draw_data = []

    for i in range(num_episode):
        observation, info = env.reset()
        points = 0
        training_times = 0
        for j in range(max_num_timesteps):
            state_qn = np.expand_dims(observation, axis=0)
            q_values = dqn.q_network(state_qn)
            action = dqn.get_action(q_values)

            next_state, reward, done, step_limit_reached, info = env.step(action)
            dqn.memory_buffer.append(experience(observation, action, next_state, reward, done))

            if (training_times + 1) % dqn.config.update == 0 and len(dqn.memory_buffer) > dqn.config.batch:
                experiences = random.sample(dqn.memory_buffer, k=MINIBATCH_SIZE)
                states = tf.convert_to_tensor(np.array([e.state for e in experiences if e is not None]), dtype=tf.float32)
                actions = tf.convert_to_tensor(np.array([e.action for e in experiences if e is not None]), dtype=tf.float32)
                rewards = tf.convert_to_tensor(np.array([e.reward for e in experiences if e is not None]), dtype=tf.float32)
                next_states = tf.convert_to_tensor(np.array([e.next_state for e in experiences if e is not None]), dtype=tf.float32)
                done_vals = tf.convert_to_tensor(np.array([e.done for e in experiences if e is not None]).astype(np.uint8),
                                                 dtype=tf.float32)
                experiences_training = (states, actions, rewards, next_states, done_vals)
                dqn.train(experiences_training)
            training_times += 1
            points += reward
            observation = next_state.copy()
            if done:
                break
        dqn.update_epsilon()
        points_history.append(points)
        print(f'\rEpisode {i+1} | total_points: {points:.2f}', end="")

        if (i + 1) % num_av == 0:
            av_latest_points = np.mean(points_history[-num_av:])
            draw_data.append(av_latest_points)
            print(f'\rEpisode {i+1} | Total point average of the last {num_av} episodes: {av_latest_points:.2f}')
    tot_time = time.time() - t1
    print(f"\n程序运行结束，总耗时{tot_time:.2f} s ({tot_time / 60 :.2f}min)")
    # Plotting training performance
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
    plt.rcParams['axes.unicode_minus'] = False
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))

    # X-axis values
    x_data = np.arange(1, len(draw_data) + 1) * num_av

    # Plot smoothed data (e.g., every 100 episodes)
    ax.scatter(x_data, draw_data, marker='o', s=50, edgecolors='b', facecolors='white', label="每100回合平均得分", alpha=0.8)
    ax.plot(x_data, draw_data, linestyle='-.', linewidth=2, color='#00BFFF')

    # Plot raw data (every episode)
    x_raw = np.arange(1, len(points_history) + 1)
    ax.plot(x_raw, points_history, color='lightgray', linewidth=1, alpha=0.6, label="每回合得分")

    ax.set_title("DQN在LunarLander-v2环境中的训练得分", fontsize=14)
    ax.set_xlabel("回合数", fontsize=12)
    ax.set_ylabel("得分", fontsize=12)
    ax.grid(True, linestyle='--', linewidth=1, alpha=0.7)
    ax.legend(fontsize=10, loc='upper left')
    ax.tick_params(axis='both', which='major', labelsize=10)
    plt.xticks(np.insert(x_data, 0, 0), np.insert(x_data, 0, 0))
    plt.tight_layout()
    plt.savefig('training_curve.png', dpi=300)
    plt.show()
    plt.close('all')
    dqn.save_model()
