import tensorflow as tf
import tensorflow.keras
import numpy as np
import gym

class DQNVisualizer:
    def __init__(self, model_path='lunar_lander_model.h5'):
        # Load trained Q-network
        self.model = tf.keras.models.load_model(model_path)
        self.env = gym.make('LunarLander-v2', render_mode="human")

    def run_episode(self):
        observation, info = self.env.reset()
        done = False
        total_reward = 0
        step = 0
        while not done:
            obs_tensor = np.expand_dims(observation, axis=0).astype(np.float32)
            q_values = self.model(obs_tensor)
            action = np.argmax(q_values.numpy()[0])
            observation, reward, done, truncated, info = self.env.step(action)
            total_reward += reward
            step += 1
        print(f"Episode finished in {step} steps. Total return: {total_reward:.2f}")
        self.env.close()

if __name__ == "__main__":
    model_weight_path = r'lunar_lander_model.h5'
    visualizer = DQNVisualizer(model_weight_path)
    visualizer.run_episode()
