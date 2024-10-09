import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# configs = {
#     'env': 'CarRacing-v3',
#     'env_config': {
#         "render_mode": None,  # Disable rendering for faster collection
#         "continuous": True,
#         "domain_randomize": False,
#         "lap_complete_percent": 0.95,
#     },
#     'model': 'world_models.vae',
#     'model_config': {
#         'latent_dim': 32,
#         'encoder': {
#             'hidden_sizes': [256, 256],
#             'activation': 'relu'
#         },
#         'decoder': {
#             'hidden_sizes': [256, 256],
#             'activation': 'relu'
#         },
#     },
# }

# def process_and_save_image(env, num_episodes):
#     obs = []
#     for _ in tqdm(range(num_episodes)):
#         state = env.reset()
#         done = False
#         while not done:
#             action = env.action_space.sample()
#             state, _, done, _, _ = env.step(action)

#             obs.append(state)
#     obs = np.array(obs)
#     np.save('obs.npy', obs)

# def main():
#     env = gym.make(configs['env'], **configs['env_config'])
#     process_and_save_image(env, num_episodes=1)
#     env.close()

#     obs = np.load('obs.npy')
#     print(obs.shape)
#     for i in range(5):
#         plt.imshow(obs[i])
#         plt.show()

# if __name__ == '__main__':
#     main()

env = gym.make('CarRacing-v3')
env.reset()

def process_and_save_image(img, step):
    # Crop out the black bar at the bottom of the image and resize the entire image to 84x84
    cropped_img = img[:-12, :-12, :]
    plt.imsave(f"{step}.png", cropped_img)

# Number of steps to run for each scenario
num_steps = 100
env.reset()
for step in range(num_steps):
    img, _, done, _, _ = env.step(env.action_space.sample())
    if step > 50:  # Skip initial frames to avoid the zoomed-in images
        process_and_save_image(img, step)
    if done:
        break

env.close()