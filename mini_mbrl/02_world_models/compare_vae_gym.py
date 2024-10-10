""" Compare the original and reconstructed frames using a trained VAE in the CarRacing-v3 environment with keyboard input. """
import gymnasium as gym
import torch
import numpy as np
import os
import cv2
from torchvision import transforms
from vae import VanillaVAE

# Define the pre-processing function to match the VAE input requirements
def preprocess_frame(frame):
    """Crop and resize the frame to 64x64 to remove game status and black lines."""
    cropped_frame = frame[:-80, :-80, :]
    resized_frame = cv2.resize(cropped_frame, (64, 64), interpolation=cv2.INTER_AREA)
    return resized_frame

# Global variables to store the current action and steering
current_action = [0, 0, 0]
steering_value = 0
STEERING_SPEED = 0.1
RETURN_SPEED = 0.2

def get_action(keys):
    global current_action, steering_value
    
    # Reset acceleration and braking
    current_action[1] = 0  # Acceleration
    current_action[2] = 0  # Braking

    if ord('w') in keys:
        current_action[1] = 1    # Accelerate
    if ord('s') in keys:
        current_action[2] = 0.8  # Brake
    
    # Handle steering
    if ord('a') in keys:
        steering_value = max(-1, steering_value - STEERING_SPEED)
    elif ord('d') in keys:
        steering_value = min(1, steering_value + STEERING_SPEED)
    else:
        # Return to center when no steering input
        if steering_value > 0:
            steering_value = max(0, steering_value - RETURN_SPEED)
        elif steering_value < 0:
            steering_value = min(0, steering_value + RETURN_SPEED)
    
    current_action[0] = steering_value
    
    return current_action

def main(configs, record_gif=False):
    # Initialize the gym environment
    env = gym.make('CarRacing-v3', render_mode='rgb_array')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the VAE model
    vae = configs['VAE'](in_channels=3, **configs).to(device)
    model_name = f"vae_{configs['num_epochs']}.pth"
    vae.load_state_dict(torch.load(os.path.join(configs['model_dir'], model_name), map_location=device, weights_only=True))
    vae.eval()

    obs, _ = env.reset()
    done = False

    frames = []

    while not done:
        # Render the environment and preprocess the frame
        frame = env.render()
        preprocessed_frame = preprocess_frame(frame).transpose((2, 0, 1))
        preprocessed_frame = torch.tensor(preprocessed_frame, dtype=torch.float32).unsqueeze(0).to(device) / 255.0

        # Pass the preprocessed frame through the VAE
        with torch.no_grad():
            reconstructed_frame, _, _, _ = vae(preprocessed_frame)
            reconstructed_frame = reconstructed_frame.squeeze(0).cpu().numpy().transpose((1, 2, 0))

        # Convert back to uint8 for display purposes
        original_frame_display = preprocess_frame(frame)
        reconstructed_frame_display = (reconstructed_frame * 255).astype(np.uint8)

        # Stack the frames horizontally for side-by-side display
        combined_frame = np.hstack((original_frame_display, reconstructed_frame_display))
        combined_frame = cv2.putText(combined_frame, 'Original', (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255), 1, cv2.LINE_AA)
        combined_frame = cv2.putText(combined_frame, 'Reconstructed', (original_frame_display.shape[1] + 10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255), 1, cv2.LINE_AA)

        # Display the combined frame
        cv2.imshow('Original and Reconstructed Frames', combined_frame)

        if record_gif:
            frames.append(combined_frame)

        key = cv2.waitKey(1)

        if key == ord('q'):
            break

        # Get the action based on keyboard input and convert it to a NumPy array
        action = np.array(get_action([key]), dtype=np.float32)
        obs, reward, done, truncated, _ = env.step(action)

        if done or truncated:
            obs, _ = env.reset()

    
    env.close()
    cv2.destroyAllWindows()
    if record_gif and frames:
        save_gif(frames, f"{configs['model_dir']}/car_racing.gif")
        print(f"Saved GIF to {configs['model_dir']}/car_racing.gif")

def save_gif(frames, filename):
    import imageio
    imageio.mimsave(filename, frames, fps=30)

# Example configs dictionary (replace with your actual settings)
configs = {
    'VAE': VanillaVAE,
    'num_epochs': 50,
    'latent_dim': 32,
    'model_dir': './models'
}

if __name__ == "__main__":
    main(configs, record_gif=True)

