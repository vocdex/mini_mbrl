import mujoco
import mujoco.viewer
import numpy as np
from robot_descriptions.loaders.mujoco import load_robot_description

# Load the model
model = load_robot_description("panda_mj_description")
data = mujoco.MjData(model)

# Run simulation with viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    # Simulation loop
    while viewer.is_running():
        # Apply random actions
        # data.ctrl = np.random.uniform(-1, 1, size=12)

        # Null the action for the first 100 steps
        # if data.time < 0.1:
        data.ctrl = np.zeros(8)

        # Step the simulation
        mujoco.mj_step(model, data)

        # Update the viewer
        viewer.sync()
