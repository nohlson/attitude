import os
import json
import numpy as np
import matplotlib.pyplot as plt
import datetime
from stable_baselines3 import PPO
from SpacecraftAttitudeEnv import SpacecraftAttitudeEnv
from stable_baselines3.common.logger import configure
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.callbacks import BaseCallback


# -------------------------------
# Define a Custom Callback
# -------------------------------
class CustomMetricsCallback(BaseCallback):
    """
    A custom callback that collects and logs the attitude error and angular velocity
    from the info dictionary provided by your environment.
    """
    def __init__(self, verbose=0):
        super(CustomMetricsCallback, self).__init__(verbose)
        self.att_errors = []
        self.angular_vels = []

    def _on_step(self) -> bool:
        # Get infos from the current step (works for both vectorized and single envs)
        infos = self.locals.get("infos", None)
        if infos is None:
            info = self.locals.get("info", {})
            infos = [info]
        for info in infos:
            if "attitude_error" in info:
                self.att_errors.append(info["attitude_error"])
            if "angular_velocity" in info:
                self.angular_vels.append(info["angular_velocity"])
        return True

    def _on_rollout_end(self) -> None:
        # Compute average metrics for this rollout
        avg_att_error = np.mean(self.att_errors) if self.att_errors else 0
        avg_ang_vel = np.mean(self.angular_vels) if self.angular_vels else 0

        # Record these custom metrics under the "custom" namespace so they appear in TensorBoard.
        self.logger.record("custom/attitude_error", avg_att_error)
        self.logger.record("custom/angular_velocity", avg_ang_vel)

        # Reset for the next rollout
        self.att_errors = []
        self.angular_vels = []

def create_output_directory():
    """
    Creates a new directory for training artifacts.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join("training_runs", f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def save_training_metadata(output_dir, env_config, train_config):
    """
    Saves training configuration to a JSON file.
    """
    config = {
        "environment": env_config,
        "training": train_config
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

def plot_logs(env, output_dir):
    """
    Saves plots of attitude, angular velocity, and thruster activations.
    """
    log = env.scStateLogger
    thruster_log = env.thrusterLogger  # Thruster log

    time = np.array(log.times()) * 1e-9
    sigma = np.array(log.sigma_BN)
    sigma_norm = np.linalg.norm(sigma, axis=1)
    omega = np.array(log.omega_BN_B)
    omega_mag = np.linalg.norm(omega, axis=1)
    thruster_on_times = np.array(thruster_log.OnTimeRequest)

    # Plot Attitude (MRP)
    plt.figure(figsize=(12, 6))

    plt.subplot(5, 1, 1)
    plt.plot(time, sigma[:, 0], label="σ₁")
    plt.plot(time, sigma[:, 1], label="σ₂")
    plt.plot(time, sigma[:, 2], label="σ₃")
    plt.title("Attitude (σ)")
    plt.legend()
    plt.ylabel("MRP Components")

    # Plot attitude error
    plt.subplot(5, 1, 2)
    plt.plot(time, sigma_norm)
    plt.title("Attitude Error")
    plt.ylabel("σ Error")

    # Plot Angular Velocity
    plt.subplot(5, 1, 3)
    plt.plot(time, omega[:, 0], label="ω₁")
    plt.plot(time, omega[:, 1], label="ω₂")
    plt.plot(time, omega[:, 2], label="ω₃")
    plt.title("Angular Velocity (rad/s)")
    plt.legend()
    plt.ylabel("AngVel ω")

    # Plot angular velocity magnitude
    plt.subplot(5, 1, 4)
    plt.plot(time, omega_mag)
    plt.title("Angular Velocity Magnitude (rad/s)")
    plt.ylabel("ω mag")
    

    # Plot Thruster Firings
    plt.subplot(5, 1, 5)
    for i in range(4):
        plt.step(time, thruster_on_times[:, i], label=f"Thruster {i}")
    plt.xlabel("Time (s)")
    plt.ylabel("On-Time (s)")
    plt.legend()
    plt.title("Thruster Firings")

    plt.savefig(os.path.join(output_dir, "thruster_firing_plot.png"))
    plt.close()

def save_logs(env, output_dir):
    """
    Saves attitude, angular velocity, and thruster firing logs to CSV.
    """
    log = env.scStateLogger
    thruster_log = env.thrusterLogger  # Access thruster logs

    time = np.array(log.times()) * 1e-9  # Convert nanoseconds to seconds
    sigma = np.array(log.sigma_BN)
    omega = np.array(log.omega_BN_B)
    thruster_on_times = np.array(thruster_log.OnTimeRequest)  # Thruster on-time commands

    # Create angular velocity magnitude log
    omega_mag = np.linalg.norm(omega, axis=1)

    # Save attitude log
    attitude_data = np.column_stack([time, sigma, omega])
    np.savetxt(os.path.join(output_dir, "attitude_log.csv"), attitude_data, delimiter=",",
               header="Time(s), Sigma_1, Sigma_2, Sigma_3, Omega_1, Omega_2, Omega_3", comments="")
    
    # Create attitude error log
    sigma_norm = np.linalg.norm(sigma, axis=1)
    attitude_error_data = np.column_stack([time, sigma_norm])
    np.savetxt(os.path.join(output_dir, "attitude_error_log.csv"), attitude_error_data, delimiter=",",
                header="Time(s), Attitude_Error", comments="")
    
    # Save angular velocity magnitude log
    omega_mag_data = np.column_stack([time, omega_mag])
    np.savetxt(os.path.join(output_dir, "angular_velocity_log.csv"), omega_mag_data, delimiter=",",
               header="Time(s), Omega_Magnitude", comments="")
    

    # Save thruster log
    thruster_data = np.column_stack([time] + [thruster_on_times[:, i] for i in range(thruster_on_times.shape[1])])
    thruster_header = "Time(s)," + ",".join([f"Thruster_{i}" for i in range(thruster_on_times.shape[1])])
    np.savetxt(os.path.join(output_dir, "thruster_log.csv"), thruster_data, delimiter=",", header=thruster_header, comments="")

def main():
    """
    Runs training and saves all artifacts.
    """
    # Create output directory
    output_dir = create_output_directory()
    print(f"Training output directory: {output_dir}")

    # Define environment
    env = SpacecraftAttitudeEnv(step_size=0.1, viz_file=os.path.join(output_dir, "test_output.viz"))
    #env = SpacecraftAttitudeEnv(step_size=0.1, viz_file=None)
    envTL = TimeLimit(env, max_episode_steps=1000)

    # Training configuration
    train_config = {
        "total_timesteps": 50000,
        "learning_rate": 0.0003,
        "policy": "MlpPolicy"
    }

    # Save environment and training configuration
    env_config = {
        "step_size": env.step_size,
        "mass_properties": env.mass_properties,
        "thruster_config": env.thruster_config
    }
    save_training_metadata(output_dir, env_config, train_config)


    # Train model
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=train_config["learning_rate"])
    # Configure SB3 logging
    new_logger = configure(folder="logs/", format_strings=["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)

    # Custom callback for monitoring training
    custom_callback = CustomMetricsCallback()

    # Start training
    model.learn(total_timesteps=train_config["total_timesteps"], callback=custom_callback)

    # Save trained model
    model_path = os.path.join(output_dir, "trained_model.zip")
    model.save(model_path)
    print(f"Model saved to {model_path}")

    # # Run a test episode
    # obs, _ = env.reset()
    # done = False
    # while not done:
    #     action, _ = model.predict(obs)
    #     obs, _, done, _, _ = env.step(action)

    # Save logs and plots
    save_logs(env, output_dir)
    plot_logs(env, output_dir)

    print(f"Training artifacts saved in: {output_dir}")

if __name__ == "__main__":
    main()