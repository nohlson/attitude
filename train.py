from stable_baselines3 import PPO
from SpacecraftAttitudeEnv import SpacecraftAttitudeEnv
import gymnasium as gym

def main():
    # Create environment WITHOUT VIZ (faster for training)
    env = SpacecraftAttitudeEnv(
        step_size=0.1,
        viz_file=None  # no Vizard file during training
    )

    # Wrap with Gymnasium's vectorized API (important for compatibility)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    
    # Create RL model
    model = PPO("MlpPolicy", env, verbose=1)
    
    # Train for some timesteps (example: 50,000)
    model.learn(total_timesteps=50000)
    
    # Save trained model
    model.save("my_trained_model")
    print("Model saved as my_trained_model.zip")

if __name__ == "__main__":
    main()