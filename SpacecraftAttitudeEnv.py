import gymnasium as gym
import numpy as np
import sys
import os

# Get the absolute path to the 'basilisk' directory
basilisk_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "basilisk"))
sys.path.insert(0, basilisk_path)  # Insert at the beginning of sys.path

# The path to the location of Basilisk
from Basilisk import __path__
bskPath = __path__[0]
fileName = os.path.basename(os.path.splitext(__file__)[0])

# Basilisk imports
from Basilisk.utilities import SimulationBaseClass
from Basilisk.simulation import spacecraft
from Basilisk.simulation import thrusterDynamicEffector
from Basilisk.utilities import macros, unitTestSupport
from Basilisk.utilities import vizSupport
from Basilisk.architecture import messaging
from Basilisk.utilities import simIncludeThruster


class SpacecraftAttitudeEnv(gym.Env):
    """
    A Gymnasium environment for spacecraft attitude control using on/off thrusters.
    Thrusters are commanded via on-time requests, matching the style in scenarioBasicOrbitStream.py.
    """
    def __init__(
        self,
        step_size=0.1,
        mass_properties=None,
        thruster_config=None,
        viz_file=None  # Pass a filename here to create a .viz file
    ):
        super().__init__()

        # Store configuration
        self.step_size = step_size


        mass = 500.0  # kg
        # For a cuboid with length (l)=2.0, width (w)=1.5, height (h)=1.0
        Ixx = (1/12) * mass * (1.0**2 + 1.5**2)   # ~135.42
        Iyy = (1/12) * mass * (2.0**2 + 1.0**2)   # ~208.33
        Izz = (1/12) * mass * (2.0**2 + 1.5**2)   # ~260.42
        mass_properties = {
            'mass': mass,
            'inertia': [
                [Ixx, 0.0, 0.0],
                [0.0, Iyy, 0.0],
                [0.0, 0.0, Izz]
            ]
        }
        self.mass_properties = mass_properties        
        thruster_config = {
            'locations': [
                # Thrusters on the +X face
                [ 1.0,  0.5,  0.5],
                [ 1.0, -0.5, -0.5],
                # Thrusters on the -X face
                [-1.0,  0.5, -0.5],
                [-1.0, -0.5,  0.5],
                # Thrusters on the +Y face
                [ 0.5,  0.75,  0.5],
                [-0.5,  0.75, -0.5],
                # Thrusters on the -Y face
                [ 0.5, -0.75, -0.5],
                [-0.5, -0.75,  0.5],
                # Thrusters on the +Z face
                [ 0.5,  0.5,  0.75],
                [-0.5, -0.5,  0.75],
                # Thrusters on the -Z face
                [ 0.5, -0.5, -0.75],
                [-0.5,  0.5, -0.75]
            ],
            'directions': [
                # Directions are the outward normals of the faces
                # +X face
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                # -X face
                [-1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                # +Y face
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                # -Y face
                [0.0, -1.0, 0.0],
                [0.0, -1.0, 0.0],
                # +Z face
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                # -Z face
                [0.0, 0.0, -1.0],
                [0.0, 0.0, -1.0],
            ],
            'max_thrust': 5.0,   # Newtons; adjust based on your actuation requirements
            'min_thrust': 0.0
        }

        self.thruster_config = thruster_config

        # --- Reward Weights ---
        self.reward_weights = {'attitude': 1.0, 'angular': 0.1}        

        # Define action/observation spaces
        n_thrusters = len(self.thruster_config['locations'])
        # 2**n_thrusters = each thruster on/off -> discrete
        self.action_space = gym.spaces.Discrete(2**n_thrusters)

        # MRP (3) + angular velocity (3) => 6D observation
        self.observation_space = gym.spaces.Box(
            low=-10.0, high=10.0, shape=(6,), dtype=np.float32
        )

        self.viz_file = viz_file  # None => no .viz file

        # Initialize Basilisk simulation
        self._setup_simulation()

    def _setup_simulation(self):
        """
        1) Create sim instance, process, and task
        2) Create spacecraft + thrusters
        3) Add spacecraft & thruster set to the task
        4) (Optional) Vizard .viz output
        """
        # 1) Create the simulation instance
        self.scSim = SimulationBaseClass.SimBaseClass()

        simProcessName = "simProcess"
        simTaskName    = "simTask"

        # Create a new process and a task with time step = self.step_size
        dynProcess = self.scSim.CreateNewProcess(simProcessName)
        timeStepNanos = macros.sec2nano(self.step_size)
        dynProcess.addTask(self.scSim.CreateNewTask(simTaskName, timeStepNanos))

        # 2) Create the spacecraft object
        self.scObject = spacecraft.Spacecraft()
        self.scObject.ModelTag = "spacecraft"

        # Set mass + inertia
        inertia_matrix = np.array(self.mass_properties['inertia'], dtype=float).flatten()
        self.scObject.hub.mHub = self.mass_properties['mass']
        self.scObject.hub.IHubPntBc_B = unitTestSupport.np2EigenMatrix3d(inertia_matrix)

        # 3) Create the thruster set and add it as a separate model
        self.thrusterSet = thrusterDynamicEffector.ThrusterDynamicEffector()
        self.thrusterSet.ModelTag = "thrusterSet"
        self.scSim.AddModelToTask(simTaskName, self.thrusterSet)

        # 4) Use thruster factory to create thrusters
        self.thFactory = simIncludeThruster.thrusterFactory()

        # Example: If you want to choose between different thruster types:
        for loc, direc in zip(self.thruster_config['locations'], self.thruster_config['directions']):
                self.thFactory.create('MOOG_Monarc_1', loc, direc)

        # 5) Tie thrusters to the spacecraft
        thrModelTag = "ACSThrusterDynamics"
        self.thFactory.addToSpacecraft(thrModelTag, self.thrusterSet, self.scObject)

        # **Create the actual on-time command message object**
        self.thrusterOnTimeMsg = messaging.THRArrayOnTimeCmdMsg()

        # This is your data payload
        n_thrusters = self.thFactory.getNumOfDevices()
        self.thrusterCmdMsg = messaging.THRArrayOnTimeCmdMsgPayload()
        self.thrusterCmdMsg.OnTimeRequest = [0.0]*n_thrusters

        # Set initial zero ontime message
        self.thrusterOnTimeMsg.write(self.thrusterCmdMsg)

        # **Subscribe the thruster set to read from this message**
        self.thrusterSet.cmdsInMsg.subscribeTo(self.thrusterOnTimeMsg)

        # 8) Add the spacecraft to the simulation task
        self.scSim.AddModelToTask(simTaskName, self.scObject)

        self._setup_logging(simTaskName)

        # (Optional) Vizard .viz file
        if self.viz_file is not None:
            vizSupport.enableUnityVisualization(
                self.scSim,
                simTaskName,
                self.scObject,
                thrEffectorList=self.thrusterSet,
                saveFile=self.viz_file
            )


    def _setup_logging(self, simTaskName):
        
        # Setup logging for attitude and angular velocity
        self.scStateLogger = self.scObject.scStateOutMsg.recorder()
        self.scSim.AddModelToTask(simTaskName, self.scStateLogger)

        # Setup logging for thruster commands
        self.thrusterLogger = self.thrusterSet.cmdsInMsg.recorder()
        self.scSim.AddModelToTask(simTaskName, self.thrusterLogger)


    def reset(self, seed=None, options=None):
        """
        Reset environment for a new episode.
        Gymnasium requires returning (obs, info).
        """
        super().reset(seed=seed)

        # Re-init Basilisk
        self.scSim.InitializeSimulation()

        # Clear old logs
        self.scStateLogger.clear()

        # Random initial MRP and angular velocity
        sigma0 = np.random.uniform(-0.3, 0.3, 3)
        #omega0 = np.random.uniform(-0.1, 0.1, 3)
        omega0 = np.array([0.0, 0.0, 0.0]) # initial zero angular velocity

        self.scObject.hub.sigma_BNInit = sigma0
        self.scObject.hub.omega_BN_BInit = omega0

        self.current_time = 0.0

        return self._get_observation(), {}

    def step(self, action):
        """
        Gymnasium step => (obs, reward, terminated, truncated, info)
        """
        start_nanos = macros.sec2nano(self.current_time)
        end_nanos   = macros.sec2nano(self.current_time + self.step_size)
        self.next_stop_time = end_nanos

        self._apply_action(action)

        # Advance Basilisk by step_size
        self.scSim.ConfigureStopTime(end_nanos)
        self.scSim.ExecuteSimulation()
        self.current_time += self.step_size

        obs = self._get_observation()
        reward = self._compute_reward(obs)
        terminated = self._check_done(obs)
        truncated = False  # or add your own time limit

        # Create info for custom metrics
        info = {
            'attitude_error': np.linalg.norm(obs[:3]),
            'angular_velocity': np.linalg.norm(obs[3:]),
        }

        return obs, reward, terminated, truncated, info

    def _get_observation(self):
        """Read MRP and angular velocity from scStateOutMsg."""
        stateMsg = self.scObject.scStateOutMsg.read()
        sigma = np.array(stateMsg.sigma_BN, dtype=np.float32)      # 3D MRP
        omega = np.array(stateMsg.omega_BN_B, dtype=np.float32)    # 3D angular velocity

        #print("Angular velocity: ", omega)

        obs = np.concatenate([sigma, omega], axis=0)
        return obs

    def _apply_action(self, action):
        """
        Convert discrete action bits => on-time requests.
        Each thruster is on for 0.1s if bit=1, else 0.0s.
        """

        n_thrusters = len(self.thrusterCmdMsg.OnTimeRequest)
        commands = [(action >> i) & 1 for i in range(n_thrusters)]

        # Prepare on-time array
        thruster_ontimes = [0.0] * n_thrusters
        for i, cmd in enumerate(commands):
            thruster_ontimes[i] = cmd * self.step_size

        self.thrusterCmdMsg.OnTimeRequest = thruster_ontimes

        # Write the updated on-time commands
        self.thrusterOnTimeMsg.write(self.thrusterCmdMsg, self.next_stop_time)

    def _compute_reward(self, obs):
        """
        Computes the reward as the negative weighted sum of the attitude (MRP) error and angular velocity magnitude.
        The weights can be adjusted via self.reward_weights.
        """
        sigma = obs[:3]
        omega = obs[3:]
        sigma_norm = np.linalg.norm(sigma)
        omega_mag  = np.linalg.norm(omega)
        # Negative penalty: you can adjust the weights to emphasize one error term over the other.
        reward = - (self.reward_weights['attitude'] * sigma_norm**2 +
                    self.reward_weights['angular']  * omega_mag**2)
        return reward

    def _check_done(self, obs):
        """
        Terminate if near zero MRP and near zero angular velocity,
        or if it spins too fast.
        """
        sigma = obs[:3]
        omega = obs[3:]
        sigma_norm = np.linalg.norm(sigma)
        omega_mag  = np.linalg.norm(omega)

        # success if stable
        if sigma_norm < 0.01 and omega_mag < 0.01:
            return True
        # fail if spinning out of control
        if omega_mag > 5.0:
            return True

        return False