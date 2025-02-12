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
        self.mass_properties = mass_properties or {
            'mass': 1000.0,  # kg
            'inertia': [
                [1000.0, 0.0,   0.0],
                [0.0,   800.0,  0.0],
                [0.0,   0.0,    600.0]
            ]  # kg*m^2
        }
        self.thruster_config = thruster_config or {
            'locations': [
                [1.0, 1.0, 0.0],     # Thruster 1
                [-1.0, 1.0, 0.0],    # Thruster 2
                [-1.0, -1.0, 0.0],   # Thruster 3
                [1.0, -1.0, 0.0]     # Thruster 4
            ],
            'directions': [
                [0.0, 0.0, 1.0],     # +Z
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
                [0.0, 0.0, -1.0]
            ],
            'max_thrust': 10.0,   # Newtons
            'min_thrust': 0.0
        }

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

        # Create thruster effector
        self.thrusterSet = thrusterDynamicEffector.ThrusterDynamicEffector()
        self.thrusterSet.ModelTag = "thrusterSet"

        # We'll also add thrusterSet as a separate model to the task, as seen in scenario examples
        self.scSim.AddModelToTask(simTaskName, self.thrusterSet)

        # 3) Add thrusters
        #    Use Basilisk's THRSimConfig for each thruster
        n_thrusters = len(self.thruster_config['locations'])
        for loc, direc in zip(self.thruster_config['locations'], self.thruster_config['directions']):
            thr_config = thrusterDynamicEffector.THRSimConfig()
            thr_config.thrLoc_B = thrusterDynamicEffector.DoubleVector(list(loc))
            thr_config.thrDir_B = thrusterDynamicEffector.DoubleVector(list(direc))
            thr_config.MaxThrust = self.thruster_config['max_thrust']
            self.thrusterSet.addThruster(thr_config)

        # Option A: Attach thrusterSet to the spacecraft object
        self.scObject.addDynamicEffector(self.thrusterSet)

        # 4) Create the on-time command message
        # This is the recommended approach in scenarioBasicOrbitStream
        self.n_thrusters = n_thrusters
        self.thrusterOnTimeData = messaging.THRArrayOnTimeCmdMsgPayload()
        self.thrusterOnTimeData.OnTimeRequest = [0.0] * n_thrusters

        # Create a Msg object to write these commands
        self.thrusterOnTimeMsg = messaging.THRArrayOnTimeCmdMsg()
        # Subscribe thrusterSet to read from this on-time message
        self.thrusterSet.cmdsInMsg.subscribeTo(self.thrusterOnTimeMsg)

        # Add the spacecraft object to the task
        self.scSim.AddModelToTask(simTaskName, self.scObject)

        # (Optional) Vizard .viz file
        if self.viz_file is not None:
            vizSupport.enableUnityVisualization(
                self.scSim,
                simTaskName,
                self.scObject,
                saveFile=self.viz_file
            )

    def reset(self, seed=None, options=None):
        """
        Reset environment for a new episode.
        Gymnasium requires returning (obs, info).
        """
        super().reset(seed=seed)

        # Re-init Basilisk
        self.scSim.InitializeSimulation()

        # Random initial MRP and angular velocity
        sigma0 = np.random.uniform(-0.3, 0.3, 3)
        omega0 = np.random.uniform(-0.1, 0.1, 3)

        self.scObject.hub.sigma_BNInit = sigma0
        self.scObject.hub.omega_BN_BInit = omega0

        self.current_time = 0.0

        return self._get_observation(), {}

    def step(self, action):
        """
        Gymnasium step => (obs, reward, terminated, truncated, info)
        """
        self._apply_action(action)

        # Advance Basilisk by step_size
        start_nanos = macros.sec2nano(self.current_time)
        end_nanos   = macros.sec2nano(self.current_time + self.step_size)
        self.scSim.ConfigureStopTime(end_nanos)
        self.scSim.ExecuteSimulation()
        self.current_time += self.step_size

        print("Stepping {} seconds from {} to {}".format(self.step_size, start_nanos, end_nanos))

        obs = self._get_observation()
        reward = self._compute_reward(obs)
        terminated = self._check_done(obs)
        truncated = False  # or add your own time limit

        return obs, reward, terminated, truncated, {}

    def _get_observation(self):
        """Read MRP and angular velocity from scStateOutMsg."""
        stateMsg = self.scObject.scStateOutMsg.read()
        sigma = np.array(stateMsg.sigma_BN, dtype=np.float32)      # 3D MRP
        omega = np.array(stateMsg.omega_BN_B, dtype=np.float32)    # 3D angular velocity

        obs = np.concatenate([sigma, omega], axis=0)
        return obs

    def _apply_action(self, action):
        """
        Convert discrete action bits => on-time requests.
        Each thruster is on for 0.1s if bit=1, else 0.0s.
        """
        commands = [(action >> i) & 1 for i in range(self.n_thrusters)]

        # Prepare on-time array
        for i, cmd in enumerate(commands):
            self.thrusterOnTimeData.OnTimeRequest[i] = 0.1 if cmd else 0.0

        # Write the updated on-time commands
        self.thrusterOnTimeMsg.write(self.thrusterOnTimeData)

    def _compute_reward(self, obs):
        """
        Reward is negative of attitude error + angular velocity penalty.
        """
        sigma = obs[:3]
        omega = obs[3:]
        sigma_norm = np.linalg.norm(sigma)
        omega_mag  = np.linalg.norm(omega)

        # Simple penalty: sum of squares
        reward = - (sigma_norm**2 + 0.1 * (omega_mag**2))

        print("Current attitude error: {:.3f}, angular velocity: {:.3f}".format(sigma_norm, omega_mag))
        print("Current angular velocity: {}".format(omega))

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