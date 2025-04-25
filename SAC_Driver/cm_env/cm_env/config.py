REWARD_SCALE_FACTOR = 10000                     # Maximum progress is 1.0
NEGATIVE_REWARD = -3000                         # Given when episode ends
TIMEOUT_NEGATIVE_REWARD = -500                   # Given when episode timesout
MAX_REWARD = 700                                # Measured at each timestep
MAX_DETECTIONS = 6                              # Maximum amount of detections for each type of cone (CHANGE IN ENVIRONMENT.py)
COLLISION_BUFFER = 0.00                         # Buffer for collision detection
MINIMUM_PROGRESS = 0.00002                      # Progress at which reward is calculated
EXPONENTIAL_INCREASE_OF_REWARD = 1.30           # Exponential factor when reward is calculated
REWARD_WHEN_BELOW_MINIMUM_PROGRESS = -0.05      # Reward given when progress is below minimum
REWARD_WHEN_COMPLETING_TRACK = 10000            # Reward at completion of objective
PROGRESS_TIMEOUT = 0                           # Timout before episode termination
EPISODE_TIME_LIMIT = 1200                       # Maximum time of episode before termination
MIN_SPAWN = 10                                  # Minimum distance along the track in which the vehicle can spawn
MAX_SPAWN = 800                               # Maximum distance along the track in which the vehicle can spawn
# MAX_SPAWN = 350                                 # Maximum distance along the track in which the vehicle can spawn
VEHICLE_START_POSITION_X = -9                   # Position at which the vehicle's first position will be (NOT THE SPAWN LOCATION)
# VEHICLE_START_POSITION_X = 13922                   # Position at which the vehicle's first position will be (NOT THE SPAWN LOCATION)
# VEHICLE_START_POSITION_Y = 2385                    # Position at which the vehicle's first position will be (NOT THE SPAWN LOCATION)

VEHICLE_START_POSITION_Y = 8                    # Position at which the vehicle's first position will be (NOT THE SPAWN LOCATION)
# VEHICLE_START_POSITION_Y = -1.5                  # Position at which the vehicle's first position will be (NOT THE SPAWN LOCATION)
EPISODES_AT_START_POSITION = 0 
PROGRESS_FOR_COMPLETION = 0.999                   # Progress at which the goal is achieved
INITIAL_STEP_COUNTER = 5                        # Steps until initial reward is given

MAX_STEERING_ANGLE_CHANGE_PER_STEP = 1.5


MAX_RPM = 400                                   # RPM limit for vehicle control
MAX_BRAKE_PERCENTAGE = 100                      # Brake percentage limit for vehicle control
MAX_STEERING_ANGLE = 21.0                       # Steering limit for vehicle control
TORQUE_REQUEST = 190                            # Torque Request
ESTOP_REQUEST = 0                               # Emergency stop request
MISSION_STATUS = 2                              # Mission status of the vehicle
DIRECTION_REQUEST = 1                           # Gearbox request

SLEEP_AFTER_STOP = 2                            # Wait time after stopping episode
SLEEP_AFTER_START = 1                           # Wait time after starting episode
SLEEP_AFTER_POSITION_TIMOUT = 1                # Timout when position timout

SAME_POSITION_TIMEOUT = 80                      # Timout for unchanged vehicle position

WHEEL_DIAMETER = 0.505                          # Diameter of the wheels

POSITION_RANDOMISATION = False                  # Agent starts at random positions in each episode 
# POSITION_RANDOMISATION = True                  # Agent starts at random positions in each episode 
OBSERVATION_INTREVAL = 0.01                     # Time between action and observation
ACTION_INTREVAL = 0.04                          # Time between observation and action (time between end of step func and start of step func)

PATH_TO_TESTRUN_FILE = "/home/yasinetawfeek/Desktop/uwe-ipg-sim/Data/TestRun/small_silverstone"
# PATH_TO_TESTRUN_FILE = "/home/yasinetawfeek/Desktop/uwe-ipg-sim/Data/TestRun/rand"
# PATH_TO_TESTRUN_FILE = "/home/yasinetawfeek/Desktop/uwe-ipg-sim/Data/TestRun/FS_autonomous_TrackDrive"


REWARD_FUNCTION_TO_USE = 5                      # Which reward function to use

FUNCTION_1_A1 = 0.01                            # Constant reward givan at each timestep inside the track
FUNCTION_1_A2 = 0.1                             # Significance of minimal steering reward
FUNCTION_1_A3 = 0                               # Significance of negative reward at exit
FUNCTION_1_A4 = 2                               # Exponential factor applied to velocity to calculate penalty given
FUNCTION_1_A5 = 0                               # Negative reward given at exit
STEER_MINIMUM = 0.001                           # Minimum before difference is considered 0

FUNCTION_2_A1 = 0.5                             # Significance of velocity based reward 
FUNCTION_2_A2 = 1.3                             # Exponant applied to velocity before calculating reward
FUNCTION_2_A3 = 0.001                           # Significance of steering based penalty
FUNCTION_2_A4 = 1.2                             # Exponant applied to steering before calculating penalty
FUNCTION_2_A5 = 0                               # significance of collision-velocity based penalty
FUNCTION_2_A6 = 2                               # Exponential factor applied to velocity to calculate penalty given

FUNCTION_3_A1 = 2                               # Significance of velocity based reward
FUNCTION_3_A2 = 2                               # Exponant applied to velocity before calculating reward
FUNCTION_3_A3 = 0.0001                          # Significance of centre-line distance penalty
FUNCTION_3_A4 = 1                               # Exponant applied to cl distance before calculating reward

FUNCTION_4_A1 = 10                              # Significance of the offset based reward
FUNCTION_4_RANDOM_POINT = False                 # bool for picking random point in area or centre-point

FUNCTION_5_A1 = 1.0                             # Reward for reaching checkpoint
FUNCTION_5_A2 = 200                            # Significance of timestep based reward
FUNCTION_5_A3 = 1.2                             # Exponant applied to time difference
# FUNCTION_5_A4 = 0.002                           # Significance of steer change reward
FUNCTION_5_A4 = 0                               # Significance of steer change reward
FUNCTION_5_A5 = 1.0                             # Exponant applied to steer change before calculating reward
FUNCTION_5_PENALTY_FOR_EXIT = 1000              # Penalty given when exit or collision
FUNCTION_5_CONST_REWARD = -0.005                 # Constant reward
# PROGRESS_BETWEEN_CHECKPOINTS = 0.004            # Distance between checkpoints
# CHECKPOINT_REACHED_THRESHOLD = 0.0001            # Distance threshold at which a target is reached

FUNCTION_FINAL_A1 = 1                           # Reward given when reached checkpoint
FUNCTION_FINAL_A2 = 0                         # Penalty for collision or exit
FUNCTION_FINAL_A3 = 0                           # Reward for reaching end
# DISCOUNT_FACTOR = 1                         # Discount factor for future rewards
DISCOUNT_FACTOR = 0.999                         # Discount factor for future rewards
# DISCOUNT_FACTOR = 0.99                         # Discount factor for future rewards
# PROGRESS_BETWEEN_CHECKPOINTS = 0.0025
PROGRESS_BETWEEN_CHECKPOINTS = 0.001


TIMESTEPS_PER_MODEL_SAVE = 50000                # Training timesteps until model is saved
DISTANCE_FROM_CL_TO_SIDE = 4
BRAKE_TIMEOUT_COUNTER = 120 
# TIMESTEPS_PER_MODEL_SAVE = 100