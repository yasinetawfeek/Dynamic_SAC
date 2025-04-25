import numpy as np
import zmq
import pickle
import threading
from boundary_detect_v2 import TrackBoundary
from config import *
from track_maps.silverstone import *
from stable_baselines3 import SAC
from config import *
import time
from gym_env_new import CmEnv
import pandas as pd

# context = zmq.Context()

# socket_commands = context.socket(zmq.PUB)
# socket_commands.bind("tcp://*:5552")
# socket_commands.setsockopt(zmq.SNDHWM, 1)
# socket_commands.setsockopt(zmq.IMMEDIATE, 1)

# socket_observation_deploy = context.socket(zmq.SUB)
# socket_observation_deploy.connect("tcp://localhost:5555")
# socket_observation_deploy.setsockopt_string(zmq.SUBSCRIBE, "")
# socket_observation_deploy.setsockopt(zmq.RCVTIMEO, 2)

# socket_vehicle_condition = context.socket(zmq.SUB)
# socket_vehicle_condition.connect("tcp://localhost:5557")
# socket_vehicle_condition.setsockopt_string(zmq.SUBSCRIBE, "")
# socket_vehicle_condition.setsockopt(zmq.RCVTIMEO, 2)

# socket_wheel_speed_deploy = context.socket(zmq.SUB)
# socket_wheel_speed_deploy.connect("tcp://localhost:5558")
# socket_wheel_speed_deploy.setsockopt_string(zmq.SUBSCRIBE, "")
# socket_wheel_speed_deploy.setsockopt(zmq.RCVTIMEO, 2)

# socket_what_I_see = context.socket(zmq.PUB)
# socket_what_I_see.bind("tcp://*:5553")
# socket_what_I_see.setsockopt(zmq.SNDHWM, 1)
# socket_what_I_see.setsockopt(zmq.IMMEDIATE, 1)

# def sort_coordinates_by_proximity(coord_list, target_point):
#         x0, y0 = target_point
#         return sorted(coord_list, key=lambda p: (p[0] - x0)**2 + (p[1] - y0)**2)

# def wheel_speeds_to_mps(wheelspeed):
#     s = 2 * np.pi * (WHEEL_DIAMETER / 2) * (wheelspeed/60)
#     return s

# def get_speed():
#     wheel_speed = pickle.loads(socket_wheel_speed_deploy.recv(zmq.DONTWAIT))
#     velocity_ms = wheel_speeds_to_mps(wheel_speed)
#     return velocity_ms

class DeployHandler():
    def __init__(self):
        self.blue_cones_detected = np.zeros((MAX_DETECTIONS, 2))
        self.yellow_cones_detected = np.zeros((MAX_DETECTIONS, 2))
        self.velocity_ms = 0
        self.previous_steer_command = 0.0
        self.vehicle_commands = {
            "steer": 0.0,
            "rpm": 0,
            "brake": 0
        }

        self.get_vehicle_observation()
        self.get_wheel_speeds()
        self.send_vehicle_commands()

    def send_vehicle_commands(self):
        socket_commands.send(pickle.dumps(self.vehicle_commands))
        threading.Timer(0.001, self.send_vehicle_commands).start()

    def get_vehicle_observation(self):
        try:
            message = socket_observation_deploy.recv()
            (blue_cones, yellow_cones) = pickle.loads(message)

            self.blue_cones_detected = np.zeros((MAX_DETECTIONS, 2))
            self.yellow_cones_detected = np.zeros((MAX_DETECTIONS, 2))

            blue_cones_ordered = sort_coordinates_by_proximity(blue_cones, (0,0))[:MAX_DETECTIONS]
            yellow_cones_ordered = sort_coordinates_by_proximity(yellow_cones, (0,0))[:MAX_DETECTIONS]

            if len(blue_cones_ordered) != 0:
                self.blue_cones_detected[:len(blue_cones_ordered)] = blue_cones_ordered

            if len(yellow_cones_ordered) != 0:
                self.yellow_cones_detected[:len(yellow_cones_ordered)] = yellow_cones_ordered

            socket_what_I_see.send(pickle.dumps((self.blue_cones_detected, self.yellow_cones_detected)))

        except zmq.Again:
            pass

        threading.Timer(0.001, self.get_vehicle_observation).start()

    def get_wheel_speeds(self):
        try:
            wheel_speed = pickle.loads(socket_wheel_speed_deploy.recv(zmq.DONTWAIT))
            self.velocity_ms = wheel_speeds_to_mps(wheel_speed)
        except zmq.Again:
            pass

        threading.Timer(0.001, self.get_wheel_speeds).start()

# def get_vehicle_observation():
#     message = socket_observation_deploy.recv()
#     (blue_cones, yellow_cones) = pickle.loads(message)

#     blue_cones_detected = np.zeros((MAX_DETECTIONS, 2))
#     yellow_cones_detected = np.zeros((MAX_DETECTIONS, 2))

#     blue_cones_ordered = sort_coordinates_by_proximity(blue_cones, (0,0))[:MAX_DETECTIONS]
#     yellow_cones_ordered = sort_coordinates_by_proximity(yellow_cones, (0,0))[:MAX_DETECTIONS]

#     if len(blue_cones_ordered) != 0:
#         blue_cones_detected[:len(blue_cones_ordered)] = blue_cones_ordered

#     if len(yellow_cones_ordered) != 0:
#         yellow_cones_detected[:len(yellow_cones_ordered)] = yellow_cones_ordered

#     socket_what_I_see.send(pickle.dumps((blue_cones_detected, yellow_cones_detected)))
#     return (blue_cones_detected, yellow_cones_detected)

# handler = DeployHandler()



brake_request = 0
rpm_request = 0
path = "/media/yasinetawfeek/069F-E29D/SAC_ipg"
previous_steer_command = 0

# while True:
#     start_time = time.time()
#     observation = {
#         'blue_cones': np.array(handler.blue_cones_detected, dtype=np.float32),
#         'yellow_cones': np.array(handler.yellow_cones_detected, dtype=np.float32),
#         'velocity': np.array([handler.velocity_ms], dtype=np.float32),
#         'previous_steer': np.array([previous_steer_command / MAX_STEERING_ANGLE], dtype=np.float32),
#     }

#     action, _states = model.predict(observation, deterministic=True)

#     if action[0] >= 0:
#         rpm_request = action[0] * MAX_RPM
#         brake_request = 0
#     else:
#         brake_request = abs(action[0] * MAX_BRAKE_PERCENTAGE)
#         rpm_request = 0

#     proposed_steer_command = action[1] * MAX_STEERING_ANGLE


#     if abs(previous_steer_command - proposed_steer_command) > MAX_STEERING_ANGLE_CHANGE_PER_STEP * 2:
#         if proposed_steer_command > previous_steer_command:
#             steer_request = (previous_steer_command + MAX_STEERING_ANGLE_CHANGE_PER_STEP * 2)
#         else:
#             steer_request = (previous_steer_command - MAX_STEERING_ANGLE_CHANGE_PER_STEP * 2) 
#     else:
#         steer_request = proposed_steer_command

#     handler.vehicle_commands = {
#         "steer": steer_request,
#         "rpm": rpm_request,
#         "brake": brake_request
#     }
#     previous_steer_command = steer_request

#     time.sleep(0.04)
#     print("\033c", end="")
#     print(f"vehicle position:           ")
#     print(f"timestep:                   ")
#     # print(f"distance from cl:           {self.distance_from_centre}")
#     print(f"previous steer:             {previous_steer_command / MAX_STEERING_ANGLE}")
#     print(f"velocity:                   {handler.velocity_ms} m/s")
#     print(f"start offset:               ")
#     print(f"initial progress:           ")
#     print(f"initial episodes:           ")
#     print(f"current progress:           ")
#     print(f"target progress:            ")
#     print(f"progress made:              ")
#     print(f"given reward:               ")
#     print(f"total reward:               ")
#     print(f"is outside track:           ")
#     print(f"collision:                  ")
#     print(f"timeout:                    ")
#     print(f"episode timeout:            ")
#     print(f"terminate:                  ")
#     print(f"weird crash counter:        ")
#     print(f"step time:                  {time.time() - start_time}")
#     print(observation)

#     # send_vehicle_commands()

def flatten_obs(obs_dict):
    return np.concatenate([
        np.array(obs_dict[k], dtype=np.float32).flatten() for k in sorted(obs_dict.keys())
    ])

env = CmEnv(build_flag=False)
model = SAC.load("/media/yasinetawfeek/069F-E29D/SAC_ipg/models/Final_Function/DissertationDriver/800000.zip", env=env)
obs, info = env.reset()
done = False
while True:
    while not done:
        obs = env.get_obs_deployment()
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)

    checkpoint_df = pd.DataFrame(env.checkpoint_history_bank)
    control_df = pd.DataFrame(env.control_history_bank)
    checkpoint_df.to_csv(path + f"/models/Final_Function/DissertationDriver/checkpoint_bank_deployment")
    control_df.to_csv(path + f"/models/Final_Function/DissertationDriver/control_bank_deployment")

    obs, info = env.reset()
    done = False

# while True:
#     model.learn(total_timesteps=10000, reset_num_timesteps=False)
    # df = pd.DataFrame(env.checkpoint_history_bank)
    # df.to_csv(path + f"/models/Final_Function/DissertationDriver/checkpoint_bank_deployment")
    
