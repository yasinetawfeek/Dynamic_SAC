import gym
import numpy as np
import zmq
import pickle
import time
import threading
import pyautogui
import random
from gym import spaces
from boundary_detect_v2 import TrackBoundary
from ai2vcu_requests import AI2VCU_REQUESTS
from config import *

context = zmq.Context()

socket_commands = context.socket(zmq.PUB)
socket_commands.bind("tcp://*:5554")
socket_commands.setsockopt(zmq.SNDHWM, 1)
socket_commands.setsockopt(zmq.IMMEDIATE, 1)

socket_observation = context.socket(zmq.SUB)
socket_observation.connect("tcp://localhost:5555")
socket_observation.setsockopt_string(zmq.SUBSCRIBE, "")
socket_observation.setsockopt(zmq.RCVTIMEO, 2)

socket_map = context.socket(zmq.SUB)
socket_map.connect("tcp://localhost:5556")
socket_map.setsockopt_string(zmq.SUBSCRIBE, "")
socket_map.setsockopt(zmq.RCVTIMEO, 2)

socket_vehicle_condition = context.socket(zmq.SUB)
socket_vehicle_condition.connect("tcp://localhost:5557")
socket_vehicle_condition.setsockopt_string(zmq.SUBSCRIBE, "")
socket_vehicle_condition.setsockopt(zmq.RCVTIMEO, 2)

socket_wheel_speed = context.socket(zmq.SUB)
socket_wheel_speed.connect("tcp://localhost:5558")
socket_wheel_speed.setsockopt_string(zmq.SUBSCRIBE, "")
socket_wheel_speed.setsockopt(zmq.RCVTIMEO, 2)

socket_what_I_see = context.socket(zmq.PUB)
socket_what_I_see.bind("tcp://*:5559")
socket_what_I_see.setsockopt(zmq.SNDHWM, 1)
socket_what_I_see.setsockopt(zmq.IMMEDIATE, 1)

class CmEnv(gym.Env):
    def __init__(self, build_flag=True):
        gym.Env.__init__(self)

        # Define action and observation space
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        low = np.zeros((MAX_DETECTIONS,3))
        high = np.inf * np.ones((MAX_DETECTIONS,3))
        
        self.observation_space = spaces.Dict({
            'blue_cones': spaces.Box(low=low, high=high, shape=(MAX_DETECTIONS, 3), dtype=np.float32),
            'yellow_cones': spaces.Box(low=low, high=high, shape=(MAX_DETECTIONS, 3), dtype=np.float32),
            'velocity': spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            'previous_steer': spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        })

        # Internal state
        self.current_observation = np.zeros(3, dtype=np.float32)
        self.done = False
        self.recieved_map = False

        self.track = None
        self.current_track_state = None
        self.blue_cones_detected = np.zeros((MAX_DETECTIONS, 3))
        self.yellow_cones_detected = np.zeros((MAX_DETECTIONS,3))

        self.rpm_request = 0
        self.brake_request = 0
        self.steer_request = 0

        self.start_offset = 0
        self.total_reward = 0
        self.distance_from_centre = 0
        self.previous_steer_command = 0.0

        self.target_progress = 0.0
        self.initial_timestep = 0
        self.timestep_counter = 0

        self.wheel_speed = 0.0
        self.velocity_ms = 0.0

        self.initial_episodes_in_start_position_counter = EPISODES_AT_START_POSITION

        self.current_vehicle_position = (VEHICLE_START_POSITION_X, VEHICLE_START_POSITION_Y)
        self.time_of_last_progress = time.time()

        # self.vehicle_commands = AI2VCU_REQUESTS()
        self.vehicle_commands = {
            "steer": 0.0,
            "rpm": 0,
            "brake": 0
        }

        self.episode_timer = None
        self.initial_step_counter = INITIAL_STEP_COUNTER
        self.terminate = False

        self.map = None

        self.get_vehicle_position()
        self.get_vehicle_observation()
        self.get_wheel_speeds()
        self.send_vehicle_commands()
        
    def get_vehicle_position(self):
        # print("got vehicle position")
        try:
            self.current_vehicle_position = pickle.loads(socket_vehicle_condition.recv())
        except zmq.error.Again:
            pass
        threading.Timer(0.001, self.get_vehicle_position).start()

    # def get_vehicle_observation(self):
    #     message = socket_observation.recv()
    #     (self.blue_cones_detected, self.yellow_cones_detected) = pickle.loads(message)
    #     # print(self.blue_cones_detected)
    #     # print("#"*20)
    #     # print(self.yellow_cones_detected)

    #     threading.Timer(0.01, self.get_vehicle_observation).start()

    def sort_coordinates_by_proximity(self, coord_list, target_point):
        x0, y0 = target_point
        return sorted(coord_list, key=lambda p: (p[0] - x0)**2 + (p[1] - y0)**2)

    def get_vehicle_observation(self):
        try:
            message = socket_observation.recv()
            self.blue_cones_detected = np.zeros((MAX_DETECTIONS, 3))
            self.yellow_cones_detected = np.zeros((MAX_DETECTIONS, 3))
            (blue_cones, yellow_cones) = pickle.loads(message)

            blue_cones_ordered = self.sort_coordinates_by_proximity(blue_cones, (0,0))[:MAX_DETECTIONS]
            yellow_cones_ordered = self.sort_coordinates_by_proximity(yellow_cones, (0,0))[:MAX_DETECTIONS]

            if len(blue_cones_ordered) != 0 and len(yellow_cones_ordered) != 0:
                self.blue_cones_detected[:len(blue_cones_ordered)] = blue_cones_ordered
                self.yellow_cones_detected[:len(yellow_cones_ordered)] = yellow_cones_ordered
        except zmq.error.Again:
            pass

        socket_what_I_see.send(pickle.dumps((self.blue_cones_detected, self.yellow_cones_detected)))

        threading.Timer(0.001, self.get_vehicle_observation).start()

    def get_wheel_speeds(self):
        try:
            self.wheel_speed = pickle.loads(socket_wheel_speed.recv())
        except zmq.error.Again:
            pass
        self.velocity_ms = self.wheel_speeds_to_mps(self.wheel_speed)

        threading.Timer(0.001, self.get_wheel_speeds).start()

    def send_vehicle_commands(self):
        if not self.terminate:
            pass
            # print("sending")       
        socket_commands.send(pickle.dumps(self.vehicle_commands))
        
        threading.Timer(0.001, self.send_vehicle_commands).start()

    def wheel_speeds_to_mps(self, wheelspeed):
        s = 2 * np.pi * (WHEEL_DIAMETER / 2) * (wheelspeed/60)
        return s

    def _get_obs(self):
        return{
            'blue_cones': self.blue_cones_detected,
            'yellow_cones': self.yellow_cones_detected,
            'velocity': self.velocity_ms,
            'previous_steer': self.previous_steer_command
        }
    
    def _get_info(self):
        return {"hey": 4}
    
    # Reward function 0
    def exponential_progress_reward_function(self, progress_made, current_progress, left_track, collision):
        if progress_made < MINIMUM_PROGRESS:
            reward = REWARD_WHEN_BELOW_MINIMUM_PROGRESS
        else:
            reward = (progress_made * REWARD_SCALE_FACTOR) ** EXPONENTIAL_INCREASE_OF_REWARD

        if reward > MAX_REWARD:
            reward = REWARD_WHEN_BELOW_MINIMUM_PROGRESS

        # reward = (progress_made * REWARD_SCALE_FACTOR) - (0.00001 * REWARD_SCALE_FACTOR)

        if left_track and self.max_progress > 0.001:
            reward = -abs(NEGATIVE_REWARD)
        else:
            left_track = False

        if collision['collision']:
            reward = -abs(NEGATIVE_REWARD)

        if current_progress > PROGRESS_FOR_COMPLETION and (progress_made != 0 and progress_made < 0.05):
            reward = abs(REWARD_WHEN_COMPLETING_TRACK)

        return reward

    # Reward function 1
    def minimal_steering_with_max_velocity_reward_function(self, steer_request, previous_steer_request, terminate, velocity):
        reward = 0
        if not terminate:
            steer_diff = (abs(steer_request - previous_steer_request))

            if steer_diff <= 0.001:
                reward += (FUNCTION_1_A1 + FUNCTION_1_A2) * velocity
            else:
                reward += (FUNCTION_1_A1 + (FUNCTION_1_A2 / steer_diff)) * velocity

        else:
            reward -= (FUNCTION_1_A3 * (velocity ** FUNCTION_1_A4)) + FUNCTION_1_A5

        return reward

    # Reward function 2
    def velocity_based_reward_with_minimal_steering(self, terminate, steer_request, previous_steer_request, velocity):
        reward = 0
        if not terminate:
            reward += FUNCTION_2_A1 * (velocity ** FUNCTION_2_A2)
            reward -= FUNCTION_2_A3 * (abs(steer_request - previous_steer_request) ** FUNCTION_2_A4)
        else:
            reward -= FUNCTION_2_A5 * (velocity ** FUNCTION_2_A6)

        return reward

    # Reward function 3
    def velocity_based_reward_with_distance_to_centre(self, terminate, velocity, distance_to_cl):
        reward = 0
        reward -= FUNCTION_3_A3 * (distance_to_cl ** FUNCTION_3_A4)

        if not terminate:
            reward += FUNCTION_3_A1 * (velocity ** FUNCTION_3_A2)
        
        return reward
    
    # Reward function 4
    def target_distance_based_reward(self, terminate, position, target):
        reward = 0 

        if not terminate:
            reward += (FUNCTION_4_A1) / np.sqrt((abs(position[0] - target[0]) ** 2) + (abs(position[1] - target[1]) ** 2))
    
        return reward
    
    # Reward function 5
    def checkpoint_baised_reward(self, terminate, reached, initial_timestep, current_timestep, previous_steer, current_steer):
        reward = 0
        if not terminate:
            steer_diff = abs(current_steer - previous_steer)
            reward += FUNCTION_5_CONST_REWARD
            if reached:
                reward += FUNCTION_5_A1
                reward += FUNCTION_5_A2 / ((current_timestep - initial_timestep) ** FUNCTION_5_A3)

            if steer_diff > STEER_MINIMUM:
                reward_to_add = FUNCTION_5_A4 / (( steer_diff ** FUNCTION_5_A5))
                
                if reward_to_add > FUNCTION_5_A4:
                    reward_to_add = FUNCTION_5_A4

                reward += reward_to_add

            else:
                reward += FUNCTION_5_A4

        return reward

    def step(self, action):

        start_time = time.time()

        # print("hi")
        if self.track == None:
            if self.map == None:
                self.map = pickle.loads(socket_map.recv())
                self.map[0].append(self.map[0][0])
                self.map[1].append(self.map[1][0])

            self.track = TrackBoundary(self.map[0], self.map[1])

        if action[0] >= 0:
            self.rpm_request = action[0] * MAX_RPM
            self.brake_request = 0
        else:
            self.brake_request = abs(action[0] * MAX_BRAKE_PERCENTAGE)
            self.rpm_request = 0
        
        self.steer_request = action[1] * MAX_STEERING_ANGLE

        # self.vehicle_commands = {
        #     "steer": self.steer_request,
        #     "rpm": self.rpm_request,
        #     "brake": self.brake_request
        # }

        self.vehicle_commands = {
            "steer": self.steer_request,
            "rpm": 400,
            "brake": 0
        }


        self.current_track_state = self.track.update_positon(self.current_vehicle_position)
        self.distance_from_centre = self.track.get_distance_from_centre_line(self.current_vehicle_position)
        self.terminate = self.current_track_state['is_outside_track']

        self.terminate = self.current_track_state['is_outside_track'] or self.current_track_state['collision']['collision']
        progress_made = self.current_track_state['progress_made']
        current_progress = self.current_track_state['current_progress']
        is_outside_track = self.current_track_state['is_outside_track']
        collision_info = self.current_track_state['collision']
        collision = collision_info['collision']

        if self.target_progress == 0.0:
            if self.current_vehicle_position == (VEHICLE_START_POSITION_X, VEHICLE_START_POSITION_Y):
                self.target_progress = 0.0
            else:
                self.target_progress = current_progress + PROGRESS_BETWEEN_CHECKPOINTS
                self.initial_timestep = self.timestep_counter
        reward = 0

        if REWARD_FUNCTION_TO_USE == 0:
            reward = self.exponential_progress_reward_function(progress_made, current_progress, is_outside_track, self.terminate)

        elif REWARD_FUNCTION_TO_USE == 1:
            reward = self.minimal_steering_with_max_velocity_reward_function(self.steer_request, self.previous_steer_command, self.terminate, self.velocity_ms)
        
        elif REWARD_FUNCTION_TO_USE == 2:
            reward = self.velocity_based_reward_with_minimal_steering(self.terminate, self.steer_request, self.previous_steer_command, self.velocity_ms)

        elif REWARD_FUNCTION_TO_USE == 3:
            reward = self.velocity_based_reward_with_distance_to_centre(self.terminate, self.velocity_ms, self.distance_from_centre)

        elif REWARD_FUNCTION_TO_USE == 4:
            target = (0,0)
            if FUNCTION_4_RANDOM_POINT:
                target = self.track.random_point_in_quadrilateral(self.blue_cones_detected[-1], self.blue_cones_detected[-2], self.yellow_cones_detected[-1], self.yellow_cones_detected[-2])
            else:
                target = self.track.find_middle_point(self.blue_cones_detected[-1], self.blue_cones_detected[-2], self.yellow_cones_detected[-1], self.yellow_cones_detected[-2])
            
            reward = self.target_distance_based_reward(self.terminate, self.current_vehicle_position, target)

        elif REWARD_FUNCTION_TO_USE == 5:
            reached = False
            if abs(current_progress - self.target_progress) <= CHECKPOINT_REACHED_THRESHOLD and self.target_progress != 0.0:
                reached = True
                
            reward = self.checkpoint_baised_reward(self.terminate, reached,  self.initial_timestep, self.timestep_counter, self.previous_steer_command, self.steer_request)
            
            if reached:
                self.target_progress = current_progress + PROGRESS_BETWEEN_CHECKPOINTS
                self.initial_timestep = self.timestep_counter
            
        if REWARD_FUNCTION_TO_USE == 5:
            if reward > FUNCTION_5_A4:
                self.time_of_last_progress = time.time()

        elif reward > 0:
            self.time_of_last_progress = time.time()

        timeout = False
        if (time.time() - self.time_of_last_progress > PROGRESS_TIMOUT) and (PROGRESS_TIMOUT != 0):
            self.terminate = True
            # reward = -abs(TIMOUT_NEGATIVE_REWARD)
            timeout = True

        episode_timeout = False
        if (time.time() - self.episode_timer > EPISODE_TIME_LIMIT) and EPISODE_TIME_LIMIT != 0:
            self.terminate = True
            # reward = -abs(TIMOUT_NEGATIVE_REWARD)
            episode_timeout = True

        if (current_progress > 0.95 and self.initial_step_counter <= 0):
            reward = REWARD_WHEN_COMPLETING_TRACK
            self.terminate = True

        self.total_reward += reward

        observation = self._get_obs()
        info = self._get_info()

        self.previous_steer_command = self.steer_request

        self.timestep_counter += 1

        if self.initial_step_counter != 0:
            reward = 0
            self.initial_step_counter -= 1

        print("\033c", end="")
        print(f"vehicle position:       {self.current_vehicle_position}")
        print(f"timestep:               {self.timestep_counter}")
        print(f"velocity:               {self.velocity_ms} m/s")
        print(f"start offset:           {self.start_offset}")
        print(f"initial episodes:       {self.initial_episodes_in_start_position_counter}")
        print(f"current progress:       {current_progress}")
        print(f"target progress:        {self.target_progress}")
        print(f"progress made:          {progress_made}")
        print(f"given reward:           {reward}")
        print(f"total reward:           {self.total_reward}")
        print(f"is outside track:       {is_outside_track}")
        print(f"collision:              {collision_info}")
        print(f"timeout:                {timeout}")
        print(f"episode timeout:        {episode_timeout}")
        print(f"terminate:              {self.terminate}")
        print(f"step time:              {time.time() - start_time}")


        return observation, reward, self.terminate, False, info
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # self.track = None

        with open(PATH_TO_TESTRUN_FILE, 'r') as file:
            data = file.readlines()
        
        if self.initial_episodes_in_start_position_counter > 0:
            self.start_offset = VEHICLE_START_POSITION_X
            self.initial_episodes_in_start_position_counter -= 1
        else:
           self.start_offset = float(random.randint(MIN_SPAWN, MAX_SPAWN))

        data[34] = f'Vehicle.StartPos = {self.start_offset} 0.0\n'

        with open(PATH_TO_TESTRUN_FILE, 'w') as file:
            file.writelines(data)

        pyautogui.click(1878, 1017)
        time.sleep(SLEEP_AFTER_STOP)
        pyautogui.click(1876, 969)
        time.sleep(SLEEP_AFTER_START)

        self.track = None

        self.rpm_request = 0
        self.brake_request = 0
        self.steer_request = 0

        self.vehicle_commands = {
            "steer": self.steer_request,
            "rpm": self.rpm_request,
            "brake": self.brake_request
        }

        # print("sending")        
        # socket_commands.send(pickle.dumps(self.vehicle_commands))

        print("reset")

        observation = self._get_obs()
        info = self._get_info()
        
        self.current_observation = np.zeros(3, dtype=np.float32)
        self.done = False
        self.recieved_map = False

        # self.track = None
        self.current_track_state = None
        self.blue_cones_detected = np.zeros((MAX_DETECTIONS, 3))
        self.yellow_cones_detected = np.zeros((MAX_DETECTIONS,3))

        self.rpm_request = 0
        self.brake_request = 0
        self.steer_request = 0
        self.total_reward = 0

        self.target_progress = 0.0
        self.initial_timestep = 0
        self.timestep_counter = 0
        self.start_offset = 0
        self.distance_from_centre = 0
        self.previous_steer_command = 0.0

        self.wheel_speed = 0.0
        self.velocity_ms = 0.0 

        self.initial_step_counter = INITIAL_STEP_COUNTER

        self.current_vehicle_position = (VEHICLE_START_POSITION_X, VEHICLE_START_POSITION_Y)

        # self.vehicle_commands = AI2VCU_REQUESTS()

        self.vehicle_commands = {
            "steer": 0.0,
            "rpm": 0,
            "brake": 0
        }

        self.time_of_last_progress = time.time()
        self.episode_timer = time.time()

        # print("reached")
        return observation, info


        # ADD LINE TO MOVE VEHICLE POSITION