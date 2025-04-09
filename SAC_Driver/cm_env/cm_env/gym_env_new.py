import gymnasium
import numpy as np
import zmq
import pickle
import time
import threading
import pyautogui
import random
from gymnasium import spaces
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
# socket_map.setsockopt(zmq.RCVTIMEO, 2)

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

class CmEnv(gymnasium.Env):
    def __init__(self, build_flag=True):
        gymnasium.Env.__init__(self)

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

        self.crash_counter = 0
        self.checkpoint_counter = 0

        self.initial_episodes_in_start_position_counter = EPISODES_AT_START_POSITION

        self.current_vehicle_position = (VEHICLE_START_POSITION_X, VEHICLE_START_POSITION_Y)
        self.previous_vehicle_position = self.current_vehicle_position
        self.time_of_last_progress = time.time()

        self.vehicle_commands = {
                "steer": 0.0,
                "rpm": 0,
                "brake": 0
            } 
        
        self.episode_timer = None
        self.initial_step_counter = INITIAL_STEP_COUNTER
        self.terminate = False
        self.finished = False
        self.send_data = True

        self.same_position_timout_counter = SAME_POSITION_TIMEOUT
        self.position_timout_flag = False
        self.recieved_observation = False

        self.get_vehicle_position()
        self.get_vehicle_observation()
        self.get_wheel_speeds()
        self.send_vehicle_commands()

        
    def get_vehicle_position(self):
        # print("got vehicle position")
        try:
            self.current_vehicle_position = pickle.loads(socket_vehicle_condition.recv(zmq.DONTWAIT))
        except zmq.Again:
            pass
        threading.Timer(0.001, self.get_vehicle_position).start()

    def send_vehicle_commands(self):
        if not self.terminate and self.send_data:
            # print("sending")       
            socket_commands.send(pickle.dumps(self.vehicle_commands))
        threading.Timer(0.001, self.send_vehicle_commands).start()

    def sort_coordinates_by_proximity(self, coord_list, target_point):
        x0, y0 = target_point
        return sorted(coord_list, key=lambda p: (p[0] - x0)**2 + (p[1] - y0)**2)

    def get_vehicle_observation(self):
        self.blue_cones_detected = np.zeros((MAX_DETECTIONS, 3))
        self.yellow_cones_detected = np.zeros((MAX_DETECTIONS, 3))
        try:
            message = socket_observation.recv(zmq.DONTWAIT)
            (blue_cones, yellow_cones) = pickle.loads(message)

            blue_cones_ordered = self.sort_coordinates_by_proximity(blue_cones, (0,0))[:MAX_DETECTIONS]
            yellow_cones_ordered = self.sort_coordinates_by_proximity(yellow_cones, (0,0))[:MAX_DETECTIONS]

            if len(blue_cones_ordered) != 0:
                self.blue_cones_detected[:len(blue_cones_ordered)] = blue_cones_ordered

            if len(yellow_cones_ordered) != 0:
                self.yellow_cones_detected[:len(yellow_cones_ordered)] = yellow_cones_ordered
            if self.send_data:
                socket_what_I_see.send(pickle.dumps((self.blue_cones_detected, self.yellow_cones_detected)))

            self.recieved_observation = True
        except zmq.Again:
            pass

        threading.Timer(0.001, self.get_vehicle_observation).start()

    def get_wheel_speeds(self):
        try:
            self.wheel_speed = pickle.loads(socket_wheel_speed.recv(zmq.DONTWAIT))
            self.velocity_ms = self.wheel_speeds_to_mps(self.wheel_speed)
        except zmq.Again:
            pass

        threading.Timer(0.001, self.get_wheel_speeds).start()

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

    def calculate_reward(self, reached, terminated, finished):
        if terminated:
            return FUNCTION_FINAL_A2
        if finished:
            return FUNCTION_FINAL_A3 / self.timestep_counter
        if reached:
            return FUNCTION_FINAL_A1 * (DISCOUNT_FACTOR ** self.checkpoint_counter)
        return 0

    def step(self, action):

        start_time = time.time()

        if self.current_vehicle_position == self.previous_vehicle_position:
            self.same_position_timout_counter -= 1
        else:
            self.same_position_timout_counter = SAME_POSITION_TIMEOUT

        self.previous_vehicle_position = self.current_vehicle_position

        # print("hi")
        if self.track == None:
            message = pickle.loads(socket_map.recv())
            message[0].append(message[0][0])
            message[1].append(message[1][0])
            self.track = TrackBoundary(message[0], message[1])

        if action[0] >= 0:
            self.rpm_request = action[0] * MAX_RPM
            self.brake_request = 0
        else:
            self.brake_request = abs(action[0] * MAX_BRAKE_PERCENTAGE)
            self.rpm_request = 0
        
        self.steer_request = action[1] * MAX_STEERING_ANGLE

        self.vehicle_commands = {
            "steer": self.steer_request,
            "rpm": self.rpm_request,
            "brake": self.brake_request
        }

        # self.vehicle_commands = {
        #     "steer": self.steer_request,
        #     "rpm": 400,
        #     "brake": 0
        # }


        self.current_track_state = self.track.update_positon(self.current_vehicle_position)
        self.distance_from_centre = self.track.get_distance_from_centre_line(self.current_vehicle_position)

        progress_made = self.current_track_state['progress_made']
        current_progress = self.current_track_state['current_progress']
        is_outside_track = self.current_track_state['is_outside_track']
        collision_info = self.current_track_state['collision']
        collision = collision_info['collision']

        progress_timeout = False
        episode_timeout = False
        self.terminate = False

        if is_outside_track or collision:
            self.terminate = True

        if (time.time() - self.time_of_last_progress > PROGRESS_TIMEOUT) and (PROGRESS_TIMEOUT != 0):
            self.terminate = True
            progress_timeout = True
        
        if ((time.time() - self.episode_timer > EPISODE_TIME_LIMIT) and EPISODE_TIME_LIMIT != 0):
            self.terminate = True
            episode_timeout = True

        if (current_progress > 0.95 and self.initial_step_counter <= 0):
            self.finished = True

        if self.target_progress == 0.0:
            if self.current_vehicle_position == (VEHICLE_START_POSITION_X, VEHICLE_START_POSITION_Y):
                self.target_progress = 0.0
            else:
                self.target_progress = current_progress + PROGRESS_BETWEEN_CHECKPOINTS

        reached = False
        if current_progress >= self.target_progress and self.target_progress != 0.0:
            reached = True
            self.checkpoint_counter += 1

        reward = 0
        reward = self.calculate_reward(reached=reached, terminated=self.terminate, finished=self.finished)

        if reached:
            self.target_progress = current_progress + PROGRESS_BETWEEN_CHECKPOINTS

        if reward > 0:
            self.time_of_last_progress = time.time()
            
        observation = self._get_obs()
        info = self._get_info()

        self.previous_steer_command = self.steer_request

        self.timestep_counter += 1

        if self.initial_step_counter != 0:
            reward = 0
            self.initial_step_counter -= 1

        if self.same_position_timout_counter < 0:
            self.terminate = True
            self.position_timout_flag = True
            reward = 0

        self.total_reward += reward

        print("\033c", end="")
        print(f"vehicle position:           {self.current_vehicle_position}")
        print(f"timestep:                   {self.timestep_counter}")
        print(f"velocity:                   {self.velocity_ms} m/s")
        print(f"start offset:               {self.start_offset}")
        print(f"initial episodes:           {self.initial_episodes_in_start_position_counter}")
        print(f"current progress:           {current_progress}")
        print(f"target progress:            {self.target_progress}")
        print(f"progress made:              {progress_made}")
        print(f"given reward:               {reward}")
        print(f"total reward:               {self.total_reward}")
        print(f"is outside track:           {is_outside_track}")
        print(f"collision:                  {collision_info}")
        print(f"timeout:                    {progress_timeout}")
        print(f"episode timeout:            {episode_timeout}")
        print(f"terminate:                  {self.terminate}")
        print(f"weird crash counter:        {self.crash_counter}")
        print(f"step time:                  {time.time() - start_time}")


        return observation, reward, self.terminate, False, info
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # self.track = None
        self.send_data = False

        if (POSITION_RANDOMISATION):
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
        pyautogui.click(1878, 1017)
        time.sleep(SLEEP_AFTER_STOP)
        self.recieved_observation = False
        pyautogui.click(1878, 1017)

        if self.position_timout_flag:
            time.sleep(SLEEP_AFTER_POSITION_TIMOUT)
            self.crash_counter += 1

        pyautogui.click(1876, 969)
        time.sleep(SLEEP_AFTER_START)
        # pyautogui.click(1876, 969)
        # time.sleep(SLEEP_AFTER_START)

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

        self.checkpoint_counter = 0

        self.finished = False

        self.initial_step_counter = INITIAL_STEP_COUNTER

        self.current_vehicle_position = (VEHICLE_START_POSITION_X, VEHICLE_START_POSITION_Y)
        self.previous_vehicle_position = self.current_vehicle_position

        self.same_position_timout_counter = SAME_POSITION_TIMEOUT

        self.vehicle_commands = {
                "steer": 0.0,
                "rpm": 0,
                "brake": 0
            } 

        self.time_of_last_progress = time.time()
        self.episode_timer = time.time()

        self.send_data = True

        while not self.recieved_observation:
            pyautogui.click(1876, 969)
            time.sleep(SLEEP_AFTER_START * 2)

        # print("reached")
        return observation, info


        # ADD LINE TO MOVE VEHICLE POSITION