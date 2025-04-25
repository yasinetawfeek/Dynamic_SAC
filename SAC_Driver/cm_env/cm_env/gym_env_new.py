import gymnasium
import numpy as np
import zmq
import pickle
import time
import threading
import pyautogui
import random
from pynput.keyboard import Key, Controller
from gymnasium import spaces
from boundary_detect_v2 import TrackBoundary
from ai2vcu_requests import AI2VCU_REQUESTS
from config import *
from track_maps.silverstone import *

context = zmq.Context()
keyboard = Controller()

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

        low = -np.inf * np.ones((MAX_DETECTIONS,2))
        high = np.inf * np.ones((MAX_DETECTIONS,2))
        
        self.observation_space = spaces.Dict({
            'blue_cones': spaces.Box(low=low, high=high, shape=(MAX_DETECTIONS, 2), dtype=np.float32),
            'yellow_cones': spaces.Box(low=low, high=high, shape=(MAX_DETECTIONS, 2), dtype=np.float32),
            'velocity': spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            'previous_steer': spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
            # 'distance_from_cl': spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        })

        # Internal state
        self.current_observation = np.zeros(3, dtype=np.float32)
        self.done = False
        self.recieved_map = False

        self.track = None
        self.current_track_state = None
        self.blue_cones_detected = np.zeros((MAX_DETECTIONS, 2))
        self.yellow_cones_detected = np.zeros((MAX_DETECTIONS,2))

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

        self.current_checkpoint_history = []
        self.checkpoint_history_bank = []
        self.control_history_bank = []
        self.current_control_history = []

        self.initial_episodes_in_start_position_counter = EPISODES_AT_START_POSITION

        self.current_vehicle_position = (VEHICLE_START_POSITION_X, VEHICLE_START_POSITION_Y)
        self.previous_vehicle_position = self.current_vehicle_position
        self.time_of_last_progress = time.time()

        self.training_time_outside_step = time.time()
        self.start_time = 0


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
        self.episode_start_time = time.time()

        self.brake_timeout_counter = BRAKE_TIMEOUT_COUNTER
        self.initial_progress = 0

        self.get_vehicle_position()
        self.get_vehicle_observation()
        self.get_wheel_speeds()
        self.send_vehicle_commands()

        self.time_of_last_step = time.time()

        
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
        # self.blue_cones_detected = np.zeros((MAX_DETECTIONS, 2))
        # self.yellow_cones_detected = np.zeros((MAX_DETECTIONS, 2))
        try:
            message = socket_observation.recv(zmq.DONTWAIT)
            (blue_cones, yellow_cones) = pickle.loads(message)

            self.blue_cones_detected = np.zeros((MAX_DETECTIONS, 2))
            self.yellow_cones_detected = np.zeros((MAX_DETECTIONS, 2))

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
            'previous_steer': self.previous_steer_command / MAX_STEERING_ANGLE,
            # 'distance_from_cl': self.distance_from_centre
        }
    
    def get_obs_deployment(self):
        return dict({
            'blue_cones': np.array(self.blue_cones_detected, dtype=np.float32),
            'yellow_cones': np.array(self.yellow_cones_detected, dtype=np.float32),
            'velocity': np.array([self.velocity_ms], dtype=np.float32),
            'previous_steer': np.array([self.previous_steer_command / MAX_STEERING_ANGLE], dtype=np.float32),
            # 'distance_from_cl': self.distance_from_centre
        })
    
    def _get_info(self):
        return {"hey": 4}

    def calculate_reward(self, reached, terminated, finished, start_prog = 0):
        remaining_prog = 1 - start_prog
        reward = 0
        if terminated:
            reward = FUNCTION_FINAL_A2
        elif finished:
            reward = FUNCTION_FINAL_A3 / self.timestep_counter
        elif reached:
            reward = (FUNCTION_FINAL_A1 / remaining_prog) * (DISCOUNT_FACTOR ** self.checkpoint_counter)
        
        return reward - (0.005/remaining_prog)

    def step(self, action):
        while time.time() - self.training_time_outside_step < ACTION_INTREVAL and time.time() - self.training_time_outside_step > 0:
            pass
        self.start_time = time.time()

        if self.current_vehicle_position == self.previous_vehicle_position:
            self.same_position_timout_counter -= 1
        else:
            self.same_position_timout_counter = SAME_POSITION_TIMEOUT

        self.previous_vehicle_position = self.current_vehicle_position

        # print("hi")
        if self.track == None:
            # message = pickle.loads(socket_map.recv())
            # message[0].append(message[0][0])
            # message[1].append(message[1][0])
            # self.track = TrackBoundary(message[0], message[1])
            self.track = TrackBoundary(blue_cones, yellow_cones)

        if action[0] >= 0:
            self.rpm_request = action[0] * MAX_RPM
            self.brake_request = 0
        else:
            self.brake_request = abs(action[0] * MAX_BRAKE_PERCENTAGE)
            self.rpm_request = 0

        proposed_steer_command = action[1] * MAX_STEERING_ANGLE
        # proposed_steer_command = -21.0


        if abs(self.previous_steer_command - proposed_steer_command) > MAX_STEERING_ANGLE_CHANGE_PER_STEP:
            if proposed_steer_command > self.previous_steer_command:
                self.steer_request = (self.previous_steer_command + MAX_STEERING_ANGLE_CHANGE_PER_STEP)
            else:
                self.steer_request = (self.previous_steer_command - MAX_STEERING_ANGLE_CHANGE_PER_STEP) 
        else:
            self.steer_request = proposed_steer_command

        # self.steer_request = proposed_steer_command

        self.vehicle_commands = {
            "steer": self.steer_request,
            "rpm": self.rpm_request,
            "brake": self.brake_request
        }

        self.current_control_history.append((self.steer_request, self.rpm_request/MAX_RPM, self.brake_request/MAX_BRAKE_PERCENTAGE))

        if (self.brake_request > 0):
            self.brake_timeout_counter -= 1
        else:
            self.brake_timeout_counter = BRAKE_TIMEOUT_COUNTER

        if (self.brake_timeout_counter <= 0):
            self.terminate = True

        # self.vehicle_commands = {
        #     "steer": self.steer_request,
        #     "rpm": 400,
        #     "brake": 0
        # }
        while(time.time() - self.start_time < OBSERVATION_INTREVAL):
            pass

        self.current_track_state = self.track.update_positon(self.current_vehicle_position)
        # self.distance_from_centre = self.track.get_distance_from_centre_line(self.current_vehicle_position)

        # self.distance_from_centre = self.distance_from_centre / DISTANCE_FROM_CL_TO_SIDE

        # if self.distance_from_centre > 1.0:
        #     self.distance_from_centre = 1.0

        progress_made = self.current_track_state['progress_made']
        current_progress = self.current_track_state['current_progress']
        is_outside_track = self.current_track_state['is_outside_track']
        collision_info = self.current_track_state['collision']
        collision = collision_info['collision']

        if self.initial_progress == 0 and current_progress != 0:
            self.initial_progress = current_progress

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

        if (current_progress > PROGRESS_FOR_COMPLETION and self.initial_step_counter <= 0):
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
            self.current_checkpoint_history.append((self.current_vehicle_position, self.velocity_ms))

        reward = 0
        if (POSITION_RANDOMISATION):
            reward = self.calculate_reward(reached=reached, terminated=self.terminate, finished=self.finished, start_prog=self.initial_progress)
        else:
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
        output = f"vehicle position:                 {self.current_vehicle_position}\n" +\
        f"timestep:                         {self.timestep_counter}\n" +\
        f"previous steer:                   {self.previous_steer_command / MAX_STEERING_ANGLE}\n" +\
        f"velocity:                         {self.velocity_ms} m/s\n" +\
        f"start offset:                     {self.start_offset}\n" +\
        f"initial progress:                 {(self.initial_progress * 100):.3f}%\n" +\
        f"initial episodes:                 {self.initial_episodes_in_start_position_counter}\n" +\
        f"current progress:                 {(current_progress * 100):.3f}%\n" +\
        f"target progress:                  {(self.target_progress * 100):.3f}%\n" +\
        f"progress made:                    {(progress_made * 100):.3f}%\n" +\
        f"given reward:                     {reward}\n" +\
        f"total reward:                     {self.total_reward}\n" +\
        f"is outside track:                 {is_outside_track}\n" +\
        f"collision:                        {collision_info}\n" +\
        f"timeout:                          {progress_timeout}\n" +\
        f"episode timeout:                  {episode_timeout}\n" +\
        f"terminate:                        {self.terminate}\n" +\
        f"weird crash counter:              {self.crash_counter}\n" +\
        f"time between action and obs:      {time.time() - self.start_time} \n" +\
        f"time between obs and action:      {self.start_time - self.training_time_outside_step}\n" +\
        f"time between action and action:   {time.time() - self.time_between_actions}\n" +\
        f"checkpoints achieved:             {len(self.current_checkpoint_history)}\n" +\
        f"progressive episodes:             {len(self.checkpoint_history_bank)}" # how many episodes where there more than 1 checkpoint reached
        self.time_between_actions = time.time()
        print("\033c", end="")
        print(output)
        self.training_time_outside_step = time.time()


        return observation, reward, self.terminate or self.finished, False, info
    
    def handle_track_randomisation(self):
        with open(PATH_TO_TESTRUN_FILE, 'r') as file:
            data = file.readlines()
            
            if self.initial_episodes_in_start_position_counter > 0:
                self.start_offset = VEHICLE_START_POSITION_X
                self.initial_episodes_in_start_position_counter -= 1
            else:
                self.start_offset = float(random.randint(MIN_SPAWN, MAX_SPAWN))

            fog_active = random.randint(0,1)
            for line in range(len(data)):
                if "Vehicle.StartPos =" in data[line]:
                    data[line] = f'Vehicle.StartPos = {self.start_offset} 0.0\n'
                if "Env.FogActive =" in data[line]:
                    data[line] = f'Env.FogActive = {fog_active}\n'
                if "Env.VisRangeInFog" in data[line]:
                    if fog_active:
                        data[line] = f"Env.VisRangeInFog = {random.randint(20, 100)}\n"
                    else:
                        data[line] = f"Env.VisRangeInFog = {1000}\n"
                if 'Env.RainRate =' in data[line]:
                    data[line] = f"Env.RainRate = {random.randint(0, 10)/10}\n"

            with open(PATH_TO_TESTRUN_FILE, 'w') as file:
                file.writelines(data)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # self.track = None
        self.send_data = False

        if (POSITION_RANDOMISATION):
            self.handle_track_randomisation()

        pyautogui.click(1878, 1017)
        pyautogui.click(1878, 1017)

        pyautogui.click(1768, 183)
        keyboard.press(Key.ctrl)
        keyboard.press('c')
        keyboard.release('c')
        keyboard.release(Key.ctrl)

        time.sleep(SLEEP_AFTER_STOP)

        self.recieved_observation = False
        pyautogui.click(1878, 1017)

        if self.position_timout_flag:
            time.sleep(SLEEP_AFTER_POSITION_TIMOUT)
            self.crash_counter += 1

        pyautogui.click(1876, 969)
        time.sleep(SLEEP_AFTER_START)

        # 2. Press Up Arrow
        pyautogui.click(1768, 183)
        keyboard.press(Key.up)
        keyboard.release(Key.up)

        time.sleep(0.2)

        # 3. Press Enter
        keyboard.press(Key.enter)
        keyboard.release(Key.enter)

        time.sleep(0.5)


        # pyautogui.click(1876, 969)
        # time.sleep(SLEEP_AFTER_START)

        self.track = None

        self.rpm_request = 0
        self.brake_request = 0
        self.steer_request = 0

        self.brake_timeout_counter = BRAKE_TIMEOUT_COUNTER

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
        self.blue_cones_detected = np.zeros((MAX_DETECTIONS, 2))
        self.yellow_cones_detected = np.zeros((MAX_DETECTIONS,2))

        self.rpm_request = 0
        self.brake_request = 0
        self.steer_request = 0
        self.total_reward = 0

        self.target_progress = 0.0
        self.initial_timestep = 0
        self.start_offset = 0
        self.distance_from_centre = 0
        self.previous_steer_command = 0.0

        self.wheel_speed = 0.0
        self.velocity_ms = 0.0

        self.checkpoint_counter = 0
        if (len(self.current_checkpoint_history) > 1):
            self.checkpoint_history_bank.append((self.current_checkpoint_history, self.finished, self.timestep_counter, time.time() - self.episode_start_time))
        if (len(self.current_control_history) > 100):
            self.control_history_bank.append((self.current_control_history, self.finished, self.timestep_counter, time.time() - self.episode_start_time))
        self.current_checkpoint_history = []
        self.current_control_history = []
        self.timestep_counter = 0

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
            if (POSITION_RANDOMISATION):
                self.handle_track_randomisation()
                    
            time.sleep(SLEEP_AFTER_START * 2)

        # print("reached")

        self.initial_progress = 0
        self.training_time_outside_step = time.time()
        self.start_time = 0
        self.time_between_actions = time.time()
        self.episode_start_time = time.time()

        return observation, info


        # ADD LINE TO MOVE VEHICLE POSITION