import gym
import numpy as np
import zmq
import pickle
import rclpy
import time
import threading

from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PoseWithCovarianceStamped, Point
from bristol_msgs.msg import ConeArrayWithCovariance
from ads_dv_msgs.msg import AI2VCURequests, AutonomousState, WheelSpeeds
from visualization_msgs.msg import Marker

TORQUE_REQUEST = 190                        # Torque Request
ESTOP_REQUEST = 0                           # Emergency stop request
MISSION_STATUS = 2                          # Mission status of the vehicle
DIRECTION_REQUEST = 1                       # Gearbox request
MAX_DETECTIONS = 5

context = zmq.Context()

socket_observation = context.socket(zmq.PUB)
socket_observation.bind("tcp://*:5555")
socket_observation.setsockopt(zmq.SNDHWM, 1)
socket_observation.setsockopt(zmq.IMMEDIATE, 1)

socket_map = context.socket(zmq.PUB)
socket_map.bind("tcp://*:5556")
socket_map.setsockopt(zmq.SNDHWM, 1)
socket_map.setsockopt(zmq.IMMEDIATE, 1)

socket_vehicle_condition = context.socket(zmq.PUB)
socket_vehicle_condition.bind("tcp://*:5557")
socket_vehicle_condition.setsockopt(zmq.SNDHWM, 1)
socket_vehicle_condition.setsockopt(zmq.IMMEDIATE, 1)

socket_wheel_speed = context.socket(zmq.PUB)
socket_wheel_speed.bind("tcp://*:5558")
socket_wheel_speed.setsockopt(zmq.SNDHWM, 1)
socket_wheel_speed.setsockopt(zmq.IMMEDIATE, 1)

socket_vehicle_commands = context.socket(zmq.SUB)
socket_vehicle_commands.connect("tcp://localhost:5554")
socket_vehicle_commands.setsockopt_string(zmq.SUBSCRIBE, "")
socket_vehicle_commands.setsockopt(zmq.RCVTIMEO, 10)

socket_what_vehicle_sees = context.socket(zmq.SUB)
socket_what_vehicle_sees.connect("tcp://localhost:5559")
socket_what_vehicle_sees.setsockopt_string(zmq.SUBSCRIBE, "")
socket_what_vehicle_sees.setsockopt(zmq.RCVTIMEO, 1)

class CmEnv(gym.Env, Node):
    def __init__(self):
        Node.__init__(self, 'carmaker_gym_env')
        
        # ROS2 Publisher and Subscriber
        self.request_publisher = self.create_publisher(AI2VCURequests, '/AI2VCU/requests', 10)
        self.left_observation_publisher = self.create_publisher(Marker, '/sac_observations/left', 10)
        self.right_observation_publisher = self.create_publisher(Marker, '/sac_observations/right', 10)

        self.observation_subscription = self.create_subscription(ConeArrayWithCovariance, '/carmaker/cone_detections', self.observation_callback, 10)
        self.map_subscription = self.create_subscription(ConeArrayWithCovariance, '/carmaker/ground_truth/map', self.map_callback, 10)
        self.vehicle_condition_subscription = self.create_subscription(PoseWithCovarianceStamped, '/carmaker/ground_truth/pose', self.vehicle_condition_callback, 10)
        self.wheel_speed_subscription = self.create_subscription(WheelSpeeds, 'VCU2AI/wheel_speeds', self.wheel_speed_callback, 10)

        self.vehicle_commands_timer = self.create_timer(0.001, self.vehicle_commands_timer_callback)
        self.what_vehicle_sees_timer = self.create_timer(0.001, self.what_vehicle_sees_timer_callback)

        self.recieved_map = False
        self.vehicle_commands = {
            "steer": 0.0,
            "rpm": 0,
            "brake": 0
        }

        self.blue_cones = Marker()
        self.blue_cones.header.frame_id = "world"
        self.blue_cones.ns = "blue_points_ns"
        self.blue_cones.type = Marker.POINTS
        self.blue_cones.action = Marker.ADD
        self.blue_cones.id = 0
        self.blue_cones.scale.x = 0.1
        self.blue_cones.scale.y = 0.1
        self.blue_cones.color.r = 0.0
        self.blue_cones.color.g = 0.0
        self.blue_cones.color.b = 1.0
        self.blue_cones.color.a = 1.0

        self.yellow_cones = Marker()
        self.yellow_cones.header.frame_id = "world"
        self.yellow_cones.ns = "yellow_points_ns"
        self.yellow_cones.type = Marker.POINTS
        self.yellow_cones.action = Marker.ADD
        self.yellow_cones.id = 1
        self.yellow_cones.scale.x = 0.1
        self.yellow_cones.scale.y = 0.1
        self.yellow_cones.color.r = 1.0
        self.yellow_cones.color.g = 1.0
        self.yellow_cones.color.b = 0.0
        self.yellow_cones.color.a = 1.0

        self.observation = None

        self.blue_cones_recieved = [(0.0,0.0,0.0)]
        self.yellow_cones_recieved = [(0.0,0.0,0.0)]

        self.blue_cones_map = []
        self.yellow_cones_map = []

        self.cones_seen_by_vehicle = []
        self.position = (0,0)
        self.wheel_speed = 0

        self.send_observation_to_gym()   
        self.send_map_to_gym() 
        self.send_position_to_gym()
        self.send_wheelspeed_to_gym()
    
    # def observation_callback(self, msg):
    #     blue_cones_detected = np.zeros((MAX_DETECTIONS, 3))
    #     yellow_cones_detected = np.zeros((MAX_DETECTIONS,3))

    #     current_cones_seen = msg
    #     self.observation = current_cones_seen

    #     for cone in blue_cones_detected:
    #         if current_cones_seen.blue_cones:
    #             cone[0] = current_cones_seen.blue_cones[0].point.x
    #             cone[1] = current_cones_seen.blue_cones[0].point.y
    #             cone[2] = current_cones_seen.blue_cones[0].point.z

    #             point = Point()

    #             point.x = current_cones_seen.blue_cones[0].point.x
    #             point.y = current_cones_seen.blue_cones[0].point.y
    #             point.z = current_cones_seen.blue_cones[0].point.z

    #             self.blue_cones.points.append(point)

    #             current_cones_seen.blue_cones.pop(0)
    #         else:
    #             break
        
    #     for cone in yellow_cones_detected:
    #         if current_cones_seen.yellow_cones:
    #             cone[0] = current_cones_seen.yellow_cones[0].point.x
    #             cone[1] = current_cones_seen.yellow_cones[0].point.y
    #             cone[2] = current_cones_seen.yellow_cones[0].point.z

    #             point = Point()

    #             point.x = current_cones_seen.yellow_cones[0].point.x
    #             point.y = current_cones_seen.yellow_cones[0].point.y
    #             point.z = current_cones_seen.yellow_cones[0].point.z

    #             self.yellow_cones.points.append(point)

    #             current_cones_seen.yellow_cones.pop(0)
    #         else:
    #             break                    
        
    #     socket_observation.send(pickle.dumps((blue_cones_detected, yellow_cones_detected)))
    #     # print("sent observation")

    def observation_callback(self, msg):
        start = time.time()
        self.blue_cones_recieved = []
        self.yellow_cones_recieved = []

        for cone in msg.blue_cones:
            self.blue_cones_recieved.append((cone.point.x, cone.point.y, cone.point.z))

        for cone in msg.yellow_cones:
            self.yellow_cones_recieved.append((cone.point.x, cone.point.y, cone.point.z))
        
        
        print("time to observe", time.time() - start)

    def send_observation_to_gym(self):
        socket_observation.send(pickle.dumps((self.blue_cones_recieved, self.yellow_cones_recieved)), zmq.DONTWAIT)
        threading.Timer(0.001, self.send_observation_to_gym).start()

    def send_map_to_gym(self):
        if len(self.blue_cones_map) > 0 and len(self.yellow_cones_map) > 0:
            socket_map.send(pickle.dumps((self.blue_cones_map, self.yellow_cones_map)), zmq.DONTWAIT)
        threading.Timer(0.001, self.send_map_to_gym).start()

    def send_position_to_gym(self):
        socket_vehicle_condition.send(pickle.dumps(self.position), zmq.DONTWAIT)
        threading.Timer(0.001, self.send_position_to_gym).start()
    
    def send_wheelspeed_to_gym(self):
        socket_wheel_speed.send(pickle.dumps(self.wheel_speed), zmq.DONTWAIT)
        threading.Timer(0.001, self.send_wheelspeed_to_gym).start()
        
    
    def map_callback(self, msg):
        self.blue_cones_map = []
        self.yellow_cones_map = []
        for cone in msg.blue_cones:
            self.blue_cones_map.append((cone.point.x, cone.point.y))

        for cone in msg.yellow_cones:
            self.yellow_cones_map.append((cone.point.x, cone.point.y))
        self.recieved_map = True

        # print(blue_cones_map)
        # print("#"*10)
        # print(yellow_cones_map)
        
        # print("sent map")
    
    def vehicle_condition_callback(self, msg):
        self.position = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        # print(position)
        # print("sent vehicle conditions")

    def wheel_speed_callback(self, msg):
        wheel_speed_lf = msg.fl_speed_rpm
        wheel_speed_rf = msg.fr_speed_rpm
        wheel_speed_lb = msg.rl_speed_rpm
        wheel_speed_rb = msg.rr_speed_rpm

        self.wheel_speed = (wheel_speed_lf + wheel_speed_rf + wheel_speed_lb + wheel_speed_rb) / 4

    def vehicle_commands_timer_callback(self):
        # print("reached 1")
        start_time = time.time()
        try:
            self.vehicle_commands = pickle.loads(socket_vehicle_commands.recv())
        except zmq.error.Again:
            self.vehicle_commands = {
                "steer": 0.0,
                "rpm": 0,
                "brake": 0
            }        
        end_time = time.time()

        msg = AI2VCURequests()
        msg.steer_request_deg               = self.vehicle_commands["steer"]
        msg.front_axle_trq_request          = TORQUE_REQUEST
        msg.front_motor_speed_max           = int(self.vehicle_commands["rpm"])
        msg.rear_axle_trq_request           = TORQUE_REQUEST
        msg.rear_motor_speed_max            = int(self.vehicle_commands["rpm"])
        msg.hyd_press_f_req_pct             = int(self.vehicle_commands["brake"])
        msg.hyd_press_r_req_pct             = int(self.vehicle_commands["brake"])
        msg.estop_request                   = ESTOP_REQUEST
        msg.mission_status                  = MISSION_STATUS
        msg.direction_request               = DIRECTION_REQUEST


        self.request_publisher.publish(msg)
        
        print("\033c", end="")
        print(f"steer_request_deg:          {msg.steer_request_deg}")
        print(f"front_axle_trq_request:     {msg.front_axle_trq_request}")
        print(f"front_motor_speed_max:      {msg.front_motor_speed_max}")
        print(f"rear_axle_trq_request:      {msg.rear_axle_trq_request}")
        print(f"rear_motor_speed_max:       {msg.rear_motor_speed_max}")
        print(f"hyd_press_f_req_pct:        {msg.hyd_press_f_req_pct}")
        print(f"hyd_press_r_req_pct:        {msg.hyd_press_r_req_pct}")
        print(f"estop_request:              {msg.estop_request}")
        print(f"mission_status:             {msg.mission_status}")
        print(f"direction_request:          {msg.direction_request}")
        print(f"time to recieve commands:   {end_time - start_time}")
 

    def what_vehicle_sees_timer_callback(self):
        try:
            self.cones_seen_by_vehicle = pickle.loads(socket_what_vehicle_sees.recv())
        except zmq.error.Again:
            pass
            
        self.cones_seen_by_vehicle = (self.blue_cones_recieved, self.yellow_cones_recieved)
        if len(self.cones_seen_by_vehicle) != 0 and self.cones_seen_by_vehicle != None: 
            for cone in self.cones_seen_by_vehicle[0]:
                point = Point()

                point.x = cone[0]
                point.y = cone[1]
                point.z = cone[2]

                self.blue_cones.points.append(point)

            for cone in self.cones_seen_by_vehicle[1]:
                point = Point()

                point.x = cone[0]
                point.y = cone[1]
                point.z = cone[2]

                self.yellow_cones.points.append(point)

        self.left_observation_publisher.publish(self.blue_cones)
        self.right_observation_publisher.publish(self.yellow_cones)

        self.blue_cones = Marker()
        self.blue_cones.header.frame_id = "world"
        self.blue_cones.ns = "blue_points_ns"
        self.blue_cones.type = Marker.POINTS
        self.blue_cones.action = Marker.ADD
        self.blue_cones.id = 0
        self.blue_cones.scale.x = 0.1
        self.blue_cones.scale.y = 0.1
        self.blue_cones.color.r = 0.0
        self.blue_cones.color.g = 0.0
        self.blue_cones.color.b = 1.0
        self.blue_cones.color.a = 1.0

        self.yellow_cones = Marker()
        self.yellow_cones.header.frame_id = "world"
        self.yellow_cones.ns = "yellow_points_ns"
        self.yellow_cones.type = Marker.POINTS
        self.yellow_cones.action = Marker.ADD
        self.yellow_cones.id = 1
        self.yellow_cones.scale.x = 0.1
        self.yellow_cones.scale.y = 0.1
        self.yellow_cones.color.r = 1.0
        self.yellow_cones.color.g = 1.0
        self.yellow_cones.color.b = 0.0
        self.yellow_cones.color.a = 1.0

def main(args=None):
    rclpy.init(args=args)

    cm_env = CmEnv()

    rclpy.spin(cm_env)

    rclpy.shutdown()

    # executor = rclpy.executors.MultiThreadedExecutor()
    # executor.add_node(cm_env)

    # try:
    #     executor.spin()
    # except KeyboardInterrupt:
    #     pass
    # finally:
    #     cm_env.destroy_node()
    #     rclpy.shutdown()

if __name__ == '__main__':
    main()