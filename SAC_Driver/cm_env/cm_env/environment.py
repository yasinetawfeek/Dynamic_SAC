import gym
import numpy as np
import zmq
import pickle
import rclpy
import time

from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
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
# socket_what_vehicle_sees.setsockopt(zmq.RCVTIMEO, 2)


qos_profile = QoSProfile(
    depth=1,  
    reliability=QoSReliabilityPolicy.BEST_EFFORT,  
    durability=QoSDurabilityPolicy.VOLATILE 
)

class CmEnv(gym.Env, Node):
    def __init__(self):
        Node.__init__(self, 'carmaker_gym_env')
        
        # ROS2 Publisher and Subscriber
        self.request_publisher = self.create_publisher(AI2VCURequests, '/AI2VCU/requests', 10)
        self.left_observation_publisher = self.create_publisher(Marker, '/sac_observations/left', 10)
        self.right_observation_publisher = self.create_publisher(Marker, '/sac_observations/right', 10)

        self.observation_subscription = self.create_subscription(ConeArrayWithCovariance, '/carmaker/cone_detections', self.observation_callback, qos_profile)
        self.map_subscription = self.create_subscription(ConeArrayWithCovariance, '/carmaker/ground_truth/map', self.map_callback, qos_profile)
        self.vehicle_condition_subscription = self.create_subscription(PoseWithCovarianceStamped, '/carmaker/ground_truth/pose', self.vehicle_condition_callback, qos_profile)
        self.wheel_speed_subscription = self.create_subscription(WheelSpeeds, 'VCU2AI/wheel_speeds', self.wheel_speed_callback, qos_profile)

        self.timer = self.create_timer(0.001, self.timer_callback)

        self.recieved_map = False
        self.vehicle_commands = None

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

        self.previous_steer = 0.0

        self.blue_cones_recieved = None
        self.yellow_cones_recieved = None    

        self.cones_seen_by_vehicle = []
        self.position = (0,0)

        self.send_command_flag = False
    
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
            # self.blue_cones_recieved.append((cone.point.x, cone.point.y, cone.point.z))
            if cone.point.x < 20 and cone.point.y < 20:
                self.blue_cones_recieved.append((cone.point.x, cone.point.y))


        for cone in msg.yellow_cones:
            # self.yellow_cones_recieved.append((cone.point.x, cone.point.y, cone.point.z))
            if cone.point.x < 20 and cone.point.y < 20:
                self.yellow_cones_recieved.append((cone.point.x, cone.point.y))

        
        socket_observation.send(pickle.dumps((self.blue_cones_recieved, self.yellow_cones_recieved)))
        print("time to observe:            %.5f" %(time.time() - start))
    
    def map_callback(self, msg):
        blue_cones_map = []
        yellow_cones_map = []
        for cone in msg.blue_cones:
            blue_cones_map.append((cone.point.x, cone.point.y))

        for cone in msg.yellow_cones:
            yellow_cones_map.append((cone.point.x, cone.point.y))
        self.recieved_map = True

        # print(blue_cones_map)
        # print("#"*10)
        # print(yellow_cones_map)
        if len(blue_cones_map) > 0 and len(yellow_cones_map) > 0:
            socket_map.send(pickle.dumps((blue_cones_map, yellow_cones_map)))
        # print("sent map")
    
    def vehicle_condition_callback(self, msg):
        self.position = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        # print(position)
        socket_vehicle_condition.send(pickle.dumps(self.position))
        # print("sent vehicle conditions")

    def wheel_speed_callback(self, msg):
        wheel_speed_lf = msg.fl_speed_rpm
        wheel_speed_rf = msg.fr_speed_rpm
        wheel_speed_lb = msg.rl_speed_rpm
        wheel_speed_rb = msg.rr_speed_rpm

        wheel_speed = (wheel_speed_lf + wheel_speed_rf + wheel_speed_lb + wheel_speed_rb) / 4

        socket_wheel_speed.send(pickle.dumps(wheel_speed))


    def timer_callback(self):
        # print("reached 1")
        try:
            start_time = time.time()
            try:
                self.vehicle_commands = pickle.loads(socket_vehicle_commands.recv())

                # proposed_steer_command = self.vehicle_commands['steer']
                # if abs(self.previous_steer - proposed_steer_command) > 0.14:
                #     if proposed_steer_command > self.previous_steer:
                #         self.vehicle_commands['steer'] = (self.previous_steer + 0.14)
                #     else:
                #         self.vehicle_commands['steer'] = (self.previous_steer - 0.14) 
                # else:
                #     self.vehicle_commands['steer'] = proposed_steer_command
                
                # self.previous_steer = proposed_steer_command

                self.send_command_flag = True
            except zmq.Again:
                self.vehicle_commands = {
                    "steer": 0.0,
                    "rpm": 0,
                    "brake": 0
                }
                self.send_command_flag = False

            try:
                self.cones_seen_by_vehicle = pickle.loads(socket_what_vehicle_sees.recv(zmq.DONTWAIT))
            except zmq.Again:
                pass

            # cones_seen_by_vehicle = (self.blue_cones_recieved, self.yellow_cones_recieved)
            if len(self.cones_seen_by_vehicle) == 2:
                for cone in self.cones_seen_by_vehicle[0]:
                    point = Point()

                    point.x = cone[0]
                    point.y = cone[1]
                    # point.z = cone[2]
                    point.z = 0.0


                    self.blue_cones.points.append(point)

                for cone in self.cones_seen_by_vehicle[1]:
                    point = Point()

                    point.x = cone[0]
                    point.y = cone[1]
                    # point.z = cone[2]
                    point.z = 0.0


                    self.yellow_cones.points.append(point)

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

            print("\033c", end="")

            if self.send_command_flag:
                self.request_publisher.publish(msg)
                print("PUBLISHING")
            else:
                print("NOT PUBLISHING")
            self.left_observation_publisher.publish(self.blue_cones)
            self.right_observation_publisher.publish(self.yellow_cones)

            print("steer_request_deg:          %.5f" %msg.steer_request_deg)
            print("front_axle_trq_request:     %d" %msg.front_axle_trq_request)
            print("front_motor_speed_max:      %d" %msg.front_motor_speed_max)
            print("rear_axle_trq_request:      %d" %msg.rear_axle_trq_request)
            print("rear_motor_speed_max:       %d" %msg.rear_motor_speed_max)
            print("hyd_press_f_req_pct:        %d" %msg.hyd_press_f_req_pct)
            print("hyd_press_r_req_pct:        %d" %msg.hyd_press_r_req_pct)
            print("estop_request:              %d" %msg.estop_request)
            print("mission_status:             %d" %msg.mission_status)
            print("direction_request:          %d" %msg.direction_request)
            print("time to send everything:    %.5f" %(time.time() - start_time))


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
        
        except zmq.error.Again:
            pass

def main(args=None):
    rclpy.init(args=args)

    cm_env = CmEnv()

    rclpy.spin(cm_env)

    rclpy.shutdown()

if __name__ == '__main__':
    main()