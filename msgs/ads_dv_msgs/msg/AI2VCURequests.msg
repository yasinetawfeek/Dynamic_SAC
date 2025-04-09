# AI2VCU_steer
float32 steer_request_deg

# AI2VCU_drive stuff
uint16 front_axle_trq_request
uint16 front_motor_speed_max
uint16 rear_axle_trq_request
uint16 rear_motor_speed_max

# AI2VCU_brake stuff
uint16 hyd_press_f_req_pct
uint16 hyd_press_r_req_pct

# AI2VCU_status stuff
uint16 estop_request
uint16 mission_status
uint16 direction_request

# Values for the ESTOP
uint16 ESTOP_NO_STOP = 0
uint16 ESTOP_STOP = 1

# Values for the mission status
uint16 MISSION_STATUS_NO_SELECTION = 0
uint16 MISSION_STATUS_SELECTED = 1
uint16 MISSION_STATUS_RUNNING = 2
uint16 MISSION_STATUS_FINISHED = 3

# Values for the direction
uint16 DIRECTION_NEUTRAL = 0
uint16 DIRECTION_FORWARD = 1

