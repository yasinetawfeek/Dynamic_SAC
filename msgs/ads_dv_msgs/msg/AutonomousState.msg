# State of the Autonomous System
uint16 as_state

uint16 AS_OFF=1
uint16 AS_READY=2
uint16 AS_DRIVING=3
uint16 AS_EMERGENCY_BRAKE=4
uint16 AS_FINISHED=5

# Mission indicator
uint16 ami_state

uint16 AMI_NOT_SELECTED=0
uint16 AMI_ACCELERATION=1
uint16 AMI_SKIDPAD=2
uint16 AMI_AUTOCROSS=3
uint16 AMI_TRACK_DRIVE=4
uint16 AMI_STATIC_INSPECTION_A=5
uint16 AMI_STATIC_INSPECTION_B=6
uint16 AMI_AUTONOMOUS_DEMO=7

# RES go signal
uint16 res_go_signal

uint16 RES_GO_SIGNAL_NO_GO = 0
uint16 RES_GO_SIGNAL_GO = 1
