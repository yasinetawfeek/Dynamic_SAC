cmake_minimum_required(VERSION 3.5)
project(bristol_msgs)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

# List of message files
set(msg_files
  "msg/AckermannDrive.msg"
  "msg/AckermannDriveStamped.msg"
  "msg/CanState.msg"
  "msg/ConeArrayWithCovariance.msg"
  "msg/ConeWithCovariance.msg"
  "msg/PointArray.msg"
  "msg/WheelSpeeds.msg"
  "msg/WheelSpeedsStamped.msg"
)

set(action_files
)

set(srv_files
)

# Generate messages & services
rosidl_generate_interfaces(${PROJECT_NAME}
  ${msg_files}
  ${action_files}
  ${srv_files}
  DEPENDENCIES std_msgs geometry_msgs sensor_msgs
)

ament_export_dependencies(rosidl_default_runtime)

ament_package()
