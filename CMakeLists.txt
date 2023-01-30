cmake_minimum_required(VERSION 2.8.3)
project(dlo3ds)

add_compile_options(-std=c++11)
set(CMAKE_CXX_STANDARD 14)

find_package(catkin REQUIRED COMPONENTS
  actionlib
  actionlib_msgs
  geometry_msgs
  moveit_msgs
  moveit_core
  moveit_visual_tools
  moveit_ros_planning_interface
  roscpp
  std_msgs
  sensor_msgs
  cv_bridge
  image_transport
)

find_package(Eigen3 REQUIRED)
find_package(Franka 0.5.0 REQUIRED)
find_package(OpenCV REQUIRED)

catkin_package(
  CATKIN_DEPENDS 
  actionlib
  actionlib_msgs
  geometry_msgs
  moveit_msgs 
  moveit_core
  moveit_visual_tools
  moveit_ros_planning_interface
  roscpp 
  rospy 
  sensor_msgs
  std_msgs 
  DEPENDS Franka

)







