cmake_minimum_required(VERSION 2.8.3)
project(lio_lite)

######################
### Cmake flags
######################
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++17")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g -pthread")

######################
### Packages
######################
find_package(catkin REQUIRED COMPONENTS
    tf
    roscpp
    rospy
    roslib
    # msg
    std_msgs
    sensor_msgs
    geometry_msgs
    nav_msgs
    livox_ros_driver2
    # cv
    cv_bridge
    # pcl
    pcl_conversions
    # msg generation
    message_generation
)

find_package(OpenMP REQUIRED)
find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
set(GTSAM_DIR ${PROJECT_SOURCE_DIR}/3rdparty/gtsam/lib/cmake/GTSAM)
find_package(GTSAM REQUIRED QUIET)
find_package(Boost REQUIRED COMPONENTS filesystem program_options system timer)


######################
### Message generation
######################
add_message_files(
  DIRECTORY msg
  FILES
  cloud_info.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
)

######################
### Catkin
######################
catkin_package(
  DEPENDS PCL GTSAM
)

include_directories(
  3rdparty/Eigen
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${GTSAM_INCLUDE_DIR}
)

link_directories(
  ${PCL_LIBRARY_DIRS}
  ${OpenCV_LIBRARY_DIRS}
  ${GTSAM_LIBRARY_DIRS}
)


######################
### lidar odometry
######################

# IMU Preintegration
add_executable(${PROJECT_NAME}_imuPreintegration src/imuPreintegration.cpp)
target_link_libraries(${PROJECT_NAME}_imuPreintegration ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} gtsam tbb Boost::timer)
# Range Image Projection
add_executable(${PROJECT_NAME}_imageProjection src/imageProjection.cpp)
add_dependencies(${PROJECT_NAME}_imageProjection ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_generate_messages_cpp)
target_link_libraries(${PROJECT_NAME}_imageProjection ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES})
# Feature Association
add_executable(${PROJECT_NAME}_featureExtraction src/featureExtraction.cpp)
add_dependencies(${PROJECT_NAME}_featureExtraction ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_generate_messages_cpp)
target_link_libraries(${PROJECT_NAME}_featureExtraction ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES})

# Mapping Optimization
add_executable(${PROJECT_NAME}_mapOptmization src/mapOptmization.cpp)
add_dependencies(${PROJECT_NAME}_mapOptmization ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_generate_messages_cpp)
target_compile_options(${PROJECT_NAME}_mapOptmization PRIVATE ${OpenMP_CXX_FLAGS})
target_link_libraries(${PROJECT_NAME}_mapOptmization PRIVATE ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} ${OpenMP_CXX_FLAGS} gtsam tbb Boost::timer)
# Global Localization
add_executable(${PROJECT_NAME}_globalLocalize src/globalLocalize.cpp)
add_dependencies(${PROJECT_NAME}_globalLocalize ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_generate_messages_cpp)
target_compile_options(${PROJECT_NAME}_globalLocalize PRIVATE ${OpenMP_CXX_FLAGS})
target_link_libraries(${PROJECT_NAME}_globalLocalize PRIVATE ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} ${OpenMP_CXX_FLAGS} gtsam tbb Boost::timer)
