cmake_minimum_required(VERSION 3.16)
project(uw_ct_vio LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-Wall -Wextra -Wpedantic)

find_package(ament_cmake REQUIRED)
# GTSAM 4.2.0 built with GTSAM_BUILD_UNSTABLE=ON (see README.md); point GTSAM_DIR at it.
find_package(GTSAM REQUIRED)
find_package(GTSAM_UNSTABLE REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs video calib3d)
find_package(rclcpp REQUIRED)
find_package(rosbag2_cpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(yaml-cpp REQUIRED)

add_library(${PROJECT_NAME}_core
  src/DenseBetweenFactor.cpp
  src/DenseRegistration.cpp
  src/DenseStereo.cpp
  src/DepthGpFactor.cpp
  src/DvlFactor.cpp
  src/Estimator.cpp
  src/KltStereoTracker.cpp
  src/PartialPosePrior.cpp
  src/TrackFile.cpp)
target_include_directories(${PROJECT_NAME}_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(${PROJECT_NAME}_core PUBLIC gtsam gtsam_unstable ${OpenCV_LIBS})

add_executable(run_bag app/run_bag.cpp)
target_link_libraries(run_bag ${PROJECT_NAME}_core yaml-cpp)
ament_target_dependencies(run_bag rclcpp rosbag2_cpp geometry_msgs sensor_msgs)
# The GTSAM prefix is not on the loader path; record it so the installed binary runs as is.
set_target_properties(run_bag PROPERTIES INSTALL_RPATH "${GTSAM_DIR}/../..;${CMAKE_INSTALL_PREFIX}/lib")
install(TARGETS run_bag DESTINATION lib/${PROJECT_NAME})

install(TARGETS ${PROJECT_NAME}_core LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_depth_gp_factor test/test_depth_gp_factor.cpp)
  target_link_libraries(test_depth_gp_factor ${PROJECT_NAME}_core)
  ament_add_gtest(test_dvl_factor test/test_dvl_factor.cpp)
  target_link_libraries(test_dvl_factor ${PROJECT_NAME}_core)
  ament_add_gtest(test_estimator_dvl test/test_estimator_dvl.cpp TIMEOUT 900)
  target_link_libraries(test_estimator_dvl ${PROJECT_NAME}_core)
  ament_add_gtest(test_partial_pose_prior test/test_partial_pose_prior.cpp)
  target_link_libraries(test_partial_pose_prior ${PROJECT_NAME}_core)
  ament_add_gtest(test_estimator_synthetic test/test_estimator_synthetic.cpp TIMEOUT 600)
  target_link_libraries(test_estimator_synthetic ${PROJECT_NAME}_core)
  ament_add_gtest(test_dense test/test_dense.cpp)
  target_link_libraries(test_dense ${PROJECT_NAME}_core)
  ament_add_gtest(test_map_prediction test/test_map_prediction.cpp)
  target_link_libraries(test_map_prediction ${PROJECT_NAME}_core)
endif()

ament_package()
