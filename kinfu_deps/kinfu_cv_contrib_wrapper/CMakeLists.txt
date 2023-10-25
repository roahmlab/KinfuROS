cmake_minimum_required(VERSION 3.0.2)
project(kinfu_cv_contrib_wrapper)

## Find catkin macros and libraries
find_package(catkin REQUIRED)

# Download and set it up in this wrapper using ExternalProject
include(ExternalProject)
include(FetchContent)

# Start initial environment variable stuff
set(PREFIX_DIR            ${CMAKE_CURRENT_BINARY_DIR})
set(FETCHCONTENT_BASE_DIR ${PREFIX_DIR})

# Download VTK with FetchContent so it's available during the configure stage
FetchContent_Declare(
  vtk_820
  URL https://gitlab.kitware.com/vtk/vtk/-/archive/v8.2.0/vtk-v8.2.0.tar.gz
)
FetchContent_GetProperties(vtk_820)
if(NOT vtk_820_POPULATED)
  FetchContent_Populate(vtk_820)
endif()
# Rename FetchContent variables to make some of these more explicit
set(VTK_SOURCE_DIR  ${vtk_820_SOURCE_DIR})
set(VTK_BUILD_DIR   ${vtk_820_BINARY_DIR})
set(VTK_INSTALL_DIR ${VTK_BUILD_DIR})
file(MAKE_DIRECTORY ${VTK_BUILD_DIR})

# Configure VTK
execute_process(
  COMMAND ${CMAKE_COMMAND} -B${VTK_BUILD_DIR} -S${VTK_SOURCE_DIR}
)
# Build VTK
execute_process(
  COMMAND make -C ${VTK_BUILD_DIR} -j8
)

# Download OpenCV-Contrib
FetchContent_Declare(
  opencv_contrib_src
  GIT_REPOSITORY https://github.com/opencv/opencv_contrib.git
  GIT_TAG 4.2.0
  GIT_SHALLOW TRUE
)
FetchContent_GetProperties(opencv_contrib_src)
if(NOT opencv_contrib_src_POPULATED)
  FetchContent_Populate(opencv_contrib_src)
endif()

# Download OpenCV
FetchContent_Declare(
  opencv_src
  GIT_REPOSITORY https://github.com/opencv/opencv.git
  GIT_TAG 4.2.0
  GIT_SHALLOW TRUE
)
FetchContent_GetProperties(opencv_src)
if(NOT opencv_src_POPULATED)
  FetchContent_Populate(opencv_src)
endif()

# Our module location
set(KINFU_SOURCE_PREFIX_DIR     ${CMAKE_CURRENT_SOURCE_DIR}/opencv_contrib)
# OpenCV source locations
set(OPENCV_SOURCE_DIR           ${opencv_src_SOURCE_DIR})
set(OPENCV_CONTRIB_SOURCE_DIR   ${opencv_contrib_src_SOURCE_DIR})
set(OPENCV_BUILD_DIR            ${opencv_src_BINARY_DIR})
set(OPENCV_INSTALL_PREFIX       ${PREFIX_DIR}/install)
set(OPENCV_EXTRA_MODULES_PATHS  "${OPENCV_CONTRIB_SOURCE_DIR}/modules\;${KINFU_SOURCE_PREFIX_DIR}/modules")
file(MAKE_DIRECTORY             ${OPENCV_INSTALL_PREFIX})

# Configure OpenCV
execute_process(
  COMMAND 
    ${CMAKE_COMMAND}
    -B${OPENCV_BUILD_DIR}
    -S${OPENCV_SOURCE_DIR}
    -DOPENCV_EXTRA_MODULES_PATH=${OPENCV_EXTRA_MODULES_PATHS}
    -DBUILD_LIST=rgbdroahm,viz,highgui
    -DWITH_VTK=ON
    -DVTK_DIR=${VTK_BUILD_DIR} -DOPENCV_ENABLE_NONFREE=ON
    -DCMAKE_INSTALL_PREFIX=${OPENCV_INSTALL_PREFIX}
)
# Build OpenCV
execute_process(
  COMMAND make -C ${OPENCV_BUILD_DIR} -j8
)
# Install OpenCV
execute_process(
  COMMAND make -C ${OPENCV_BUILD_DIR} install
)

# Make it so that we can build during the build state too
add_custom_target(kinfu_cv ALL DEPENDS rgbdroahm)
add_custom_command(
  OUTPUT rgbdroahm
  COMMAND make -C ${OPENCV_BUILD_DIR} -j8
  COMMAND make -C ${OPENCV_BUILD_DIR} install
)

# Use findpackage to get stuff
find_package(OpenCV PATHS ${OPENCV_INSTALL_PREFIX} NO_DEFAULT_PATH)
find_package(VTK PATHS ${VTK_INSTALL_DIR} NO_DEFAULT_PATH)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${VTK_INCLUDE_DIRS})

# Tell the catkin_package to include this.
catkin_package(
  INCLUDE_DIRS
    ${OpenCV_INCLUDE_DIRS}
    ${VTK_INCLUDE_DIRS}
  LIBRARIES
    # ${OpenCV_LIBRARIES}
    opencv_rgbdroahm opencv_viz
    ${VTK_LIBRARIES})


