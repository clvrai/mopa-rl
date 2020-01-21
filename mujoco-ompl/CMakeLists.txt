cmake_minimum_required (VERSION 2.8.5)
project(mujoco-ompl)

# set(CMAKE_CXX_COMPILER clang++)
# set(CMAKE_CXX_FLAGS "-g -Wall -std=c++11 -fPIC")

set( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_LIST_DIR}/cmake )

include(UseCython)

add_custom_target( ReplicatePythonSourceTree ALL ${CMAKE_COMMAND} -P
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/ReplicatePythonSourceTree.cmake
  ${CMAKE_CURRENT_BINARY_DIR}
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} )

include_directories( ${CYTHON_CMAKE_EXAMPLE_SOURCE_DIR}/include )
include_directories( ./include )

enable_testing()
add_subdirectory(src)
