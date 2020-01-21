#find_package(Eigen3 REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
if(NOT OMPL_INCLUDE_DIRS)
	find_package(ompl REQUIRED)
endif()

find_package(Threads REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Boost 1.58 QUIET REQUIRED COMPONENTS serialization filesystem system program_options)

add_library(mujoco-nogl SHARED IMPORTED)
add_library(mujoco SHARED IMPORTED)
add_library(glfw3 SHARED IMPORTED)

include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${OPENGL_INCLUDE_DIR} ${GLEW_INCLUDE_DIRS})

if(APPLE)
	set_target_properties(mujoco-nogl PROPERTIES
	  IMPORTED_LOCATION "$ENV{HOME}/.mujoco/mujoco200/bin/libmujoco200nogl.dylib"
	  INTERFACE_INCLUDE_DIRECTORIES "$ENV{HOME}/.mujoco/mujoco200/include" )
	
	set_target_properties(mujoco PROPERTIES
	  IMPORTED_LOCATION "$ENV{HOME}/.mujoco/mujoco200/bin/libmujoco200.dylib"
	  INTERFACE_INCLUDE_DIRECTORIES "$ENV{HOME}/.mujoco/mujoco200/include" )
	
	set_target_properties(glfw3 PROPERTIES
	  IMPORTED_LOCATION "$ENV{HOME}/.mujoco/mujoco200/bin/libglfw.3.dylib"
	  INTERFACE_INCLUDE_DIRECTORIES "$ENV{HOME}/.mujoco/mujoco200/include" )	
elseif(UNIX)
	set_target_properties(mujoco-nogl PROPERTIES
	  IMPORTED_LOCATION "$ENV{HOME}/.mujoco/mujoco200/bin/libmujoco200nogl.so"
	  INTERFACE_INCLUDE_DIRECTORIES "$ENV{HOME}/.mujoco/mujoco200/include" )

	set_target_properties(mujoco PROPERTIES
	  IMPORTED_LOCATION "$ENV{HOME}/.mujoco/mujoco200/bin/libmujoco200.so"
	  INTERFACE_INCLUDE_DIRECTORIES "$ENV{HOME}/.mujoco/mujoco200/include" )

	set_target_properties(glfw3 PROPERTIES
	  IMPORTED_LOCATION "$ENV{HOME}/.mujoco/mujoco200/bin/libglfw.so.3"
	  INTERFACE_INCLUDE_DIRECTORIES "$ENV{HOME}/.mujoco/mujoco200/include" )
endif()

include_directories(${OMPL_INCLUDE_DIRS}, ${Boost_INCLUDE_DIR})
include_directories(3rd_party/include)
#include_directories(include)

set_source_files_properties(
  ${CYTHON_CMAKE_EXAMPLE_SOURCE_DIR}/src/plan.pyx
  PROPERTIES CYTHON_IS_CXX TRUE )

# Multi-file cython modules do not appear to be working at the moment.
cython_add_module( plan plan.pyx Planner.cxx )

target_link_libraries(plan ${OMPL_LIBRARIES})
target_link_libraries(plan Eigen3::Eigen)
target_link_libraries(plan mujoco-nogl)
target_link_libraries(plan Threads::Threads)
target_link_libraries(plan yaml-cpp)
target_link_libraries(plan Boost::filesystem)


# add_library(mujoco_ompl SHARED
# 		mujoco_wrapper.cpp
# 		mujoco_ompl_interface.cpp)
# target_include_directories(mujoco_ompl PUBLIC include)
# target_link_libraries(mujoco_ompl ${OMPL_LIBRARIES})
# target_link_libraries(mujoco_ompl Eigen3::Eigen)
# target_link_libraries(mujoco_ompl mujoco)
# target_link_libraries(mujoco_ompl Threads::Threads)
# target_link_libraries(mujoco_ompl yaml-cpp)
# target_link_libraries(mujoco_ompl glfw3)
# target_link_libraries(mujoco_ompl OpenGL::GL)
# target_link_libraries(mujoco_ompl GLEW::GLEW)
