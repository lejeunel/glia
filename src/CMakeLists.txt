cmake_minimum_required(VERSION 2.8)

# For rtags
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set( CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} )
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

option(GLIA_3D "Process 3D images other than 2D." OFF)
option(GLIA_MT "Use multi-threading." OFF)
option(GLIA_BUILD_HMT "Build HMT/CCHMT module." ON)
option(GLIA_BUILD_SSHMT "Build SSHMT module. (Requires HMT module.)" ON)
option(GLIA_BUILD_LINK3D "Build 3D linking module. (Requires GLIA_3D=OFF.)" ON)
option(GLIA_BUILD_GADGET "Build gadget module." ON)
option(GLIA_BUILD_ML_RF "Build random forest module. (Requires 3rd party random forest code in place.)" OFF)

if(GLIA_BUILD_ML_RF)
  set(RF_SRC_DIR "" CACHE FILEPATH "Random forest source code directory")
endif(GLIA_BUILD_ML_RF)

if(GLIA_3D)
  add_definitions(-DGLIA_3D)
endif(GLIA_3D)

if(GLIA_MT)
  find_package(OpenMP)
  if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    add_definitions(-DGLIA_MT)
  endif(OPENMP_FOUND)
endif(GLIA_MT)

include_directories(.)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(ITK 5.0 REQUIRED)
include(${ITK_USE_FILE})

find_package( Boost 1.58.0 COMPONENTS python3 numpy3 REQUIRED )
find_package(PythonLibs 3 REQUIRED)
find_package (Eigen3 3.3 REQUIRED)
find_package(Shogun REQUIRED NO_MODULE)
INCLUDE_DIRECTORIES("${Boost_INCLUDE_DIRS}" ${PYTHON_INCLUDE_DIRS})
INCLUDE_DIRECTORIES("${shogun_INCLUDE_DIR}")
include_directories ( ${EIGEN3_INCLUDE_DIR} )

option(BOOST_MULTITHREADED "Linking against multi-threaded Boost libraries" OFF)
set(BOOST_PROGRAM_OPTIONS_LIB "boost_program_options")
mark_as_advanced(BOOST_MULTITHREADED)
if(BOOST_MULTITHREADED)
  set(BOOST_PROGRAM_OPTIONS_LIB "boost_program_options-mt")
endif(BOOST_MULTITHREADED)

link_directories(${Boost_LIBRARY_DIR})

add_library(glia SHARED pyglia.hxx pyglia.cxx )
target_link_libraries(glia ${ITK_LIBRARIES} ${PYTHON_LIBRARIES} ${Boost_LIBRARIES} shogun)


set(SETUP_PY_IN "${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in")
set(SETUP_PY    "${PROJECT_SOURCE_DIR}/setup.py")
set(DEPS      glia)
set(OUTPUT      "${CMAKE_CURRENT_BINARY_DIR}/build")

configure_file(${SETUP_PY_IN} ${SETUP_PY})
add_custom_command(OUTPUT ${OUTPUT}
		COMMAND ${CMAKE_COMMAND} -E copy
			"${CMAKE_CURRENT_SOURCE_DIR}/libglia.so"
			"${PROJECT_SOURCE_DIR}/glia/libglia.so"
	       DEPENDS ${DEPS})

add_custom_target(target ALL DEPENDS ${OUTPUT})

install(CODE "execute_process(COMMAND pip install -e ${CMAKE_CURRENT_SOURCE_DIR})")
