cmake_minimum_required(VERSION 3.16)
project(RegistrationUDT)

set(EXAMPLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})

FIND_PACKAGE(ITK 5.0.1 REQUIRED COMPONENTS)
if ( ITK_FOUND )
  message(STATUS "ITK found.")
  include_directories( ${ITK_INCLUDE_DIRS} )
  link_directories( ${ITK_LIBRARY_DIRS} )
endif( ITK_FOUND )


include_directories(${CMAKE_SOURCE_DIR}/include/)
option(BUILD_EXAMPLES "Build examples programs" ON)
# option(WITH_OMP "Parallelization with OpenMP" OFF)

# if (WITH_OMP)
#   find_package(OpenMP)
#   if (OPENMP_FOUND)
#     set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#     set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#   endif()
# endif()



if (BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()


message(STATUS ${CMAKE_CXX_FLAGS})
