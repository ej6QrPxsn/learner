cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(learner)

set(LIBTORCH_DIR $ENV{HOME}/dev/libtorch)
list(APPEND CMAKE_PREFIX_PATH ${LIBTORCH_DIR})

find_package(Torch REQUIRED)
IF (${GTK_FOUND})
   MESSAGE(STATUS "Found Torch.")
ELSE (${GTK_FOUND})
   MESSAGE(STATUS "Could not locate Torch.")
ENDIF (${GTK_FOUND})

enable_language(CXX)
set(CMAKE_CXX_STANDARD 20) # C++20...
set(CMAKE_CXX_STANDARD_REQUIRED ON) #...is required...
set(CMAKE_CXX_EXTENSIONS OFF) #...without compiler extensions like gnu++11

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

file(GLOB learner_SRC
   "src/*.cpp"
)

add_executable(learner ${learner_SRC})
target_include_directories(learner PUBLIC ./include)
target_link_libraries(learner "${TORCH_LIBRARIES}")
