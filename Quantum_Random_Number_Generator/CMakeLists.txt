cmake_minimum_required(VERSION 3.14)
project(QRNG)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add library
add_library(qrng
    src/qrng.cpp
)

# Add include directories
target_include_directories(qrng
    PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Add executables
add_executable(qrng_app src/main.cpp)
target_link_libraries(qrng_app PRIVATE qrng)

# Add algorithm comparison tool
add_executable(compare_algorithms src/compare_algorithms.cpp)
target_link_libraries(compare_algorithms PRIVATE qrng)

# Installation
install(TARGETS qrng
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(TARGETS qrng_app
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/ DESTINATION include)

# REMOVE OR COMMENT OUT THIS LINE IF IT EXISTS:
# add_subdirectory(tests)