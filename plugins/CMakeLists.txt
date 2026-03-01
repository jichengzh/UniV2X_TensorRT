cmake_minimum_required(VERSION 3.12)

project(tensorrt_ops LANGUAGES CXX CUDA)
find_package(CUDA REQUIRED)

execute_process(COMMAND nvcc -lcuda ${CMAKE_CURRENT_SOURCE_DIR}/cudaComputeVersion.cu -o /tmp/cudaComputeVersion)
execute_process(COMMAND /tmp/cudaComputeVersion
        RESULT_VARIABLE CUDA_RETURN_CODE
        OUTPUT_VARIABLE ARCH)
execute_process(COMMAND rm /tmp/cudaComputeVersion)

if(NOT CMAKE_TENSORRT_PATH)
    message( "Need: -DCMAKE_TENSORRT_PATH=/path/to/TensorRT")

endif()

include_directories(common)
include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include)
include_directories(/usr/local/include)

# cuDNN (may be installed via pip nvidia-cudnn-cu12)
if(NOT CUDNN_INCLUDE_DIR)
    set(CUDNN_INCLUDE_DIR "" CACHE PATH "Path to cuDNN include directory")
endif()
if(NOT "${CUDNN_INCLUDE_DIR}" STREQUAL "")
    include_directories(${CUDNN_INCLUDE_DIR})
    message(STATUS "cuDNN include: ${CUDNN_INCLUDE_DIR}")
endif()

if(NOT CUDNN_LIBRARY_DIR)
    set(CUDNN_LIBRARY_DIR "" CACHE PATH "Path to cuDNN library directory")
endif()
if(NOT "${CUDNN_LIBRARY_DIR}" STREQUAL "")
    link_directories(${CUDNN_LIBRARY_DIR})
    message(STATUS "cuDNN library: ${CUDNN_LIBRARY_DIR}")
endif()

message(STATUS "Linking TensorRT directory: ${CMAKE_TENSORRT_PATH}")
link_directories(${CMAKE_TENSORRT_PATH}/lib)
include_directories(${CMAKE_TENSORRT_PATH}/include)

# Exclude bev_pool_v2: uses LSS-style voxel pooling not needed by UniV2X (BEVFormer-based)
file(GLOB_RECURSE TENSORRT_OPS_SRCS common/*.cpp common/*.cu plugin/*/*.cu plugin/*/*.cpp)
list(FILTER TENSORRT_OPS_SRCS EXCLUDE REGEX ".*/bev_pool_v2/.*")

set(SHARED_TARGET tensorrt_ops)
add_library(${SHARED_TARGET} SHARED ${TENSORRT_OPS_SRCS})
target_compile_options(${SHARED_TARGET} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_${ARCH}>)

target_link_libraries(${SHARED_TARGET} PUBLIC cudnn nvinfer cublas)
set_target_properties(${SHARED_TARGET} PROPERTIES CUDA_ARCHITECTURES "${ARCH}")

install(
        TARGETS ${SHARED_TARGET}
        LIBRARY DESTINATION ${CMAKE_SOURCE_DIR}/lib
)
