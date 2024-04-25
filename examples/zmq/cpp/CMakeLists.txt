cmake_minimum_required(VERSION 3.10)

project(zmq)

#link_directories(/usr/lib/aarch64-linux-gnu/)
#set (cppzmq_DIR /home/radxa/vcpkg/installed/arm64-linux/share/cppzmq/ )
#set (ZeroMQ_DIR /home/radxa/vcpkg/installed/arm64-linux/share/zeromq/ )

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdparty/ 3rdparty.out)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../../utils/ utils.out)

#opencv
if(CMAKE_SIZEOF_VOID_P EQUAL 8)
message(STATUS "64bit")
set (TARGET_LIB_ARCH lib64)
else()
message(STATUS "32bit")
set (TARGET_LIB_ARCH lib)
endif()

# CPPZMQ
find_package(cppzmq REQUIRED)

find_package(OpenCV REQUIRED)
message(STATUS OpenCV_DIR=${OpenCV_DIR})
message(STATUS OpenCV_LIBS=${OpenCV_LIBS})

set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib")

file(GLOB SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)

#dma
#include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdparty/allocator/dma)
#drm
#include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdparty/allocator/drm)


#set(postprocess_file postprocess.cc)
#set(yolov5_seg_file rknpu2/yolov5_seg.cc)


add_executable(${PROJECT_NAME}
    main.cc
    postprocess.cc
    rknpu2/yolov5.cc
)


target_link_libraries(${PROJECT_NAME}
    fileutils
    imageutils
    imagedrawing
    cppzmq
    cppzmq-static    
    ${OpenCV_LIBS}    
    ${LIBRKNNRT}
)

if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(THREADS_PREFER_PTHREAD_FLAG ON)
    find_package(Threads REQUIRED)
    target_link_libraries(${PROJECT_NAME} Threads::Threads)
endif()

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${LIBRKNNRT_INCLUDES}
)

install(TARGETS ${PROJECT_NAME} DESTINATION .)
install(FILES ${LIBRKNNRT} DESTINATION lib)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../model/coco_80_labels_list.txt DESTINATION model)

file(GLOB RKNN_FILES "${CMAKE_CURRENT_SOURCE_DIR}/../model/*.rknn")
install(FILES ${RKNN_FILES} DESTINATION model)
