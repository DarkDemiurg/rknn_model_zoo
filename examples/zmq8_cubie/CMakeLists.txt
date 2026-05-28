cmake_minimum_required(VERSION 2.8)

project(zmq8_cubie)

set(CMAKE_CXX_STANDARD 14)

# ============================================================================
# Платформа
# ============================================================================
if(EXTERN_DEFINE_TARGET)
    message(STATUS "USE EXTERN DEFINE TARGET: ${TARGET_NAME}")
endif()

string(TOUPPER "${TARGET_NAME}" TARGET_UPPER)
string(TOLOWER "${TARGET_NAME}" TARGET_LOWER)

if(NATIVE_BUILD)
    # ==========================================================================
    # Нативная сборка на плате
    # ==========================================================================
    message(STATUS "=== NATIVE BUILD on target ===")

    # OpenCV — системный
    find_package(OpenCV REQUIRED)

    # ZMQ — системный или из 3rdparty
    find_library(ZMQ_LIB NAMES zmq
        PATHS ${CMAKE_SOURCE_DIR}/3rdparty/zmq/lib /usr/lib /usr/local/lib
    )
    find_path(ZMQ_INCLUDE zmq.hpp
        PATHS ${CMAKE_SOURCE_DIR}/3rdparty/zmq/include /usr/include /usr/local/include
    )

    # NPU Runtime — системные библиотеки или из 3rdparty/npu_lib/
    find_library(NPU_LIB_HAL NAMES VIPhal
        PATHS /usr/lib /usr/local/lib ${CMAKE_SOURCE_DIR}/3rdparty/npu_lib NO_DEFAULT_PATH)
    find_library(NPU_LIB_LINKER NAMES NBGlinker
        PATHS /usr/lib /usr/local/lib ${CMAKE_SOURCE_DIR}/3rdparty/npu_lib NO_DEFAULT_PATH)
    set(NPU_RT_LIBS ${NPU_LIB_HAL} ${NPU_LIB_LINKER})

    # NPU runtime source (npulib.cpp, npu_util.cpp) — лежат в npu_src/
    set(NPU_SRC_INCLUDE ${CMAKE_SOURCE_DIR}/npu_src)
    set(NPU_SOURCES
        ${CMAKE_SOURCE_DIR}/npu_src/npulib.cpp
        ${CMAKE_SOURCE_DIR}/npu_src/npu_util.cpp
    )

    # G2D header — в npu_src/
    set(G2D_INCLUDE ${CMAKE_SOURCE_DIR}/npu_src)

    set(SOURCES
        main.cpp
        npu_yolov8.cpp
        postprocess.cpp
        g2d_resize.cpp
        ${NPU_SOURCES}
    )

    include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${NPU_SRC_INCLUDE}
        ${NPU_SRC_INCLUDE}/include
        ${G2D_INCLUDE}
        ${OpenCV_INCLUDE_DIRS}
        ${ZMQ_INCLUDE}
    )

else()
    # ==========================================================================
    # Кросс-компиляция на хосте
    # ==========================================================================
    message(STATUS "=== CROSS BUILD on host ===")

    set(MODEL_ZOO_HOME_DIR ${CMAKE_SOURCE_DIR}/../../../../cubie/awnpu_model_zoo-v1.0.0-20260423-f562dd16)

    if(NOT USE_EXTERN_TOOLCHAIN)
        set(CMAKE_C_COMPILER ${MODEL_ZOO_HOME_DIR}/0-toolchains/gcc-arm-10.3-2021.07-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-gcc)
        set(CMAKE_CXX_COMPILER ${MODEL_ZOO_HOME_DIR}/0-toolchains/gcc-arm-10.3-2021.07-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-g++)
    endif()

    add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)

    # OpenCV
    set(OpenCV_DIR ${MODEL_ZOO_HOME_DIR}/3rdparty/opencv/opencv-4.9.0-aarch64-linux-sunxi-glibc/lib/cmake/opencv4)
    find_package(OpenCV REQUIRED)

    # ZMQ
    set(ZMQ_DIR ${CMAKE_SOURCE_DIR}/3rdparty/zmq)
    find_library(ZMQ_LIB NAMES zmq PATHS ${ZMQ_DIR}/lib NO_DEFAULT_PATH)
    set(ZMQ_INCLUDE ${ZMQ_DIR}/include)

    # NPU Runtime
    set(NPU_RT_DIR ${MODEL_ZOO_HOME_DIR}/common/npuruntime)
    file(GLOB NPU_RT_LIBS "${NPU_RT_DIR}/lib_linux_aarch64/${TARGET_UPPER}/*.so*")

    # G2D header
    set(G2D_INCLUDE ${CMAKE_SOURCE_DIR}/npu_src)

    set(SOURCES
        main.cpp
        npu_yolov8.cpp
        postprocess.cpp
        g2d_resize.cpp
        ${CMAKE_SOURCE_DIR}/npu_src/npulib.cpp
        ${CMAKE_SOURCE_DIR}/npu_src/npu_util.cpp
    )

    include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/npu_src
        ${CMAKE_SOURCE_DIR}/npu_src/include
        ${G2D_INCLUDE}
        ${OpenCV_INCLUDE_DIRS}
        ${ZMQ_INCLUDE}
    )
endif()

message(STATUS "CMAKE_C_COMPILER = ${CMAKE_C_COMPILER}")
message(STATUS "ZMQ_LIB = ${ZMQ_LIB}")
message(STATUS "NPU_RT_LIBS = ${NPU_RT_LIBS}")
message(STATUS "OpenCV_DIR = ${OpenCV_DIR}")

# ============================================================================
# Target
# ============================================================================
add_executable(${PROJECT_NAME} ${SOURCES})

target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    ${NPU_RT_LIBS}
    ${ZMQ_LIB}
    -lm -ldl -lpthread
)

# ============================================================================
# Install
# ============================================================================
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install/${PROJECT_NAME}_linux_${TARGET_LOWER})
message(STATUS "CMAKE_INSTALL_PREFIX = ${CMAKE_INSTALL_PREFIX}")

install(TARGETS ${PROJECT_NAME} DESTINATION ./)

if(NOT NATIVE_BUILD)
    install(FILES ${NPU_RT_LIBS} DESTINATION ./lib/)
    file(GLOB ZMQ_SO_FILES "${ZMQ_DIR}/lib/libzmq.so*")
    install(FILES ${ZMQ_SO_FILES} DESTINATION ./lib/)
endif()

# Модель и тестовые данные
file(GLOB MODEL_FILES "${CMAKE_SOURCE_DIR}/model/*.nb")
file(GLOB TEST_IMAGES "${CMAKE_SOURCE_DIR}/model/*.jpg")
file(GLOB LABEL_FILES "${CMAKE_SOURCE_DIR}/model/*.txt")
install(FILES ${MODEL_FILES} DESTINATION ./model/)
install(FILES ${TEST_IMAGES} DESTINATION ./model/)
install(FILES ${LABEL_FILES} DESTINATION ./model/)
