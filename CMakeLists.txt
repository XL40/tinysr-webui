cmake_minimum_required(VERSION 3.14)
project(tinysr-webui CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
if(MSVC)
  # /utf-8: 源码含中文注释; NOMINMAX: windows.h 的 min/max 宏会砸坏 std::min
  # /MP: 文件级并行编译 —— MSBuild 的 /m 只并行项目级, ncnn 单项目无收益,
  #      不加 /MP 的话几百个 .cpp 串行编译, CI 上要多花 10 分钟以上
  add_compile_options(/utf-8 /bigobj /MP)
  add_compile_definitions(WIN32_LEAN_AND_MEAN NOMINMAX _CRT_SECURE_NO_WARNINGS)
  # 静态运行时: 用户机器免装 VC Redist (ncnn 源码构建会继承此设置)
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

# ---- ncnn (vulkan fp16 GPU / cpu) ----
# 两种来源, 任选其一:
#   1) tools/setup_third_party.sh 拉取的源码 (third_party/ncnn, 与开发验证的 commit 一致)
#   2) find_package: -Dncnn_DIR=<ncnn 安装/预编译包路径> (需含 Vulkan 后端)
option(TINYSR_VULKAN "build with Vulkan GPU backend" ON)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ncnn/CMakeLists.txt)
  set(NCNN_VULKAN ${TINYSR_VULKAN} CACHE BOOL "" FORCE)
  set(NCNN_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
  set(NCNN_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(NCNN_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(NCNN_BUILD_BENCHMARK OFF CACHE BOOL "" FORCE)
  set(NCNN_INSTALL_PROTOBUF OFF CACHE BOOL "" FORCE)
  set(NCNN_SYSTEM_GLSLANG OFF CACHE BOOL "" FORCE)  # ncnn 自带 glslang, CI 不用装 shaderc
  add_subdirectory(third_party/ncnn)
  set(TINYSR_NCNN_SOURCE ON)
else()
  find_package(ncnn CONFIG QUIET)
  if(NOT ncnn_FOUND)
    message(FATAL_ERROR
      "ncnn 未找到: 先运行 tools/setup_third_party.sh 拉取源码, "
      "或用 -Dncnn_DIR=<路径> 指向含 ncnn.cmake 的预编译包 (需 Vulkan 后端)")
  endif()
  if(NCNN_VULKAN AND NOT ncnn_VULKAN)
    message(FATAL_ERROR "找到的 ncnn 预编译包不含 Vulkan 后端, 请换 vulkan 版包或 -DTINYSR_VULKAN=OFF")
  endif()
  set(TINYSR_NCNN_SOURCE OFF)
endif()

find_package(Threads REQUIRED)

add_executable(tinysr-webui
  src/main.cpp
  src/engine.cpp
)
target_include_directories(tinysr-webui PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/include)
if(TINYSR_NCNN_SOURCE)
  # 源码构建: net.h / gpu.h 在源码树, platform.h 在二进制目录
  target_include_directories(tinysr-webui PRIVATE
    third_party/ncnn/src
    ${CMAKE_CURRENT_BINARY_DIR}/third_party/ncnn/src)
endif()
target_link_libraries(tinysr-webui PRIVATE ncnn Threads::Threads)
if(WIN32)
  target_link_libraries(tinysr-webui PRIVATE ws2_32)  # httplib
endif()

# 前端单文件拷到构建目录, server 启动时读取
add_custom_command(TARGET tinysr-webui POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory
  ${CMAKE_CURRENT_SOURCE_DIR}/web $<TARGET_FILE_DIR:tinysr-webui>/web)
