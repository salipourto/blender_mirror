# SPDX-License-Identifier: GPL-2.0-or-later



#include(CMakePrintHelpers)

set(SRC_KERNEL
      ${CMAKE_SOURCE_DIR}/intern/cycles/kernel/device/hiprt/kernel.cpp
   )

set (HIPRTSDK $ENV{HIPRT_SDK_DIR})
set(INC_APP
	../../../../../extern/hipew
	${HIPRTSDK}
    ${CMAKE_SOURCE_DIR}/intern/cycles/
	)

  set(SRC_APP
      ${CMAKE_SOURCE_DIR}/intern/cycles/device/hiprt/shader_generator/hiprtc_shader_cache.cpp
  )


set(LIB_APP
	${HIPRTSDK}
	${CMAKE_BINARY_DIR}/lib
	)

set(DEPEND_DIR_APP
    ${CMAKE_SOURCE_DIR}/intern/cycles/device
)

set(hiprt_sources device/hiprt/kernel.cpp
    ${SRC_KERNEL_HEADERS}
    ${SRC_KERNEL_DEVICE_GPU_HEADERS}
    ${SRC_KERNEL_DEVICE_HIPRT_HEADERS}
    ${SRC_UTIL_HEADERS}
  )

add_definitions(-DWITH_HIP)
add_definitions(-DWITH_HIPRT)

set(WITH_SHARED_MEMORY_STACK TRUE)

if(WITH_SHARED_MEMORY_STACK)
add_definitions(-DHIPRT_SHARED_STACK)
endif()

list(APPEND CMAKE_PREFIX_PATH ${HIPRTSDK})

include_directories(hiprtc_shader_cache ${INC_APP})
link_directories(hiprtc_shader_cache ${LIB_APP})
add_executable(hiprtc_shader_cache ${SRC_APP})


add_dependencies(hiprtc_shader_cache cycles_device)

target_link_libraries(hiprtc_shader_cache ${HIPRTSDK}/hiprt64.lib)
target_link_libraries(hiprtc_shader_cache ${CMAKE_BINARY_DIR}/lib/$<CONFIG>/cycles_device.lib)


set (INCLUDE_PATH
${CMAKE_CURRENT_SOURCE_DIR}/../../..
)

set(NANO_INCLUDE_PATH)

if (WITH_NANOVDB)
set(NANO_INCLUDE_PATH
${NANOVDB_INCLUDE_DIR})
endif()

cmake_print_variables(${INCLUDE_PATH})

add_custom_command(
    TARGET hiprtc_shader_cache POST_BUILD DEPENDS ${hiprt_sources}
    COMMAND hiprtc_shader_cache --source_path ${SRC_KERNEL} --include_path ${INCLUDE_PATH} --nano_include ${NANO_INCLUDE_PATH}  --output_path ${CMAKE_BINARY_DIR}
)
