/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2011-2022 Blender Foundation */

#pragma once

#include "kernel/types.h"

#include "kernel/integrator/state.h"

#include "kernel/util/profiling.h"

CCL_NAMESPACE_BEGIN

struct KernelGlobalsGPU {
  int unused[1];
#if (defined(__HIPCC_RTC__) || defined(__OFFLINE_COMPILER__))
#  ifdef HIPRT_SHARED_STACK
  int *shared_stack;
#  endif
#endif
};
typedef ccl_global KernelGlobalsGPU *ccl_restrict KernelGlobals;
#if (defined(__HIPCC_RTC__) || defined(__OFFLINE_COMPILER__))
#  ifdef HIPRT_SHARED_STACK
typedef hiprtGlobalStack Stack;
#  endif
#endif

#ifdef HIPRT_SHARED_STACK
#  define SET_SHARED_MEMORY() \
    ccl_gpu_shared int shared_stack[SHARED_STACK_SIZE * BLOCK_SIZE]; \
    ccl_global KernelGlobalsGPU kg_gpu; \
    KernelGlobals kg = &kg_gpu; \
    kg->shared_stack = &shared_stack[0];
#else
#  define SET_SHARED_MEMORY() KernelGlobals kg = NULL;
#endif
struct KernelParamsHIPRT {
  KernelData data;
#define KERNEL_DATA_ARRAY(type, name) const type *name;
  KERNEL_DATA_ARRAY(int, __blender_object_id)
  KERNEL_DATA_ARRAY(uint64_t, __instance_geometry)
  KERNEL_DATA_ARRAY(int2, __curve_intersect_data_offset)
  KERNEL_DATA_ARRAY(int2, __curve_intersect_data)
#include "kernel/data_arrays.h"

  /* Integrator state */
  IntegratorStateGPU integrator_state;
};

#ifdef __KERNEL_GPU__
__constant__ KernelParamsHIPRT kernel_params;
__attribute__((device)) int global_stack_buffer[1024 * 1024 * 512];
__attribute__((used)) __attribute__((constant)) __attribute__((device))
hiprtCustomFuncTable __table_closest_intersect;
__attribute__((used)) __attribute__((constant)) __attribute__((device))
hiprtCustomFuncTable __table_shadow_intersect;
__attribute__((used)) __attribute__((constant)) __attribute__((device))
hiprtCustomFuncTable __table_local_intersect;
__attribute__((used)) __attribute__((constant)) __attribute__((device))
hiprtCustomFuncTable __table_volume_intersect;
#endif

/* Abstraction macros */
#define kernel_data kernel_params.data
#define kernel_data_fetch(name, index) kernel_params.name[(index)]
#define kernel_data_array(name) (kernel_params.name)
#define kernel_integrator_state kernel_params.integrator_state

CCL_NAMESPACE_END
