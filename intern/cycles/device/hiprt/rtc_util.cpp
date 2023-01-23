/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2011-2022 Blender Foundation */

#include <string>
#include "device/hiprt/rtc_util.h"
# include "device/kernel.h"

//using namespace hiprt_rtc_helper;

void hiprt_rtc_helper::get_custom_function_names(hiprtFuncNameSet *func_name_set)
{  
  //"motion_triangle_custom_local_intersect", "motion_triangle_custom_volume_intersect"

  for (int filter_function = 0; filter_function < Max_Intersect_Filter_Function;
       filter_function++) {
    for (int prim = 0; prim < Max_Primitive_Type; prim++) {
      int table_index = prim + filter_function * Max_Intersect_Filter_Function;
      if (prim != Triangle && filter_function != Opaque) {
        func_name_set[table_index].filterFuncName = filter_functions[filter_function];
        func_name_set[table_index].intersectFuncName = intersect_function[prim];
      }
      else if (prim == Triangle)
        // triangle primitives dont need a custom intersection function
        func_name_set[table_index].filterFuncName = filter_functions[filter_function];
      else
        // custom primitives for scene_intersect don't need a filter function because the custom
        // intersection function can handle whatever filter function plans to achieve
        func_name_set[table_index].intersectFuncName = intersect_function[prim];
    }
  }
}

void hiprt_rtc_helper::get_kernel_names(
    std::vector<std::string> &function_names_str, std::vector<const char *> &function_names)
{

  for (int i = 0; i < (int)ccl::DEVICE_KERNEL_NUM; i++) {

    if (i == ccl::DEVICE_KERNEL_INTEGRATOR_MEGAKERNEL) {
      continue;
    }

    const std::string function_name = std::string("kernel_gpu_") +
                                      ccl::device_kernel_as_string((ccl::DeviceKernel)i);

    function_names_str.push_back(function_name);
  }

  for (int i = 0; i < function_names_str.size(); i++) {

    function_names.push_back(function_names_str[i].c_str());
  }
}

void hiprt_rtc_helper::get_compiler_options(std::vector<const char *> &rtc_options,
                                            shared_stack_property &stack_property, bool use_lds)
{
  const std::string block_size_str = std::to_string(HIPRT_THREAD_GROUP_SIZE);
  const std::string stack_size_str = std::to_string(HIPRT_SHARED_STACK_SIZE);
  const std::string global_stack_size_thread = std::to_string(HIPRT_THREAD_STACK_SIZE);
  const std::string global_stack_size = std::to_string(HIPRT_GLOBAL_STACK_SIZE);

  stack_property.block_size_def = "-D HIPRT_THREAD_GROUP_SIZE=" + block_size_str;
  stack_property.stack_size_def = "-D HIPRT_SHARED_STACK_SIZE=" + stack_size_str;
  stack_property.global_stack_size_thread_def = "-D HIPRT_THREAD_STACK_SIZE=" +
                                             global_stack_size_thread;
  stack_property.global_stack_size_def = "-D HIPRT_GLOBAL_STACK_SIZE=" + global_stack_size;

  if (use_lds) {

    rtc_options.push_back(stack_property.block_size_def.c_str());
    rtc_options.push_back(stack_property.stack_size_def.c_str());
    rtc_options.push_back(stack_property.global_stack_size_thread_def.c_str());
    rtc_options.push_back(stack_property.global_stack_size_def.c_str());

    rtc_options.push_back("-DHIPRT_SHARED_STACK");
  }

  rtc_options.push_back("-D __HIPRT__");
  rtc_options.push_back("-nostdinc");
  rtc_options.push_back("-ffast-math");
  rtc_options.push_back("-mno-cumode");

#ifdef HIPRT_INTERSECTION_FILTERS
  rtc_options.push_back("-DHIPRT_INTERSECTION_FILTERS ");
#endif
}