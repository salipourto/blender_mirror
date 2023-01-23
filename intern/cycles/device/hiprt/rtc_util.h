/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2011-2022 Blender Foundation */

#include <vector>
#include <string>
#include "hiprt.h"

#define HIPRT_INTERSECTION_FILTERS
//#  define KERNEL_TIME

#define HIPRT_GLOBAL_STACK_SIZE 512 * 1024 * 1024
#define HIPRT_SHARED_STACK_SIZE 24  // LDS allocation for each thread
#define HIPRT_THREAD_STACK_SIZE 64  // global stack allocation per thread
#define HIPRT_THREAD_GROUP_SIZE \
  256  // total locaal stack size would be number of threads * HIPRT_SHARED_STACK_SIZE

namespace hiprt_rtc_helper
{

enum Filter_Function { Opaque = 0, Shadows, SSR, Volume, Max_Intersect_Filter_Function };

enum Primitive_Type { Triangle = 0, Curve, Motion_Triangle, Point, Max_Primitive_Type };

static const char *filter_functions[] = {
    "opaque_intersection_filter",
    "shadow_intersection_filter",
    "local_intersection_filter",
    "volume_intersection_filter",
};

static const char *intersect_function[] = {"none",
                                    "curve_custom_intersect",
                                    "motion_triangle_custom_intersect",
                                    "point_custom_intersect"};

struct shared_stack_property {
  std::string block_size_def;
  std::string stack_size_def;
  std::string global_stack_size_thread_def;
  std::string global_stack_size_def;
};

void get_kernel_names(
    std::vector<std::string> &function_names_str, std::vector<const char *> &function_names);
void get_custom_function_names(hiprtFuncNameSet *func_name_set);
void get_compiler_options(std::vector<const char *> &rtc_options,
                          shared_stack_property &stack_property, bool use_lds);

}  // namespace hiprt_rtc_helper
