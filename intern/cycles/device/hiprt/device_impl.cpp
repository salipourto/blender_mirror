/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2011-2022 Blender Foundation */

#ifdef WITH_HIPRT

#  include "device/hiprt/device_impl.h"

#  include "util/debug.h"
#  include "util/foreach.h"
#  include "util/log.h"
#  include "util/map.h"
#  include "util/md5.h"
#  include "util/path.h"
#  include "util/string.h"
#  include "util/system.h"
#  include "util/time.h"
#  include "util/types.h"
#  include "util/windows.h"

#  include "bvh/hiprt.h"
#  include "util/progress.h"

#  include "scene/hair.h"
#  include "scene/mesh.h"
#  include "scene/pointcloud.h"
#  include "scene/object.h"

#  include "kernel/device/hiprt/globals.h"


CCL_NAMESPACE_BEGIN


class HIPRTDevice;


BVHLayoutMask HIPRTDevice::get_bvh_layout_mask() const
{
  return BVH_LAYOUT_HIPRT;
}


HIPRTDevice::HIPRTDevice(const DeviceInfo &info, Stats &stats, Profiler &profiler)
    : use_lds(true),
      instance_id_map_(this, "Instance ID Map", MEM_READ_ONLY),
      blender_object_id(this, "__blender_object_id", MEM_GLOBAL),
      visibility(this, "Visibility Mask", MEM_READ_ONLY),
      geometry(this, "HIPRT BLAS", MEM_READ_WRITE),
      blas_ptr(this, "__instance_geometry", MEM_GLOBAL),
      transform_matrix_(this, "Transform Matrix", MEM_READ_ONLY),
      transform_headers_(this, "Transform Header", MEM_READ_ONLY),
      custom_prim_info_offset(this, "__custom_prim_info_offset", MEM_GLOBAL),
      custom_prim_info(this, "__custom_prim_info", MEM_GLOBAL),
      prim_time_offset(this, "__prim_time_offset", MEM_GLOBAL),
      prim_time(this, "__prim_time", MEM_GLOBAL),
      hiprt_context(NULL),
      scene(NULL),
      functions_table(NULL),
      HIPDevice(info, stats, profiler)
{
  hiprtContextCreationInput hiprt_context_input = {0};
  hiprt_context_input.ctxt = hipContext;
  hiprt_context_input.device = hipDevice;
  hiprt_context_input.deviceType = hiprtDeviceAMD;
  hiprtError rt_result = hiprtCreateContext(
      HIPRT_API_VERSION, hiprt_context_input, &hiprt_context);

  if (rt_result != hipSuccess) {
    set_error(string_printf("Failed to create HIPRT context"));
    return;
  }
}

HIPRTDevice::~HIPRTDevice()
{
  instance_id_map_.free();
  blender_object_id.free();
  visibility.free();
  geometry.free();
  blas_ptr.free();
  transform_matrix_.free();
  transform_headers_.free();
  custom_prim_info_offset.free();
  custom_prim_info.free();
  prim_time_offset.free();
  prim_time.free();
  hiprtDestroyFuncTable(hiprt_context, functions_table);
  hiprtDestroyScene(hiprt_context, scene);
  hiprtDestroyContext(hiprt_context);
}

 unique_ptr<DeviceQueue> HIPRTDevice::gpu_queue_create()
{
  return make_unique<HIPRTDeviceQueue>(this);
}

string HIPRTDevice::compile_kernel_get_common_cflags(const uint kernel_features)
{
  string cflags = HIPDevice::compile_kernel_get_common_cflags(kernel_features);

  cflags += " -D __HIPRT__ ";

  if (use_lds)
    cflags += " -D HIPRT_SHARED_STACK ";
  #  ifdef HIPRT_INTERSECTION_FILTERS
  cflags += " -D HIPRT_INTERSECTION_FILTERS ";
  #endif

  return cflags;
}

bool HIPRTDevice::compile_RT_kernel(const string fatbin_rt,
                                    const string include_path,
                                    const string source_path,
                                    hiprtFuncNameSet *func_name_set)
{
  if (!path_exists(fatbin_rt)) {

    vector<const char *> function_names;
    vector<string> function_names_str;

    for (int i = 0; i < (int)DEVICE_KERNEL_NUM; i++) {

      if (i == DEVICE_KERNEL_INTEGRATOR_MEGAKERNEL) {
        continue;
      }

      const string function_name = std::string("kernel_gpu_") +
                                        device_kernel_as_string((DeviceKernel)i);

      function_names_str.push_back(function_name);
    }

    for (int i = 0; i < function_names_str.size(); i++) {

      function_names.push_back(function_names_str[i].c_str());
    }


    vector<const char *> rtc_options;

    const string block_size_str = to_string(HIPRT_THREAD_GROUP_SIZE);
    const string stack_size_str = to_string(HIPRT_SHARED_STACK_SIZE);
    const string global_stack_size_thread = to_string(HIPRT_THREAD_STACK_SIZE);
    const string global_stack_size = to_string(HIPRT_GLOBAL_STACK_SIZE);

    string block_size_def = "-D HIPRT_THREAD_GROUP_SIZE=" + block_size_str;
    string stack_size_def = "-D HIPRT_SHARED_STACK_SIZE=" + stack_size_str;
    string global_stack_size_thread_def = "-D HIPRT_THREAD_STACK_SIZE=" + global_stack_size_thread;
    string global_stack_size_def = "-D HIPRT_GLOBAL_STACK_SIZE=" + global_stack_size;

if (use_lds) {

      rtc_options.push_back(block_size_def.c_str());
      rtc_options.push_back(stack_size_def.c_str());
      rtc_options.push_back(global_stack_size_thread_def.c_str());
      rtc_options.push_back(global_stack_size_def.c_str());

      rtc_options.push_back("-DHIPRT_SHARED_STACK");

    }

    rtc_options.push_back("-D __HIPRT__");
    rtc_options.push_back("-nostdinc");
    rtc_options.push_back("-ffast-math");
    rtc_options.push_back("-mno-cumode");

 #  ifdef HIPRT_INTERSECTION_FILTERS
    rtc_options.push_back("-DHIPRT_INTERSECTION_FILTERS ");
#  endif

    string include_option = "-I" + include_path;
    rtc_options.push_back(include_option.c_str());

    std::string src_txt;
    path_read_text(source_path, src_txt);

    hiprtcProgram intersection = 0;
    vector<uint8_t> intersection_binary;

    hiprtError e = hiprtBuildTraceProgram(hiprt_context,
                                          function_names.size(),
                                          function_names.data(),
                                          src_txt.c_str(),  // source code
                                          0,                // program name, can be null
                                          0,
                                          0,
                                          0,
                                          rtc_options.size(),
                                          rtc_options.data(),
                                          Max_Primitive_Type,
                                          Max_Intersect_Filter_Function,
                                          func_name_set,
                                          &intersection);

    size_t binary_size = 0;
    if (e == 0)
      e = hiprtBuildTraceGetBinary(&intersection, &binary_size, nullptr);

    if (binary_size > 0) {
      intersection_binary.resize(binary_size);
      e = hiprtBuildTraceGetBinary(&intersection, &binary_size, intersection_binary.data());
      if (path_write_binary(fatbin_rt, intersection_binary))
        return true;
    }
    return false;
  }
  return true;
}

bool HIPRTDevice::set_function_table(hiprtFuncNameSet *func_name_set)
{
  const char *filter_functions[] = {
      "opaque_intersection_filter",
      "shadow_intersection_filter",
      "local_intersection_filter",
      "volume_intersection_filter",
  };

  const char *intersect_function[] = {
      "none", "curve_custom_intersect", "motion_triangle_custom_intersect", "point_custom_intersect"};
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
          //triangle primitives dont need a custom intersection function
        func_name_set[table_index].filterFuncName = filter_functions[filter_function];
      else
          //custom primitives for scene_intersect don't need a filter function because the custom intersection
          //function can handle whatever filter function plans to achieve
        func_name_set[table_index].intersectFuncName = intersect_function[prim];
    }
  }

  hiprtFuncDataSet func_data_set;
  hiprtError result = hiprtCreateFuncTable(
      hiprt_context, Max_Primitive_Type, Max_Intersect_Filter_Function, &functions_table);
  if (result == 0)
    result = hiprtSetFuncTable(hiprt_context,
                               functions_table,
                               Max_Primitive_Type,
                               Max_Intersect_Filter_Function,
                               func_data_set);

  return (result == hiprtSuccess);

}

string HIPRTDevice::compile_kernel(const uint kernel_features, const char *name, const char *base)
{

  int major, minor;
  hipDeviceGetAttribute(&major, hipDeviceAttributeComputeCapabilityMajor, hipDevId);
  hipDeviceGetAttribute(&minor, hipDeviceAttributeComputeCapabilityMinor, hipDevId);
  hipDeviceProp_t props;
  hipGetDeviceProperties(&props, hipDevId);

  char *arch = strtok(props.gcnArchName, ":");
  if (arch == NULL) {
    arch = props.gcnArchName;
  }

  hiprtFuncNameSet func_name_sets[Max_Primitive_Type * Max_Intersect_Filter_Function];

  if (!set_function_table(func_name_sets))
    return string();

  if (!use_adaptive_compilation()) {
    const string fatbin = path_get(string_printf("lib/%s_%s.fatbin", name, arch));
    VLOG(1) << "Testing for pre-compiled kernel " << fatbin << ".";
    if (path_exists(fatbin)) {
      VLOG(1) << "Using precompiled kernel.";
      return fatbin;
    }
  }

  string source_path = path_get("source");
  const string source_md5 = path_files_md5_hash(source_path);

  string common_cflags = compile_kernel_get_common_cflags(kernel_features);
  const string kernel_md5 = util_md5_string(source_md5 + common_cflags);
  
  
  const string include_path = source_path;
  const string fatbin_file = string_printf("cycles_%s_%s_%s", name, arch, kernel_md5.c_str());
  const string fatbin = path_cache_get(path_join("kernels", fatbin_file));

  VLOG(1) << "Testing for locally compiled kernel " << fatbin << ".";
  if (path_exists(fatbin)){
    VLOG(1) << "Using locally compiled kernel.";
    return fatbin;
  }

#  ifdef _WIN32
  if (!use_adaptive_compilation() && have_precompiled_kernels()) {
    if (!hipSupportsDevice(hipDevId)) {
      set_error(
          string_printf("HIP backend requires compute capability 10.1 or up, but found %d.%d. "
                        "Your GPU is not supported.",
                        major,
                        minor));
    }
    else {
      set_error(
          string_printf("HIP binary kernel for this graphics card compute "
                        "capability (%d.%d) not found.",
                        major,
                        minor));
    }
    return string();
  }
#  endif

  path_create_directories(fatbin);

  source_path = path_join(path_join(source_path, "kernel"),
                          path_join("device", path_join(base, string_printf("%s.cpp", name))));

  printf("Compiling  %s and caching to %s", source_path.c_str(), fatbin.c_str());

  double starttime = time_dt();

if (!compile_RT_kernel(fatbin, include_path, source_path, func_name_sets)) {
    set_error(
        "HIP RTC kernel compilation failed, "
        "see console for details.");
    return string();
  }

  printf("Kernel compilation finished in %.2lfs.\n", time_dt() - starttime);

  return fatbin;
}

bool HIPRTDevice::load_kernels(const uint kernel_features)
{
  if (hipModule) {
    if (use_adaptive_compilation()) {
      VLOG(1) << "Skipping HIP kernel reload for adaptive compilation, not currently supported.";
    }
    return true;
  }

  if (hipContext == 0)
    return false;

  if (!support_device(kernel_features)) {
    return false;
  }

  /* get kernel */
  const char *kernel_name = "kernel";
  string fatbin = compile_kernel(kernel_features, kernel_name);
  if (fatbin.empty())
    return false;

  /* open module */
  HIPContextScope scope(this);

  string fatbin_data;
  hipError_t result;
  
  if (path_read_text(fatbin, fatbin_data)) {

    result = hipModuleLoadData(&hipModule, fatbin_data.c_str());
  }
  else
    result = hipErrorFileNotFound;

  if (result != hipSuccess)
    set_error(string_printf(
        "Failed to load HIP kernel from '%s' (%s)", fatbin.c_str(), hipewErrorString(result)));

  if (result == hipSuccess) {
    kernels.load(this);
    //reserve_local_memory(kernel_features);
    {
      const DeviceKernel test_kernel = (kernel_features & KERNEL_FEATURE_NODE_RAYTRACE) ?
                                           DEVICE_KERNEL_INTEGRATOR_SHADE_SURFACE_RAYTRACE :
                                       (kernel_features & KERNEL_FEATURE_MNEE) ?
                                           DEVICE_KERNEL_INTEGRATOR_SHADE_SURFACE_MNEE :
                                           DEVICE_KERNEL_INTEGRATOR_SHADE_SURFACE;

      HIPRTDeviceQueue queue(this);

      device_ptr d_path_index = 0;
      device_ptr d_render_buffer = 0;
      int d_work_size = 0;
      DeviceKernelArguments args(&d_path_index, &d_render_buffer, &d_work_size);

      queue.init_execution();
      queue.enqueue(test_kernel, 1, args);
      queue.synchronize();
    }
  }

  return (result == hipSuccess);
}

void HIPRTDevice::const_copy_to(const char *name, void *host, size_t size)
{
  HIPContextScope scope(this);
  hipDeviceptr_t mem;
  size_t bytes;

  if (strcmp(name, "data") == 0) {
    assert(size <= sizeof(KernelData));
    KernelData *const data = (KernelData *)host;
    *(hiprtScene *)&data->device_bvh = scene;
  }


  hip_assert(hipModuleGetGlobal(&mem, &bytes, hipModule, "kernel_params"));
  assert(bytes == sizeof(KernelParamsHIPRT));

#  define KERNEL_DATA_ARRAY(data_type, data_name) \
    if (strcmp(name, #data_name) == 0) { \
        hip_assert(hipMemcpyHtoD(mem + offsetof(KernelParamsHIPRT, data_name), host, size)); \
      return; \
    }
  KERNEL_DATA_ARRAY(KernelData, data)
  KERNEL_DATA_ARRAY(IntegratorStateGPU, integrator_state)
  KERNEL_DATA_ARRAY(int, __blender_object_id)
  KERNEL_DATA_ARRAY(uint64_t, __instance_geometry)
  KERNEL_DATA_ARRAY(int2, __custom_prim_info_offset)
  KERNEL_DATA_ARRAY(int2, __custom_prim_info)
  KERNEL_DATA_ARRAY(int, __prim_time_offset)
  KERNEL_DATA_ARRAY(float2, __prim_time)
#  include "kernel/data_arrays.h"
#  undef KERNEL_DATA_ARRAY

}

hiprtGeometryBuildInput HIPRTDevice::prepare_triangle_blas(BVHHIPRT *bvh, Mesh *mesh)
{

  hiprtGeometryBuildInput geomInput;
  geomInput.geomType = Triangle;

  if (mesh->has_motion_blur() &&
      !(bvh->params.num_motion_triangle_steps == 0 || bvh->params.use_spatial_split)) {

    const Attribute *attr_mP = mesh->attributes.find(ATTR_STD_MOTION_VERTEX_POSITION);
    const size_t num_triangles = mesh->num_triangles();

    const int num_bvh_steps = bvh->params.num_motion_triangle_steps * 2 + 1;
    const float num_bvh_steps_inv_1 = 1.0f / (num_bvh_steps - 1);

     int num_bounds = 0;
    bvh->custom_primitive_bound.alloc(num_triangles * num_bvh_steps);

    for (uint j = 0; j < num_triangles; j++) {
      Mesh::Triangle t = mesh->get_triangle(j);
      const float3 *verts = mesh->get_verts().data();

      const size_t num_verts = mesh->get_verts().size();
      const size_t num_steps = mesh->get_motion_steps();
      const float3 *vert_steps = attr_mP->data_float3();

      float3 prev_verts[3];
      t.motion_verts(verts, vert_steps, num_verts, num_steps, 0.0f, prev_verts);
      BoundBox prev_bounds = BoundBox::empty;
      prev_bounds.grow(prev_verts[0]);
      prev_bounds.grow(prev_verts[1]);
      prev_bounds.grow(prev_verts[2]);

      for (int bvh_step = 1; bvh_step < num_bvh_steps; ++bvh_step) {
        const float curr_time = (float)(bvh_step)*num_bvh_steps_inv_1;
        float3 curr_verts[3];
        t.motion_verts(verts, vert_steps, num_verts, num_steps, curr_time, curr_verts);
        BoundBox curr_bounds = BoundBox::empty;
        curr_bounds.grow(curr_verts[0]);
        curr_bounds.grow(curr_verts[1]);
        curr_bounds.grow(curr_verts[2]);
        BoundBox bounds = prev_bounds;
        bounds.grow(curr_bounds);
        if (bounds.valid()) {
          const float prev_time = (float)(bvh_step - 1) * num_bvh_steps_inv_1;
          bvh->custom_primitive_bound[num_bounds] = bounds;
          bvh->custom_prim_info[num_bounds].x = j; 
          bvh->custom_prim_info[num_bounds].y = mesh->primitive_type();
          bvh->prim_time[num_bounds].x = curr_time;
          bvh->prim_time[num_bounds].y = prev_time;
          num_bounds++;
        }
        prev_bounds = curr_bounds;
      }
    }

    bvh->custom_prim_aabb.aabbCount = bvh->custom_primitive_bound.size();
    bvh->custom_prim_aabb.aabbStride = sizeof(BoundBox);
    bvh->custom_primitive_bound.copy_to_device();
    bvh->custom_prim_aabb.aabbs = (void*)bvh->custom_primitive_bound.device_pointer;



    geomInput.type = hiprtPrimitiveTypeAABBList;
    geomInput.aabbList.primitive = &bvh->custom_prim_aabb;
    geomInput.geomType = Motion_Triangle;
  }
  else {

    size_t triangle_size = mesh->get_triangles().size();
    void *triangle_data = mesh->get_triangles().data();

    size_t vertex_size = mesh->get_verts().size();
    void *vertex_data = mesh->get_verts().data();

    bvh->triangle_mesh.triangleCount = mesh->num_triangles();
    bvh->triangle_mesh.triangleStride = 3 * sizeof(int);
    bvh->triangle_mesh.vertexCount = vertex_size;
    bvh->triangle_mesh.vertexStride = sizeof(float3);

    bvh->triangle_index.host_pointer = triangle_data;
    bvh->triangle_index.data_elements = 1;
    bvh->triangle_index.data_type = TYPE_INT;
    bvh->triangle_index.data_size = triangle_size;
    bvh->triangle_index.copy_to_device();
    bvh->triangle_mesh.triangleIndices = (void *)(bvh->triangle_index.device_pointer);
    // either has to set the host pointer to zero, or increment the refcount on triangle_data
    bvh->triangle_index.host_pointer = 0;
    bvh->vertex_data.host_pointer = vertex_data;
    bvh->vertex_data.data_elements = 4;
    bvh->vertex_data.data_type = TYPE_FLOAT;
    bvh->vertex_data.data_size = vertex_size;
    bvh->vertex_data.copy_to_device();
    bvh->triangle_mesh.vertices = (void *)(bvh->vertex_data.device_pointer);
    bvh->vertex_data.host_pointer = 0;

    geomInput.type = hiprtPrimitiveTypeTriangleMesh;
    geomInput.triangleMesh.primitive = &(bvh->triangle_mesh);
  }
  return geomInput;
}

hiprtGeometryBuildInput HIPRTDevice::prepare_curve_blas(BVHHIPRT *bvh, Hair *hair)
{

  hiprtGeometryBuildInput geomInput;

  const PrimitiveType primitive_type = hair->primitive_type();
  const size_t num_curves = hair->num_curves();
  const size_t num_segments = hair->num_segments();
  const Attribute *curve_attr_mP = NULL;

  if (curve_attr_mP == NULL || bvh->params.num_motion_curve_steps == 0) {

    bvh->custom_prim_info.resize(num_segments);
    bvh->custom_primitive_bound.alloc(num_segments);
  }
  else {
    size_t num_boxes = bvh->params.num_motion_curve_steps * 2* num_segments;
    bvh->custom_prim_info.resize(num_boxes);
    bvh->custom_primitive_bound.alloc(num_boxes);
    curve_attr_mP = hair->attributes.find(ATTR_STD_MOTION_VERTEX_POSITION);
    
  }

  int num_bounds = 0;


  for (uint j = 0; j < num_curves; j++) {
    const Hair::Curve curve = hair->get_curve(j);
    const float *curve_radius = &hair->get_curve_radius()[0];
    for (int k = 0; k < curve.num_keys - 1; k++) {
      if (curve_attr_mP == NULL || bvh->params.num_motion_curve_steps == 0) {
        BoundBox bounds = BoundBox::empty;
        curve.bounds_grow(k, &hair->get_curve_keys()[0], curve_radius, bounds);
        if (bounds.valid()) {
          int type = PRIMITIVE_PACK_SEGMENT(primitive_type, k);
          bvh->custom_prim_info[num_bounds].x = j;
          bvh->custom_prim_info[num_bounds].y = type;  // k;
          bvh->custom_primitive_bound[num_bounds] = bounds;
          num_bounds++;
          }
        }
        else {

        const int num_bvh_steps = bvh->params.num_motion_curve_steps * 2 + 1;
        const float num_bvh_steps_inv_1 = 1.0f / (num_bvh_steps - 1);
        const size_t num_steps = hair->get_motion_steps();
        const float3 *curve_keys = &hair->get_curve_keys()[0];
        const float3 *key_steps = curve_attr_mP->data_float3();
        const size_t num_keys = hair->get_curve_keys().size();

        float4 prev_keys[4];
        curve.cardinal_motion_keys(curve_keys,
                                   curve_radius,
                                   key_steps,
                                   num_keys,
                                   num_steps,
                                   0.0f,
                                   k - 1,
                                   k,
                                   k + 1,
                                   k + 2,
                                   prev_keys);
        BoundBox prev_bounds = BoundBox::empty;
        curve.bounds_grow(prev_keys, prev_bounds);

        for (int bvh_step = 1; bvh_step < num_bvh_steps; ++bvh_step) {
          const float curr_time = (float)(bvh_step)*num_bvh_steps_inv_1;
          float4 curr_keys[4];
          curve.cardinal_motion_keys(curve_keys,
                                     curve_radius,
                                     key_steps,
                                     num_keys,
                                     num_steps,
                                     curr_time,
                                     k - 1,
                                     k,
                                     k + 1,
                                     k + 2,
                                     curr_keys);
          BoundBox curr_bounds = BoundBox::empty;
          curve.bounds_grow(curr_keys, curr_bounds);
          BoundBox bounds = prev_bounds;
          bounds.grow(curr_bounds);
          if (bounds.valid()) {
            const float prev_time = (float)(bvh_step - 1) * num_bvh_steps_inv_1;
            int packed_type = PRIMITIVE_PACK_SEGMENT(primitive_type, k);
            bvh->custom_prim_info[num_bounds].x = j;
            bvh->custom_prim_info[num_bounds].y = packed_type;  // k
            bvh->custom_primitive_bound[num_bounds] = bounds;
            bvh->prim_time[num_bounds].x = curr_time;
            bvh->prim_time[num_bounds].y = prev_time;
            num_bounds++;
          }
          prev_bounds = curr_bounds;
        }
      }
    }
  }

  bvh->custom_prim_aabb.aabbCount = bvh->custom_primitive_bound.size();
  bvh->custom_prim_aabb.aabbStride = sizeof(BoundBox);
  bvh->custom_primitive_bound.copy_to_device();
  bvh->custom_prim_aabb.aabbs = (void *)bvh->custom_primitive_bound.device_pointer;

  geomInput.type = hiprtPrimitiveTypeAABBList;
  geomInput.aabbList.primitive = &bvh->custom_prim_aabb;
  geomInput.geomType = Curve;

  return geomInput;
}

hiprtGeometryBuildInput HIPRTDevice::prepare_point_blas(BVHHIPRT *bvh, PointCloud *pointcloud)
{

  hiprtGeometryBuildInput geomInput;


  const Attribute *point_attr_mP = NULL;
  if (pointcloud->has_motion_blur()) {
    point_attr_mP = pointcloud->attributes.find(ATTR_STD_MOTION_VERTEX_POSITION);
  }

  const float3 *points_data = pointcloud->get_points().data();
  const float *radius_data = pointcloud->get_radius().data();
  const size_t num_points = pointcloud->num_points();
  const float3 *motion_data = (point_attr_mP) ? point_attr_mP->data_float3() : NULL;
  const size_t num_steps = pointcloud->get_motion_steps();

  int num_bounds = 0;

  if (point_attr_mP == NULL) {
    bvh->custom_primitive_bound.alloc(num_points);
    for (uint j = 0; j < num_points; j++) {
      const PointCloud::Point point = pointcloud->get_point(j);
      BoundBox bounds = BoundBox::empty;
      point.bounds_grow(points_data, radius_data, bounds);
      if (bounds.valid()) {
        bvh->custom_primitive_bound[num_bounds] = bounds;
        bvh->custom_prim_info[num_bounds].x = j;
        bvh->custom_prim_info[num_bounds].y = PRIMITIVE_POINT;
        num_bounds++;
      }
    }
  }
  else if (bvh->params.num_motion_point_steps == 0) {

      bvh->custom_primitive_bound.alloc(num_points*num_steps);

    for (uint j = 0; j < num_points; j++) {
      const PointCloud::Point point = pointcloud->get_point(j);
      BoundBox bounds = BoundBox::empty;
      point.bounds_grow(points_data, radius_data, bounds);
      for (size_t step = 0; step < num_steps - 1; step++) {
        point.bounds_grow(motion_data + step * num_points, radius_data, bounds);
      }
      if (bounds.valid()) {
        bvh->custom_primitive_bound[num_bounds] = bounds;
        bvh->custom_prim_info[num_bounds].x = j;
        bvh->custom_prim_info[num_bounds].y = PRIMITIVE_POINT;
        num_bounds++;
      }
    }
  }
  else {

    const int num_bvh_steps = bvh->params.num_motion_point_steps * 2 + 1;
    const float num_bvh_steps_inv_1 = 1.0f / (num_bvh_steps - 1);

    bvh->custom_primitive_bound.alloc(num_points * num_bvh_steps);

    for (uint j = 0; j < num_points; j++) {
      const PointCloud::Point point = pointcloud->get_point(j);
      const size_t num_steps = pointcloud->get_motion_steps();
      const float3 *point_steps = point_attr_mP->data_float3();

      float4 prev_key = point.motion_key(
          points_data, radius_data, point_steps, num_points, num_steps, 0.0f, j);
      BoundBox prev_bounds = BoundBox::empty;
      point.bounds_grow(prev_key, prev_bounds);

      for (int bvh_step = 1; bvh_step < num_bvh_steps; ++bvh_step) {
        const float curr_time = (float)(bvh_step)*num_bvh_steps_inv_1;
        float4 curr_key = point.motion_key(
            points_data, radius_data, point_steps, num_points, num_steps, curr_time, j);
        BoundBox curr_bounds = BoundBox::empty;
        point.bounds_grow(curr_key, curr_bounds);
        BoundBox bounds = prev_bounds;
        bounds.grow(curr_bounds);
        if (bounds.valid()) {
          const float prev_time = (float)(bvh_step - 1) * num_bvh_steps_inv_1;
          bvh->custom_primitive_bound[num_bounds] = bounds;
          bvh->custom_prim_info[num_bounds].x = j;
          bvh->custom_prim_info[num_bounds].y = PRIMITIVE_MOTION_POINT;
          bvh->prim_time[num_bounds].x = curr_time;
          bvh->prim_time[num_bounds].y = prev_time;
          num_bounds++;
        }
        prev_bounds = curr_bounds;
      }
    }
  }

  bvh->custom_prim_aabb.aabbCount = bvh->custom_primitive_bound.size();
  bvh->custom_prim_aabb.aabbStride = sizeof(BoundBox);
  bvh->custom_primitive_bound.copy_to_device();
  bvh->custom_prim_aabb.aabbs = (void *)bvh->custom_primitive_bound.device_pointer;


  geomInput.type = hiprtPrimitiveTypeAABBList;
  geomInput.aabbList.primitive = &bvh->custom_prim_aabb;
  geomInput.geomType = Point;

  return geomInput;
}

hiprtGeometry HIPRTDevice::build_blas(BVHHIPRT *bvh, Geometry *geom, hiprtBuildOptions options)
{
  hiprtGeometry hiprt_geom = NULL;
  hiprtGeometryBuildInput geomInput;

  switch (geom->geometry_type) {
    case Geometry::MESH:
    case Geometry::VOLUME: {
      Mesh *mesh = static_cast<Mesh *>(geom);

      if (mesh->num_triangles() == 0)
        return 0;

      geomInput = prepare_triangle_blas(bvh, mesh);
      break;
    }

    case Geometry::HAIR: {
      Hair *const hair = static_cast<Hair *const>(geom);

      if (hair->num_segments() == 0)
        return 0;

      geomInput = prepare_curve_blas(bvh, hair);
      break;
    }

    case Geometry::POINTCLOUD: {
      PointCloud *pointcloud = static_cast<PointCloud *>(geom);
      if (pointcloud->num_points() == 0)
        return 0;

      geomInput = prepare_point_blas(bvh, pointcloud);
      break;
    }

    default:
      assert(0);
  }

  size_t scratch_buffer_size;

  hiprtError rt_err = hiprtGetGeometryBuildTemporaryBufferSize(
      hiprt_context, &geomInput, &options, &scratch_buffer_size);

  rt_err = hiprtCreateGeometry(hiprt_context, &geomInput, &options, &hiprt_geom);

  device_vector<char> scratch_buffer(this, "scratch buffer", MEM_DEVICE_ONLY);
  scratch_buffer.alloc(scratch_buffer_size);
  scratch_buffer.zero_to_device();

  rt_err = hiprtBuildGeometry(hiprt_context,
                              hiprtBuildOperationBuild,
                              &geomInput,
                              &options,
                              (void*)scratch_buffer.device_pointer,
                              0,
                              hiprt_geom);
  scratch_buffer.free();

  return hiprt_geom;
}
hiprtScene HIPRTDevice::build_tlas(BVHHIPRT *bvh,
                                 vector<Object *> objects,
                                 hiprtBuildOptions options,
                                 bool refit)
{
  hiprtBuildOperation build_operation = refit ? hiprtBuildOperationUpdate :
                                                hiprtBuildOperationBuild;

  array<hiprtFrameMatrix> transform_matrix;

  unordered_map<Geometry *, int2> prim_info_map;
  size_t custom_prim_offset = 0;

  unordered_map<Geometry *, int> prim_time_map;
  

  size_t num_instances = 0;
  int blender_instance_id = 0;

  size_t num_object = objects.size();
  instance_id_map_.alloc(num_object);
  blender_object_id.alloc(num_object);
  visibility.alloc(num_object);
  geometry.alloc(num_object);
  blas_ptr.alloc(num_object);
  transform_headers_.alloc(num_object);
  custom_prim_info_offset.alloc(num_object);
  prim_time_offset.alloc(num_object);

  foreach (Object *ob, objects) {

    UINT32 mask = 0;
    if (ob->is_traceable())
      mask = ob->visibility_for_tracing();

    Transform current_transform = ob->get_tfm();
    Geometry *geom = ob->get_geometry();
    bool transform_applied = geom->transform_applied;


    BVHHIPRT *current_bvh = static_cast<BVHHIPRT *>(geom->bvh);
    hiprtGeometry hiprt_geom_current = (current_bvh == NULL) ?
                                           build_blas(current_bvh, geom, options) :
                                           current_bvh->hiprt_geom;


    hiprtFrameSRT hiprt_transform = {0};
    hiprt_transform.translation = make_hiprtFloat3(0.0f, 0.0f, 0.0f);
    hiprt_transform.scale = make_hiprtFloat3(1.0f, 1.0f, 1.0f);
    hiprt_transform.rotation = make_hiprtFloat4(0.0f, 0.0f, 1.0f, 0.0f);

    hiprtFrameMatrix hiprt_transform_matrix = {0};
    Transform identity_matrix = transform_identity();
    get_hiprt_transform(hiprt_transform_matrix.matrix, identity_matrix);

    if (hiprt_geom_current) {
      bool is_custom_prim = current_bvh->custom_prim_info.size() > 0;

      if (is_custom_prim) {

          bool has_motion_blur = current_bvh->prim_time.size() > 0;

        unordered_map<Geometry *, int2>::iterator it = prim_info_map.find(geom);

        if (prim_info_map.find(geom) != prim_info_map.end()) {

          custom_prim_info_offset[blender_instance_id] = it->second;

          if (has_motion_blur) {

            prim_time_offset[blender_instance_id] = prim_time_map[geom];
          }

        }
        else {
          int offset = bvh->custom_prim_info.size();

          prim_info_map[geom].x = offset;
          prim_info_map[geom].y = custom_prim_offset;

          bvh->custom_prim_info.resize(offset + current_bvh->custom_prim_info.size());
          memcpy(bvh->custom_prim_info.data() + offset,
                 current_bvh->custom_prim_info.data(),
                 current_bvh->custom_prim_info.size() * sizeof(int2));

          custom_prim_info_offset[blender_instance_id].x = offset;
          custom_prim_info_offset[blender_instance_id].y = custom_prim_offset;

          if (geom->geometry_type == Geometry::HAIR) {
            custom_prim_offset += ((Hair *)geom)->num_curves();
          }
          else if (geom->geometry_type == Geometry::POINTCLOUD) {
            custom_prim_offset += ((PointCloud *)geom)->num_points();
          }
          else {
            custom_prim_offset += ((Mesh *)geom)->num_triangles();
          }

          if (has_motion_blur) {
            int time_offset = bvh->prim_time.size();
            prim_time_map[geom] = time_offset;

            memcpy(bvh->prim_time.data() + time_offset,
                   current_bvh->prim_time.data(),
                   current_bvh->prim_time.size() * sizeof(float2));

            prim_time_offset[blender_instance_id] = time_offset;
          }
          else
            prim_time_offset[blender_instance_id] = -1;

        }
      }
      else
        custom_prim_info_offset[blender_instance_id] = {-1, -1};


      hiprtTransformHeader current_header = {0};
      current_header.frameCount = 1;
      current_header.frameIndex = transform_matrix.size();
      if (ob->get_motion().size()) {
        int motion_size = ob->get_motion().size();
        assert(motion_size == 1);

        const int num_bvh_steps = bvh->params.num_motion_triangle_steps;
        const int num_bvh_steps_blas = current_bvh->params.num_motion_curve_steps;

        array<Transform> tfm_array = ob->get_motion();
        float time_iternval = 1 / (float)(motion_size - 1);
        current_header.frameCount = motion_size;

        vector<hiprtFrameMatrix> tfm_hiprt_mb;
        tfm_hiprt_mb.resize(motion_size);
        for (int i = 0; i < motion_size; i++) {
          get_hiprt_transform(tfm_hiprt_mb[i].matrix, tfm_array[i]);
          tfm_hiprt_mb[i].time = (float)i * time_iternval;
          transform_matrix.push_back_slow(tfm_hiprt_mb[i]);
        }
      }
      else {
        if (transform_applied)
          current_transform = identity_matrix;
        get_hiprt_transform(hiprt_transform_matrix.matrix, current_transform);
        transform_matrix.push_back_slow(hiprt_transform_matrix);
      }

      transform_headers_[num_instances] = current_header;

      instance_id_map_[num_instances] = blender_instance_id;
      blender_object_id[num_instances] = blender_instance_id;
      visibility[num_instances] = mask;
      geometry[num_instances] = (uint64_t)hiprt_geom_current;
      num_instances++;
    }
    blas_ptr[blender_instance_id] = (uint64_t)hiprt_geom_current;
    blender_instance_id++;
  }
  int frame_count = transform_matrix.size();
  hiprtSceneBuildInput scene_input_ptr = {0};
  scene_input_ptr.instanceCount = num_instances;
  scene_input_ptr.frameCount = frame_count;
  scene_input_ptr.frameType = hiprtFrameTypeMatrix;
  #ifdef KERNEL_TIME
  printf("Number Instance\t%d\n", (int)num_instances);
  #endif
 
  instance_id_map_.copy_to_device();
  blender_object_id.copy_to_device();
  visibility.copy_to_device();
  geometry.copy_to_device();
  blas_ptr.copy_to_device();
  transform_headers_.copy_to_device();
  {
    transform_matrix_.alloc(frame_count);
    transform_matrix_.host_pointer = transform_matrix.data();
    transform_matrix_.data_elements = sizeof(hiprtFrameMatrix);
    transform_matrix_.data_type = TYPE_UCHAR;
    transform_matrix_.data_size = frame_count;
    transform_matrix_.copy_to_device();
    transform_matrix_.host_pointer = 0;
  }

  scene_input_ptr.instanceMasks = (void *)visibility.device_pointer;
  scene_input_ptr.instanceGeometries = (void *)geometry.device_pointer;
  scene_input_ptr.instanceTransformHeaders = (void *)transform_headers_.device_pointer;
  scene_input_ptr.instanceFrames = (void *)transform_matrix_.device_pointer;


   hiprtScene scene = 0;

  hiprtError rt_err = hiprtCreateScene(hiprt_context, &scene_input_ptr, &options, &scene);

  size_t scratch_buffer_size;
  rt_err = hiprtGetSceneBuildTemporaryBufferSize(
      hiprt_context, &scene_input_ptr, &options, &scratch_buffer_size);


  device_vector<char> scratch_buffer(this, "scratch buffer", MEM_DEVICE_ONLY);
  scratch_buffer.alloc(scratch_buffer_size);
  scratch_buffer.zero_to_device();

  rt_err = hiprtBuildScene(
      hiprt_context, build_operation, &scene_input_ptr, &options, (void *)scratch_buffer.device_pointer, 0, scene);

  scratch_buffer.free();

    if (bvh->custom_prim_info.size()) {
      size_t data_size = bvh->custom_prim_info.size();
      custom_prim_info.alloc(data_size);
      custom_prim_info.host_pointer = bvh->custom_prim_info.data();
      custom_prim_info.data_elements = 2;
      custom_prim_info.data_type = TYPE_INT;
      custom_prim_info.data_size = data_size;
      custom_prim_info.copy_to_device();
      custom_prim_info.host_pointer = 0;

      custom_prim_info_offset.copy_to_device();
    }

    if (bvh->prim_time.size()) {
      size_t data_size = bvh->prim_time.size();
      prim_time.alloc(data_size);
      prim_time.host_pointer = bvh->prim_time.data();
      prim_time.data_elements = 2;
      prim_time.data_type = TYPE_FLOAT;
      prim_time.data_size = data_size;
      prim_time.copy_to_device();
      prim_time.host_pointer = 0;

      prim_time_offset.copy_to_device();
    }

    const char *tables[] = {"__table_closest_intersect",
                            "__table_shadow_intersect",
                            "__table_local_intersect",
                            "__table_volume_intersect"};


    for (int table_index = 0; table_index < Max_Intersect_Filter_Function; table_index++) {

      size_t table_ptr_size = 0;
      device_ptr table_device_ptr;

      hip_assert(hipModuleGetGlobal(
          &table_device_ptr, &table_ptr_size, hipModule, tables[table_index]));
      hip_assert(hipMemcpyHtoD(table_device_ptr, &functions_table, table_ptr_size));
    }

  return scene;
}

void HIPRTDevice::build_bvh(BVH *bvh, Progress &progress, bool refit)
{
  progress.set_substatus("Building HIPRT acceleration structure");

  hiprtBuildOptions options;
  options.buildFlags =  hiprtBuildFlagBitPreferHighQualityBuild;
     //hiprtBuildFlagBitPreferBalancedBuild;


  BVHHIPRT *bvh_rt = static_cast<BVHHIPRT *>(bvh);

  if (!bvh_rt->is_tlas()) {

    vector<Geometry *> geometry = bvh_rt->geometry;
    assert(geometry.size() == 1);
    Geometry *geom = geometry[0];
    bvh_rt->hiprt_geom = build_blas(bvh_rt, geom, options);
  }
  else {

    const vector<Object *> objects = bvh_rt->objects;
    scene = build_tlas(bvh_rt, objects, options, refit);

  }
}

void get_hiprt_transform(float matrix[][4], Transform &tfm)
{
  int row = 0;
  int col = 0;
  matrix[row][col++] = tfm.x.x;
  matrix[row][col++] = tfm.x.y;
  matrix[row][col++] = tfm.x.z;
  matrix[row][col++] = tfm.x.w;
  row++;
  col = 0;
  matrix[row][col++] = tfm.y.x;
  matrix[row][col++] = tfm.y.y;
  matrix[row][col++] = tfm.y.z;
  matrix[row][col++] = tfm.y.w;
  row++;
  col = 0;
  matrix[row][col++] = tfm.z.x;
  matrix[row][col++] = tfm.z.y;
  matrix[row][col++] = tfm.z.z;
  matrix[row][col++] = tfm.z.w;
}

CCL_NAMESPACE_END

#endif

