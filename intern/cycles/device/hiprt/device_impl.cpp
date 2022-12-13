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
    : instance_id_map_(this, "Instance ID Map", MEM_READ_ONLY),
      blender_object_id(this, "__blender_object_id", MEM_GLOBAL),
      visibility(this, "Visibility Mask", MEM_READ_ONLY),
      geometry(this, "HIPRT BLAS", MEM_READ_WRITE),
      blas_ptr(this, "__instance_geometry", MEM_GLOBAL),
      transform_matrix_(this, "Transform Matrix", MEM_READ_ONLY),
      transform_headers_(this, "Transform Header", MEM_READ_ONLY),
      curve_intersect_data_offset(this, "__curve_intersect_data_offset", MEM_GLOBAL),
      curve_intersect_data(this, "__curve_intersect_data", MEM_GLOBAL),
      use_lds(true),
      functions_table(NULL),
      HIPDevice(info, stats, profiler)
{

  hiprt_context = 0;
  scene = 0;

  //memset(custom_functions_table, 0, sizeof(custom_functions_table));

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
  visibility.free();
  geometry.free();
  transform_matrix_.free();
  transform_headers_.free();
  curve_intersect_data_offset.free();
  curve_intersect_data.free();
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

    const string block_size_str = to_string(NUM_BLOCK_THREAD);
    const string stack_size_str = to_string(LOCAL_STACK_SIZE);

    string block_size_def = "-D BLOCK_SIZE=" + block_size_str;
    string stack_size_def = "-D SHARED_STACK_SIZE=" + stack_size_str;

if (use_lds) {

      rtc_options.push_back(block_size_def.c_str());
      rtc_options.push_back(stack_size_def.c_str());

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


    HIPRT_API hiprtError hiprtBuildTraceProgram(hiprtContext context,
                                                uint32_t numFunctions,
                                                const char **functionNames,
                                                const char *src,
                                                const char *name,
                                                uint32_t numHeaders,
                                                const char **headersIn,
                                                const char **includeNamesIn,
                                                uint32_t numOptions,
                                                const char **options,
                                                uint32_t numGeomTypes,
                                                uint32_t numRayTypes,
                                                hiprtFuncNameSet *funcNameSets,
                                                void *outProg);

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
  KERNEL_DATA_ARRAY(int2, __curve_intersect_data_offset)
  KERNEL_DATA_ARRAY(int2, __curve_intersect_data)
#  include "kernel/data_arrays.h"
#  undef KERNEL_DATA_ARRAY

}

hiprtGeometryBuildInput HIPRTDevice::prepare_triangle_blas(BVHHIPRT *bvh, Mesh *mesh)
{

  hiprtGeometryBuildInput geomInput;
  geomInput.geomType = Triangle;

  if (mesh->has_motion_blur() && bvh->params.num_motion_triangle_steps != 0) {

    const Attribute *attr_mP = mesh->attributes.find(ATTR_STD_MOTION_VERTEX_POSITION);
    const size_t num_triangles = mesh->num_triangles();
    vector<BoundBox> motion_bound;

    for (uint j = 0; j < num_triangles; j++) {
      Mesh::Triangle t = mesh->get_triangle(j);
      const float3 *verts = mesh->get_verts().data();

      const int num_bvh_steps = bvh->params.num_motion_triangle_steps * 2 + 1;
      const float num_bvh_steps_inv_1 = 1.0f / (num_bvh_steps - 1);
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
          motion_bound.push_back(bounds);
          bvh->motion_blur_time.push_back(make_float2(curr_time, prev_time));
        }
        prev_bounds = curr_bounds;
      }
    }

    hiprtAABBListPrimitive motion_trinagle_aabb;
    motion_trinagle_aabb.aabbCount = motion_bound.size();
    motion_trinagle_aabb.aabbStride = sizeof(BoundBox);

    hipError_t rt_result = hipMalloc(HIPDEVICEPTR_T(motion_trinagle_aabb.aabbs),
                                     motion_bound.size() * sizeof(BoundBox));

    rt_result = hipMemcpyHtoD(*HIPDEVICEPTR_T(motion_trinagle_aabb.aabbs),
                              motion_bound.data(),
                              motion_bound.size() * sizeof(BoundBox));

    geomInput.type = hiprtPrimitiveTypeAABBList;
    geomInput.aabbList.primitive = &motion_trinagle_aabb;
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

  bvh->packed_type.resize(num_segments);

  if (hair->has_motion_blur() && bvh->params.num_motion_curve_steps) {
      curve_attr_mP = hair->attributes.find(ATTR_STD_MOTION_VERTEX_POSITION);
  }

  int num_bounds = 0;
  std::vector<BoundBox> curve_bound;

  for (uint j = 0; j < num_curves; j++) {
    const Hair::Curve curve = hair->get_curve(j);
    const float *curve_radius = &hair->get_curve_radius()[0];
    for (int k = 0; k < curve.num_keys - 1; k++) {
      if (curve_attr_mP == NULL || bvh->params.num_motion_curve_steps == 0) {
        BoundBox bounds = BoundBox::empty;
        curve.bounds_grow(k, &hair->get_curve_keys()[0], curve_radius, bounds);
        if (bounds.valid()) {
          int type = PRIMITIVE_PACK_SEGMENT(primitive_type, k);
          bvh->packed_type[num_bounds].x = j;
          bvh->packed_type[num_bounds].y = type; //k;
          curve_bound.push_back(bounds);
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
            bvh->packed_type[num_bounds].x = j;
            bvh->packed_type[num_bounds].y = packed_type;  // k
            curve_bound.push_back(bounds);
            bvh->motion_blur_time.push_back(make_float2(curr_time, prev_time));
          }
          prev_bounds = curr_bounds;
        }
      }
    }
  }

  bvh->custom_prim_aabb.aabbCount = curve_bound.size();
  bvh->custom_prim_aabb.aabbStride = sizeof(BoundBox);

  hipError_t rt_result = hipMalloc(HIPDEVICEPTR_T(bvh->custom_prim_aabb.aabbs),
                                   curve_bound.size() * sizeof(BoundBox));

  rt_result = hipMemcpyHtoD(*HIPDEVICEPTR_T(bvh->custom_prim_aabb.aabbs),
                            curve_bound.data(),
                            curve_bound.size() * sizeof(BoundBox));

  geomInput.type = hiprtPrimitiveTypeAABBList;
  geomInput.aabbList.primitive = &bvh->custom_prim_aabb;
  geomInput.geomType = Curve;

  return geomInput;
}

hiprtGeometryBuildInput HIPRTDevice::prepare_point_blas(BVHHIPRT *bvh, PointCloud *pointcloud)
{

  hiprtGeometryBuildInput geomInput;

  hiprtAABBListPrimitive point_aabb;
  vector<BoundBox> point_bound;

  const Attribute *point_attr_mP = NULL;
  if (pointcloud->has_motion_blur()) {
    point_attr_mP = pointcloud->attributes.find(ATTR_STD_MOTION_VERTEX_POSITION);
  }

  const float3 *points_data = pointcloud->get_points().data();
  const float *radius_data = pointcloud->get_radius().data();
  const size_t num_points = pointcloud->num_points();
  const float3 *motion_data = (point_attr_mP) ? point_attr_mP->data_float3() : NULL;
  const size_t num_steps = pointcloud->get_motion_steps();

  if (point_attr_mP == NULL) {
    for (uint j = 0; j < num_points; j++) {
      const PointCloud::Point point = pointcloud->get_point(j);
      BoundBox bounds = BoundBox::empty;
      point.bounds_grow(points_data, radius_data, bounds);
      if (bounds.valid()) {
        point_bound.push_back(bounds);
      }
    }
  }
  else if (bvh->params.num_motion_point_steps == 0) {

    for (uint j = 0; j < num_points; j++) {
      const PointCloud::Point point = pointcloud->get_point(j);
      BoundBox bounds = BoundBox::empty;
      point.bounds_grow(points_data, radius_data, bounds);
      for (size_t step = 0; step < num_steps - 1; step++) {
        point.bounds_grow(motion_data + step * num_points, radius_data, bounds);
      }
      if (bounds.valid()) {
        point_bound.push_back(bounds);
      }
    }
  }
  else {

    const int num_bvh_steps = bvh->params.num_motion_point_steps * 2 + 1;
    const float num_bvh_steps_inv_1 = 1.0f / (num_bvh_steps - 1);

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
          point_bound.push_back(bounds);
          bvh->motion_blur_time.push_back(make_float2(curr_time, prev_time));
        }
        prev_bounds = curr_bounds;
      }
    }
  }

  point_aabb.aabbCount = point_bound.size();
  point_aabb.aabbStride = sizeof(BoundBox);

  hipError_t rt_result = hipMalloc(HIPDEVICEPTR_T(point_aabb.aabbs),
                                   point_bound.size() * sizeof(BoundBox));

  rt_result = hipMemcpyHtoD(*HIPDEVICEPTR_T(point_aabb.aabbs),
                            point_bound.data(),
                            point_bound.size() * sizeof(BoundBox));

  geomInput.type = hiprtPrimitiveTypeAABBList;
  geomInput.aabbList.primitive = &point_aabb;
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


  vector<hiprtFrameSRT> transforms;
  array<hiprtFrameMatrix> transform_matrix;

  vector<int2> packed_type;
  unordered_map<Geometry *, int2> packed_type_map;
  size_t curve_offset = 0;

  size_t num_instances = 0;
  int blender_instance_id = 0;

  size_t num_object = objects.size();
  instance_id_map_.alloc(num_object);
  blender_object_id.alloc(num_object);
  visibility.alloc(num_object);
  geometry.alloc(num_object);
  blas_ptr.alloc(num_object);
  transform_headers_.alloc(num_object);
  curve_intersect_data_offset.alloc(num_object);

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
    MAKE_TRANSFORM(hiprt_transform_matrix.matrix, identity_matrix)

    if (hiprt_geom_current) {
      bool has_curve_prim = current_bvh->packed_type.size() > 0;

      if (has_curve_prim) {

        unordered_map<Geometry *, int2>::iterator it = packed_type_map.find(geom);

        if (packed_type_map.find(geom) != packed_type_map.end()) {

          curve_intersect_data_offset[blender_instance_id] = it->second;
        }
        else {
          int offset = packed_type.size();

          packed_type_map[geom].x = offset;
          packed_type_map[geom].y = curve_offset;
          packed_type.resize(offset + current_bvh->packed_type.size());
          memcpy(packed_type.data() + offset,
                 current_bvh->packed_type.data(),
                 current_bvh->packed_type.size() * sizeof(int2));

          curve_intersect_data_offset[blender_instance_id].x = offset;
          curve_intersect_data_offset[blender_instance_id].y = curve_offset;
          curve_offset += ((Hair *)geom)->num_curves();

        }
      }
      else
        curve_intersect_data_offset[blender_instance_id] = {-1, -1};


      hiprtTransformHeader current_header = {0};
      current_header.frameCount = 1;
      current_header.frameIndex = transform_matrix.size();
      if (ob->get_motion().size()) {
        int motion_size = ob->get_motion().size();
        assert(motion_size == 1);
        bool motion_blur = geom->has_motion_blur();
        const Attribute *attr_mP = geom->attributes.find(ATTR_STD_MOTION_VERTEX_POSITION);
        const int num_bvh_steps = bvh->params.num_motion_triangle_steps;
        const int num_bvh_steps_blas = current_bvh->params.num_motion_curve_steps;

        array<Transform> tfm_array = ob->get_motion();
        float time_iternval = 1 / (float)(motion_size - 1);
        current_header.frameCount = motion_size;

        vector<hiprtFrameMatrix> tfm_hiprt_mb;
        tfm_hiprt_mb.resize(motion_size);
        for (int i = 0; i < motion_size; i++) {
          MAKE_TRANSFORM(tfm_hiprt_mb[i].matrix, tfm_array[i]);
          tfm_hiprt_mb[i].time = (float)i * time_iternval;
          transform_matrix.push_back_slow(tfm_hiprt_mb[i]);
        }
      }
      else {
        if (transform_applied)
          current_transform = identity_matrix;
        MAKE_TRANSFORM(hiprt_transform_matrix.matrix, current_transform);
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

  hipError_t rt_result; 

  // copy from host
  if (scene_input_ptr.frameType == hiprtFrameTypeSRT) {
    rt_result = hipMalloc(HIPDEVICEPTR_T(scene_input_ptr.instanceFrames),
                          num_instances * sizeof(hiprtFrameSRT));
    rt_result = hipMemcpyHtoD(*HIPDEVICEPTR_T(scene_input_ptr.instanceFrames),
                              transforms.begin()._Ptr,
                              num_instances * sizeof(hiprtFrameSRT));
  }
 
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

  //setting up function pointers

    hipModule_t current_hipModule = hipModule;

    if (packed_type.size()) {
      size_t data_size = packed_type.size();
      curve_intersect_data.alloc(data_size);
      curve_intersect_data.host_pointer = packed_type.data();
      curve_intersect_data.data_elements = 2;
      curve_intersect_data.data_type = TYPE_INT;
      curve_intersect_data.data_size = data_size;
      curve_intersect_data.copy_to_device();
      curve_intersect_data.host_pointer = 0;

      curve_intersect_data_offset.copy_to_device();

    }

    //const char *filter_functions[] = {"skip_self_filter_func", "shadow_filter_func", "local_filter_func", "volume_filter_func", };
    //const char *intersect_function[] = {
    //    "none",
    //    "curve_intersect_func",
    //    "motion_triangle_intersect_func",
    //    "point_intersect_func"};

    const char *tables[] = {"__table_closest_intersect",
                            "__table_shadow_intersect",
                            "__table_local_intersect",
                            "__table_volume_intersect"};


   /* device_ptr intersection_func_ptr[Max_Primitive_Type] = {0};
    device_ptr intersection_filter_func_ptr[Max_Intersect_Filter_Function] = {0};

    size_t func_ptr_size = 0;


    for (int prim_type = 1; prim_type < Max_Primitive_Type; prim_type++) {

      rt_result = hipModuleGetGlobal(&intersection_func_ptr[prim_type],
                                     &func_ptr_size,
                                     current_hipModule,
                                     intersect_function[prim_type]);
      assert(result == 0);
    }

    for (int filter_function = 0; filter_function < Max_Intersect_Filter_Function;
         filter_function++) {

        rt_result = hipModuleGetGlobal(&intersection_filter_func_ptr[filter_function],
                                     &func_ptr_size,
                                     current_hipModule,
                                     filter_functions[filter_function]);
      assert(result == 0);

    }*/

    for (int table_index = 0; table_index < Max_Intersect_Filter_Function; table_index++) {

      size_t table_ptr_size = 0;
      device_ptr table_device_ptr;

      hip_assert(hipModuleGetGlobal(
          &table_device_ptr, &table_ptr_size, current_hipModule, tables[table_index]));
      hip_assert(hipMemcpyHtoD(table_device_ptr, &functions_table, table_ptr_size));
    }


#if 0
    for (int filter_function = 0; filter_function < Max_Intersect_Filter_Function;
         filter_function++) {

        hiprtError hiprt_result = hiprtCreateCustomFuncTable(hiprt_context, &custom_functions_table[filter_function]);
        assert(hiprt_result == hiprtSuccess);
        //there are four tables for each intersection kernel and each table has a different intersection fiter
        //each table has four entries (or whatever the number of primitive is) for custom intersection per primitive
        //all table entries (for a single table) share the same intersection filter

        hiprtCustomFuncSet custom_functions = {0};
#  ifdef HIPRT_INTERSECTION_FILTERS
        rt_result = hipMemcpyDtoH(&custom_functions.filterFunc,
                                  intersection_filter_func_ptr[filter_function],
                                  func_ptr_size);

        assert(rt_result == hipSuccess);
#  endif

        for (int prim = 0; prim < Max_Primitive_Type; prim++) {

          if (prim != Triangle) {
            rt_result = hipMemcpyDtoH(
                &custom_functions.intersectFunc, intersection_func_ptr[prim], func_ptr_size);
            assert(rt_result == hipSuccess);
          }

          switch (prim) {
            //case Trianlge:
              //break;
            case Curve: {
             // copy intersection function data
              if (0){  //(packed_type.size() > 0) { //change pack_type to device_vector
               device_ptr intersect_func_data = 0;

                rt_result = hipMalloc(&intersect_func_data, packed_type.size() * sizeof(int2));
               assert(hiprt_result == hiprtSuccess);
                rt_result = hipMemcpyHtoD(
                    intersect_func_data, packed_type.data(), packed_type.size() * sizeof(int2));
               assert(hiprt_result == hiprtSuccess);

                custom_functions.intersectFuncData = (void *)intersect_func_data;
              }
              break;
            }
            case Motion_Triangle: {
              // copy time
              break;
            }
            case Point: {
              // if has motion copy time
              break;
            }
            default:
              assert((Primitive_Type)prim == Triangle);

          }

          hiprt_result = hiprtSetCustomFuncTable(
              hiprt_context, custom_functions_table[filter_function], prim, custom_functions);

          assert(hiprt_result == hiprtSuccess);

          size_t table_ptr_size = 0;
          device_ptr table_device_ptr;

          hip_assert(hipModuleGetGlobal(
              &table_device_ptr, &table_ptr_size, current_hipModule, tables[filter_function]));
          hip_assert(hipMemcpyHtoD(
              table_device_ptr, &custom_functions_table[filter_function], table_ptr_size));

        }
    }
    #endif
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

CCL_NAMESPACE_END

#endif

