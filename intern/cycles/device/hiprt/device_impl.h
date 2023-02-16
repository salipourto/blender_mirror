/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2011-2022 Blender Foundation */

#pragma once

#ifdef WITH_HIPRT

#  include "device/hip/device_impl.h"
#  include "device/hip/kernel.h"
#  include "device/hip/queue.h"
#  include "device/hiprt/queue.h"
#  include "hiprt/hiprt.h"

#  define HIPRT_INTERSECTION_FILTERS

#  define HIPRT_GLOBAL_STACK_SIZE 512 * 1024 * 1024
#  define HIPRT_SHARED_STACK_SIZE 24  // LDS allocation for each thread
#  define HIPRT_THREAD_STACK_SIZE 64  // global stack allocation per thread
#  define HIPRT_THREAD_GROUP_SIZE \
    256  // total locaal stack size would be number of threads * HIPRT_SHARED_STACK_SIZE

CCL_NAMESPACE_BEGIN

class Mesh;
class Hair;
class PointCloud;
class Geometry;
class Object;
class BVHHIPRT;

void get_hiprt_transform(float matrix[][4], Transform &tfm);

class HIPRTDevice : public HIPDevice {

 public:
  virtual BVHLayoutMask get_bvh_layout_mask() const override;

  HIPRTDevice(const DeviceInfo &info, Stats &stats, Profiler &profiler);

  virtual ~HIPRTDevice();
  virtual unique_ptr<DeviceQueue> gpu_queue_create() override;
  string compile_kernel_get_common_cflags(const uint kernel_features);
  virtual string compile_kernel(const uint kernel_features,
                                const char *name,
                                const char *base = "hiprt") override;

  virtual bool load_kernels(const uint kernel_features) override;

  virtual void const_copy_to(const char *name, void *host, size_t size) override;

  virtual void build_bvh(BVH *bvh, Progress &progress, bool refit) override;

  hiprtGeometryBuildInput prepare_triangle_blas(BVHHIPRT *bvh, Mesh *mesh);
  hiprtGeometryBuildInput prepare_curve_blas(BVHHIPRT *bvh, Hair *hair);
  hiprtGeometryBuildInput prepare_point_blas(BVHHIPRT *bvh, PointCloud *pointcloud);

  hiprtContext get_hiprt_context()
  {
    return hiprt_context;
  }

  bool use_lds;

 protected:

   enum Filter_Function { Opaque = 0, Shadows, SSR, Volume, Max_Intersect_Filter_Function };
   enum Primitive_Type { Triangle = 0, Curve, Motion_Triangle, Point, Max_Primitive_Type };

  bool set_function_table(hiprtFuncNameSet *func_name_set);

  hiprtGeometry build_blas(BVHHIPRT *bvh, Geometry *geom, hiprtBuildOptions options);
  hiprtScene build_tlas(BVHHIPRT *bvh,
                        vector<Object *> objects,
                        hiprtBuildOptions options,
                        bool refit);

  device_vector<int> instance_id_map_;
  device_vector<int> blender_object_id;
  device_vector<uint32_t> visibility;

  device_vector<uint64_t> geometry;
  device_vector<uint64_t> blas_ptr;
  device_vector<hiprtFrameMatrix> transform_matrix_;
  device_vector<hiprtTransformHeader> transform_headers_;

  device_vector<int2> custom_prim_info_offset;
  device_vector<int2> custom_prim_info;

  device_vector<int> prim_time_offset;
  device_vector<float2> prim_time;

  hiprtContext hiprt_context;
  hiprtScene scene;
  hiprtFuncTable functions_table;
};
CCL_NAMESPACE_END

#endif
