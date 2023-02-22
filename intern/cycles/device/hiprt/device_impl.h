/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2011-2022 Blender Foundation */

#pragma once

#ifdef WITH_HIPRT

#  include "device/hip/device_impl.h"
#  include "device/hip/kernel.h"
#  include "device/hip/queue.h"
#  include "device/hiprt/queue.h"
#  include "hiprt/hiprt.h"


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

static void get_hiprt_transform(float matrix[][4], Transform &tfm);

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



  //instance/object ids are not explicitly  passed to hiprt
  //hiprt assigns the ids based on the order blas pointers are passed to it (through instanceGeometries member of hiprtSceneBuildInput)
  //if blas is absent for a particular geometry (e.g. a plane), hiprt removes that entry and in scenes with objects with no blas, the instance id that hiprt
  //returns for a hit point will not necessarily match the instance id of the application
  //user_instance_id provides a map for retrieving original instance id from hiprt instance id
  //hiprt_blas_ptr is the list of all the valid blas pointers
  //blas_ptr has all the valid pointers and null pointers and blas for any geometry can be directly retrieved from this array (used in subsurface scattering)
  device_vector<int> user_instance_id;
  device_vector<uint64_t> hiprt_blas_ptr;
  device_vector<uint64_t> blas_ptr;


  device_vector<uint32_t> visibility;

  //instance_transform_matrix passes transform matrix of instances converted from Cycles Transform format to
  //instanceFrames member of hiprtSceneBuildInput
  device_vector<hiprtFrameMatrix> instance_transform_matrix;
  //Movement over a time interval for motion blur is captured through multiple transform matrices
  //in this case transform matrix of an instance cannot be directly retrieved by looking up instance_transform_matrix at the instance id
  //transform_headers maps the instance id to the appropriate index to retrieve instance transform matrix (frameIndex member of hiprtTransformHeader)
  //transform_headers also has the information on how many transform matrices are associated with an instance (frameCount member of hiprtTransformHeader)
  //transform_headers is passed to hiprt through instanceTransformHeaders member of hiprtSceneBuildInput
  device_vector<hiprtTransformHeader> transform_headers;

  device_vector<int2> custom_prim_info_offset;
  device_vector<int2> custom_prim_info;

  device_vector<int> prim_time_offset;
  device_vector<float2> prims_time;

  hiprtContext hiprt_context;
  hiprtScene scene;
  hiprtFuncTable functions_table;
};
CCL_NAMESPACE_END

#endif
