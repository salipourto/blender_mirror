#pragma once

#ifdef WITH_HIPRT

#  include "device/hip/device_impl.h"
#  include "device/hip/kernel.h"
#  include "device/hip/queue.h"
#  include "device/hip/util.h"
#  include "device/hiprt/queue.h"

#  include "hiprt.h"

//#    define OFFLINE_COMPILER
#    define HWI_RT

//#  define HIPRT_INTERSECTION_FILTERS
//#  define KERNEL_TIME


#    define MAKE_TRANSFORM(matrix, tfm) \
      int row = 0; \
      int col = 0; \
      matrix[row][col++] = tfm.x.x; \
      matrix[row][col++] = tfm.x.y; \
      matrix[row][col++] = tfm.x.z; \
      matrix[row][col++] = tfm.x.w; \
      row++;\
      col = 0; \
      matrix[row][col++] = tfm.y.x; \
      matrix[row][col++] = tfm.y.y; \
      matrix[row][col++] = tfm.y.z; \
      matrix[row][col++] = tfm.y.w; \
      row++;\
      col = 0; \
      matrix[row][col++] = tfm.z.x; \
      matrix[row][col++] = tfm.z.y; \
      matrix[row][col++] = tfm.z.z; \
      matrix[row][col++] = tfm.z.w;\

#    define HIPDEVICEPTR_T(hiprt_ptr) (hipDeviceptr_t *)(&hiprt_ptr)

#    define GLOBAL_STACK_SIZE   512*1024*1024
#    define LOCAL_STACK_SIZE 24 //allocation for each thread
#    define NUM_BLOCK_THREAD 256 //total locak stack size would be number of threads * LOCAL_STACK_SIZE

#   define TRANSFORM_MATRIX


CCL_NAMESPACE_BEGIN

class Mesh;
class Hair;
class PointCloud;
class Geometry;
class Object;
class BVHHIPRT;

class HIPRTDevice : public HIPDevice {

 public:

  hipModule_t hipModule_rtc;

  virtual BVHLayoutMask get_bvh_layout_mask() const override;

  HIPRTDevice(const DeviceInfo &info, Stats &stats, Profiler &profiler);

  virtual ~HIPRTDevice();
  virtual unique_ptr<DeviceQueue> gpu_queue_create() override;
  string compile_kernel_get_common_cflags(const uint kernel_features);
  virtual string compile_kernel(const uint kernel_features, const char *name, const char *base = "hiprt") override;

  virtual bool load_kernels(const uint kernel_features) override;

  virtual void const_copy_to(const char *name, void *host, size_t size) override;


  virtual void build_bvh(BVH *bvh, Progress &progress, bool refit) override;
  
  hiprtGeometryBuildInput prepare_triangle_blas(BVHHIPRT *bvh, Mesh *mesh);
  hiprtGeometryBuildInput prepare_curve_blas(BVHHIPRT *bvh, Hair *hair);
  hiprtGeometryBuildInput prepare_point_blas(BVHHIPRT *bvh, PointCloud *pointcloud);
  
  virtual hipModule_t get_hip_module(DeviceKernel kernel_name) override;

  hiprtContext get_hiprt_context()
  {

    return hiprt_context;
  }

  enum Filter_Function {
    Self_Intersect = 0,
    Shadows,
    SSR,
    Volume,
    Max_Intersect_Filter_Function
  };

  enum Primitive_Type {
    Triangle = 0,
    Curve,
    Motion_Triangle,
    Point,
    Max_Primitive_Type
  };

  device_vector<int> instance_id_map_;
  device_vector<int> blender_object_id;
  device_vector<uint32_t> visibility;

  device_vector<uint64_t> geometry;
  device_vector<uint64_t> blas_ptr;
  device_vector<hiprtFrameMatrix> transform_matrix_;
  device_vector<hiprtTransformHeader> transform_headers_;

  device_vector<int2> curve_intersect_data_offset;
  device_vector<int2> curve_intersect_data;
  

  hiprtCustomFuncTable custom_functions_table[Max_Intersect_Filter_Function];

  bool use_lds;

 protected:
 
 bool compile_RT_kernel(const string fatbin_rt, const string include_path, const string source_path);

  hiprtGeometry build_blas(BVHHIPRT *bvh, Geometry *geom, hiprtBuildOptions options);
  hiprtScene build_tlas(BVHHIPRT *bvh,
                        vector<Object *> objects,
                        hiprtBuildOptions options,
                        bool refit);

  hiprtContext hiprt_context;
  hiprtScene scene;

};

CCL_NAMESPACE_END

#endif

