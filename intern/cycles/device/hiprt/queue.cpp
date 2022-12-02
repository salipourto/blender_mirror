#ifdef WITH_HIPRT

#  include "device/hiprt/queue.h"

#  include "device/hip/graphics_interop.h"
#  include "device/hip/kernel.h"
#  include "device/hiprt/device_impl.h"
#  ifdef KERNEL_TIME
#    include "util/time.h"
#  endif

CCL_NAMESPACE_BEGIN

HIPRTDeviceQueue::HIPRTDeviceQueue(HIPRTDevice *device)
    : HIPDeviceQueue((HIPDevice *)device), hiprt_device_(device)
{
}

bool HIPRTDeviceQueue::enqueue(DeviceKernel kernel,
                               const int work_size,
                               DeviceKernelArguments const &args)
{
  if (hiprt_device_->have_error()) {
    return false;
  }

  bool hiprt_shaders = (kernel == DEVICE_KERNEL_INTEGRATOR_SHADE_SURFACE_RAYTRACE ||
                        kernel == DEVICE_KERNEL_INTEGRATOR_SHADE_SURFACE_MNEE ||
                        kernel == DEVICE_KERNEL_INTEGRATOR_INTERSECT_CLOSEST ||
                        kernel == DEVICE_KERNEL_INTEGRATOR_INTERSECT_SHADOW ||
                        kernel == DEVICE_KERNEL_INTEGRATOR_INTERSECT_SUBSURFACE ||
                        kernel == DEVICE_KERNEL_INTEGRATOR_INTERSECT_VOLUME_STACK);

  if (!hiprt_shaders)
    return HIPDeviceQueue::enqueue(kernel, work_size, args);

  DeviceKernelArguments arg_copy = args;

  const HIPContextScope scope(hiprt_device_);
  const HIPDeviceKernel &hip_kernel = hiprt_device_->kernels.get(kernel);

  /* Compute kernel launch parameters. */
  const int num_threads_per_block = hiprt_device_->use_lds ? NUM_BLOCK_THREAD :
                                                             hip_kernel.num_threads_per_block;
  const int num_blocks = divide_up(work_size, num_threads_per_block);

  int shared_mem_bytes = 0;
#  ifdef KERNEL_TIME
  double start_time = time_dt();
#  endif
  assert_success(hipModuleLaunchKernel(hip_kernel.function,
                                       num_blocks,
                                       1,
                                       1,
                                       num_threads_per_block,
                                       1,
                                       1,
                                       shared_mem_bytes,
                                       hip_stream_,
                                       const_cast<void **>(arg_copy.values),
                                       0),
                 "enqueue");
#  ifdef KERNEL_TIME
  if (synchronize()) {
    double kernel_time = (time_dt() - start_time) * 1000;

    if (kernel == DEVICE_KERNEL_INTEGRATOR_INTERSECT_SHADOW)
      printf("Shadows\t%.3lf ms\n", kernel_time);
    if (kernel == DEVICE_KERNEL_INTEGRATOR_INTERSECT_SUBSURFACE)
      printf("SSR\t%.3lf ms\n", kernel_time);
    if (kernel == DEVICE_KERNEL_INTEGRATOR_INTERSECT_VOLUME_STACK)
      printf("Volume\t%.3lf ms\n", kernel_time);
    if (kernel == DEVICE_KERNEL_INTEGRATOR_INTERSECT_CLOSEST)
      printf("Opaque\t%.3lf ms\n", kernel_time);
  }
#  endif

  return !(hiprt_device_->have_error());
}

CCL_NAMESPACE_END

#endif /* WITH_HIPRT */
