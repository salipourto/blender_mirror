#ifdef __HIP_DEVICE_COMPILE__

#  include "kernel/device/hiprt/compat.h"
#  include "kernel/device/hiprt/config.h"
#  if defined(__OFFLINE_COMPILER__) && !defined(__HIPCC_RTC__)
#  include "hiprt/impl/hiprt_device_impl.h"
#  endif
#  include "kernel/device/hiprt/globals.h"
#  include "kernel/device/gpu/image.h"
#  include "kernel/device/gpu/kernel.h"

#endif
