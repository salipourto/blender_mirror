#if (defined(__HIPCC_RTC__) || defined(__OFFLINE_COMPILER__))
struct RayPayload {
  RaySelfPrimitives self;
  KernelGlobals kg;
  uint visibility;
  int prim_type;
  float ray_time;
};

struct ShadowPayload {
  KernelGlobals kg;
  RaySelfPrimitives self;
  int in_state;
  uint max_hits;
  uint visibility;
  uint num_hits;
  uint *r_num_recorded_hits;
  float *r_throughput;
  bool is_hit;
  //float ray_time;
};

struct LocalPayload {
  KernelGlobals kg;
  RaySelfPrimitives self;
  int local_object;
  uint max_hits;
  bool is_hit;
  uint *lcg_state;
  LocalIntersection *local_isect;
  //float ray_tmin;
};

#define SET_HIPRT_RAY(RAY_RT, RAY)\
  RAY_RT.direction = RAY->D;\
  RAY_RT.origin = RAY->P;\
  RAY_RT.maxT = RAY->tmax;\
  RAY_RT.minT = RAY->tmin;

#  if defined(HIPRT_SHARED_STACK)
#    define GET_TRAVERSAL_STACK() \
      Stack stack(&global_stack_buffer[0], STACK_SIZE, kg->shared_stack, SHARED_STACK_SIZE);
#  else
#    define GET_TRAVERSAL_STACK()
#  endif

#  ifdef HIPRT_SHARED_STACK
#    define GET_TRAVERSAL_ANY_HIT(FUNCTION_TABLE, RAY_TYPE) \
      hiprtSceneTraversalAnyHitCustomStack<Stack> traversal(kernel_data.device_bvh, \
                                                            ray_hip, \
                                                            stack, \
                                                            visibility, \
                                                            hiprtTraversalHintDefault, \
                                                            &payload, \
															FUNCTION_TABLE, \
															RAY_TYPE); \
      hiprtSceneTraversalAnyHitCustomStack<Stack> traversal_simple( \
          kernel_data.device_bvh, ray_hip, stack, visibility);
#    define GET_TRAVERSAL_CLOSEST_HIT(FUNCTION_TABLE, RAY_TYPE) \
      hiprtSceneTraversalClosestCustomStack<Stack> traversal(kernel_data.device_bvh, \
                                                             ray_hip, \
                                                             stack, \
                                                             visibility, \
                                                             hiprtTraversalHintDefault, \
                                                             &payload, \
															 FUNCTION_TABLE, \
															 RAY_TYPE); \
      hiprtSceneTraversalClosestCustomStack<Stack> traversal_simple( \
          kernel_data.device_bvh, ray_hip, stack, visibility);
#  else
#    define GET_TRAVERSAL_ANY_HIT(FUNCTION_TABLE) \
      hiprtSceneTraversalAnyHit traversal(kernel_data.device_bvh, \
                                          ray_hip, \
                                          visibility, \
                                          FUNCTION_TABLE, \
                                          hiprtTraversalHintDefault, \
                                          &payload); \
      hiprtSceneTraversalAnyHit traversal_simple(kernel_data.device_bvh, ray_hip, visibility);
#    define GET_TRAVERSAL_CLOSEST_HIT(FUNCTION_TABLE) \
      hiprtSceneTraversalClosest traversal(kernel_data.device_bvh, \
                                           ray_hip, \
                                           visibility, \
                                           FUNCTION_TABLE, \
                                           hiprtTraversalHintDefault, \
                                           &payload); \
      hiprtSceneTraversalClosest traversal_simple(kernel_data.device_bvh, ray_hip, visibility);
#  endif

ccl_device_inline void set_intersect_point(KernelGlobals kg,
                                           hiprtHit &hit,
                                           ccl_private Intersection *isect)
{
  int prim_offset = 0;
  int object_id = kernel_data_fetch(__blender_object_id, hit.instanceID);
  prim_offset = kernel_data_fetch(object_prim_offset, object_id);
  isect->type = kernel_data_fetch(objects, object_id).primitive_type;

  isect->t = hit.t;
  isect->prim = hit.primID + prim_offset;
  isect->object = object_id;
  isect->u = hit.uv.x;
  isect->v = hit.uv.y;
}

// custom intersection functions


ccl_device_inline bool curve_custom_intersect(const hiprtRay &ray,
                                             const void *userPtr,
                                             void *payload,
											 hiprtHit &hit)

{
  Intersection isect;
  RayPayload *local_payload = (RayPayload *)payload;

  KernelGlobals kg = local_payload->kg;

  int object_id = kernel_data_fetch(__blender_object_id, hit.instanceID);
  int2 data_offset = kernel_data_fetch(__curve_intersect_data_offset, object_id);

  int prim_offset = data_offset.y;

  int curve_index = kernel_data_fetch(__curve_intersect_data, hit.primID + data_offset.x).x;
  int key_value = kernel_data_fetch(__curve_intersect_data,  hit.primID + data_offset.x).y;

  if (intersection_skip_self_shadow(local_payload->self, object_id, curve_index + prim_offset))
    return false;
  bool b_hit = curve_intersect(kg,
                             &isect,
                             ray.origin,
                             ray.direction,
                             ray.minT,
                             ray.maxT,
                             object_id,
                             curve_index + prim_offset,
                             local_payload->ray_time,
                             key_value);
  if (b_hit) {
    hit.uv.x = isect.u;
    hit.uv.y = isect.v;
    hit.t = isect.t;
    hit.primID = isect.prim;    // curve_index + prim_offset;
    local_payload->prim_type = isect.type;  // packed_curve_type;
  }
  return b_hit;
}

ccl_device_inline bool motion_triangle_custom_intersect(const hiprtRay &ray,                                                 
                                                 const void *userPtr,
                                                 void *payload,
                                                 hiprtHit &hit)
{
  return false;
}

ccl_device_inline bool point_custom_intersect(const hiprtRay &ray,                                       
                                       const void *userPtr,
                                       void *payload,
									   hiprtHit &hit)
{
  return false;
}

// intersection filters

ccl_device_inline bool opaque_intersection_filter(const hiprtRay &ray,
                                                      const void *data,
                                                      void *user_data,
                                                      const hiprtHit &hit)
{
  RayPayload *payload = (RayPayload *)user_data;
  int object_id = kernel_data_fetch(__blender_object_id, hit.instanceID);
  int prim_offset = kernel_data_fetch(object_prim_offset, object_id);
  int prim = hit.primID + prim_offset;

  if (intersection_skip_self_shadow(payload->self, object_id, prim))
    return true;
  else
    return false;
}

ccl_device_inline bool shadow_intersection_filter(const hiprtRay &ray,
                                             const void *data,
                                             void *user_data,
											 const hiprtHit &hit)


{
  ShadowPayload *payload = (ShadowPayload *)user_data;

  uint num_hits = payload->num_hits;
  uint num_recorded_hits = *(payload->r_num_recorded_hits);
  uint max_hits = payload->max_hits;
  int state = payload->in_state;
  KernelGlobals kg = payload->kg;
  RaySelfPrimitives self = payload->self;

  int object = kernel_data_fetch(__blender_object_id, hit.instanceID);
  int prim_offset = kernel_data_fetch(object_prim_offset, object);
  int prim = hit.primID + prim_offset;
  
  float ray_tmax = hit.t;

#  ifdef __VISIBILITY_FLAG__

  if ((kernel_data_fetch(objects, object).visibility & payload->visibility) == 0) {
    payload->is_hit = false;
    return true;  // no hit - continue traversal
  }
#  endif

  if (intersection_skip_self_shadow(self, object, prim)) {
    payload->is_hit = false;
    return true;  // no hit -continue traversal
  }

  float u = hit.uv.x;
  float v = hit.uv.y;
  int type = kernel_data_fetch(objects, object).primitive_type;
#  ifdef __HAIR__
  if (type & (PRIMITIVE_CURVE_THICK | PRIMITIVE_CURVE_RIBBON)) {

    const KernelCurveSegment segment = kernel_data_fetch(curve_segments, prim);
    type = segment.type;
    prim = segment.prim;

    /*if (u == 0.0f || u == 1.0f) {
      payload->is_hit = true;
        return true;
    }*/
  }
#  endif

#  ifndef __TRANSPARENT_SHADOWS__

  payload->is_hit = true;
  return false;

#  else

  if (num_hits >= max_hits ||
      !(intersection_get_shader_flags(NULL, prim, type) & SD_HAS_TRANSPARENT_SHADOW)) {
    payload->is_hit = true;
    return false;
  }

  if (type & PRIMITIVE_CURVE) {
    float throughput = *payload->r_throughput;
    throughput *= intersection_curve_shadow_transparency(kg, object, prim, type, u);
    *payload->r_throughput = throughput;
    payload->num_hits += 1;

    if (throughput < CURVE_SHADOW_TRANSPARENCY_CUTOFF) {
      payload->is_hit = true;
      return false;
    }
    else {
      return true;
    }
  }

  uint record_index = num_recorded_hits;

  num_hits += 1;
  num_recorded_hits += 1;
  payload->num_hits = num_hits;
  *(payload->r_num_recorded_hits) = num_recorded_hits;

  const uint max_record_hits = min(max_hits, INTEGRATOR_SHADOW_ISECT_SIZE);
  if (record_index >= max_record_hits) {
    float max_recorded_t = INTEGRATOR_STATE_ARRAY(state, shadow_isect, 0, t);
    uint max_recorded_hit = 0;

    for (int i = 1; i < max_record_hits; i++) {
      const float isect_t = INTEGRATOR_STATE_ARRAY(state, shadow_isect, i, t);
      if (isect_t > max_recorded_t) {
        max_recorded_t = isect_t;
        max_recorded_hit = i;
      }
    }

    if (ray_tmax >= max_recorded_t) {

      payload->is_hit = true;
      return true;
    }

    record_index = max_recorded_hit;
  }

  INTEGRATOR_STATE_ARRAY_WRITE(state, shadow_isect, record_index, u) = u;
  INTEGRATOR_STATE_ARRAY_WRITE(state, shadow_isect, record_index, v) = v;
  INTEGRATOR_STATE_ARRAY_WRITE(state, shadow_isect, record_index, t) = ray_tmax;
  INTEGRATOR_STATE_ARRAY_WRITE(state, shadow_isect, record_index, prim) = prim;
  INTEGRATOR_STATE_ARRAY_WRITE(state, shadow_isect, record_index, object) = object;
  INTEGRATOR_STATE_ARRAY_WRITE(state, shadow_isect, record_index, type) = type;
  return true;

#  endif /* __TRANSPARENT_SHADOWS__ */
}

ccl_device_inline bool local_intersection_filter(const hiprtRay &ray,
                                                 const void *data,
                                                 void *user_data,
												 const hiprtHit &hit)
{

#  ifdef __BVH_LOCAL__
  LocalPayload *payload = (LocalPayload *)user_data;
  KernelGlobals kg = payload->kg;
  int object_id = payload->local_object;
  int prim_offset = kernel_data_fetch(object_prim_offset, object_id);
  int prim = hit.primID + prim_offset;
#    ifndef __RAY_OFFSET__
  if (intersection_skip_self_local(payload->self, prim)) {
    payload->is_hit = false;
    return true;  // continue search
  }
#    endif
  uint max_hits = payload->max_hits;
  if (max_hits == 0) {
    payload->is_hit = true;
    return false;  // stop search
  }
  int hit_index = 0;
  if (payload->lcg_state) {
    for (int i = min(max_hits, payload->local_isect->num_hits) - 1; i >= 0; --i) {
      if (hit.t == payload->local_isect->hits[i].t) {
        payload->is_hit = false;
        return true;  // continue search
      }
    }
    hit_index = payload->local_isect->num_hits++;
    if (payload->local_isect->num_hits > max_hits) {
      hit_index = lcg_step_uint(payload->lcg_state) % payload->local_isect->num_hits;
      if (hit_index >= max_hits) {
        payload->is_hit = false;
        return true;  // continue search
      }
    }
  }
  else {
    if (payload->local_isect->num_hits && hit.t > payload->local_isect->hits[0].t) {
      payload->is_hit = false;
      return true;
    }
    payload->local_isect->num_hits = 1;
  }
  Intersection *isect = &payload->local_isect->hits[hit_index];
  isect->t = hit.t;
  isect->prim = prim;
  isect->object = object_id;
  isect->type = PRIMITIVE_TRIANGLE;  // kernel_data_fetch(__objects, object_id).primitive_type;

  isect->u = hit.uv.x;
  isect->v = hit.uv.y;

  payload->local_isect->Ng[hit_index] = hit.normal;

  payload->is_hit = false;
  return true;

#  endif
}

ccl_device_inline bool volume_intersection_filter(const hiprtRay &ray,
                                                      const void *data,
                                                      void *user_data,
                                                      const hiprtHit &hit)
{
  RayPayload *payload = (RayPayload *)user_data;
  int object_id = kernel_data_fetch(__blender_object_id, hit.instanceID);
  int prim_offset = kernel_data_fetch(object_prim_offset, object_id);
  int prim = hit.primID + prim_offset;
  int object_flag = kernel_data_fetch(object_flag, object_id);

  if (intersection_skip_self(payload->self, object_id, prim))
    return true;
  else if ((object_flag & SD_OBJECT_HAS_VOLUME) == 0)
    return true;
  else
    return false;
}


ccl_device_inline bool hiprt_shadow_all(KernelGlobals kg,
                                        IntegratorShadowState state,
                                        ccl_private const Ray *ray,
                                        const uint visibility,
                                        const uint max_hits,
                                        ccl_private uint *r_num_recorded_hits,
                                        ccl_private float *r_throughput)
{

  hiprtRay ray_hip;
  /*ray_hip.origin = ray->P;
  ray_hip.direction = ray->D;
  ray_hip.maxT = ray->tmax;
  ray_hip.time = ray->time;*/
  
  SET_HIPRT_RAY(ray_hip, ray)

  ShadowPayload payload;

  payload.kg = kg;
  payload.self = ray->self;
  payload.in_state = state;
  payload.max_hits = max_hits;
  payload.visibility = visibility;
  payload.num_hits = 0;
  payload.r_num_recorded_hits = r_num_recorded_hits;
  payload.r_throughput = r_throughput;
  payload.is_hit = false;

  GET_TRAVERSAL_STACK()
  GET_TRAVERSAL_ANY_HIT(__table_shadow_intersect, 1)
  hiprtHit hit = traversal.getNextHit();
#  ifndef HIPRT_INTERSECTION_FILTERS
  bool get_next = true;

  float ray_max = ray_hip.maxT;
  const uint max_record_hits = min(max_hits, INTEGRATOR_SHADOW_ISECT_SIZE);
  while (get_next) {
    if (!hit.hasHit() || (hiprtTraversalStateStackOverflow == traversal.getCurrentState()))
      return payload.is_hit;
    ray_max = hit.t;

    get_next = shadow_all_hit_filter(
        ray_hip, hit.instanceID, hit.primID, 0, &payload, hit.uv, hit.normal, ray_max);

    r_num_recorded_hits = payload.r_num_recorded_hits;
    r_throughput = payload.r_throughput;

    hit = traversal.getNextHit();
  }
#  else
  r_num_recorded_hits = payload.r_num_recorded_hits;
  r_throughput = payload.r_throughput;
#  endif
  return payload.is_hit;
}
#endif

