#if !defined( HIPRT_DEVICE_H )
#define HIPRT_DEVICE_H

/** \brief Ray traversal type.
 *
 */
enum hiprtTraversalType{
	/*!< 0 or 1 element iterator with any hit along the ray */
	hiprtTraversalTerminateAtAnyHit = 1,
	/*!< 0 or 1 element iterator with a closest hit along the ray */
	hiprtTraversalTerminateAtClosestHit = 2,
};

/** \brief Traversal state.
 *
 * On-device traversal can be in either hit state (and can be continued using
 * hiprtNextHit) or finished state.
 */
enum hiprtTraversalState
{
	hiprtTraversalStateInit,
	hiprtTraversalStateFinished,
	hiprtTraversalStateHit,
	hiprtTraversalStateStackOverflow
};

/** \brief Traversal hint.
 *
 * An additional information about the rays for the traversal object.
 * It is taken into account only on AMD Navi3x (RDNA3) and above.
 */
enum hiprtTraversalHint
{
	hiprtTraversalHintDefault		 = 0,
	hiprtTraversalHintShadowRays	 = 1,
	hiprtTraversalHintReflectionRays = 2
};

struct _hiprtContext;
struct _hiprtFuncTable;

typedef void*			 hiprtDevicePtr;
typedef hiprtDevicePtr	 hiprtGeometry;
typedef hiprtDevicePtr	 hiprtScene;
typedef uint32_t		 hiprtBuildFlags;
typedef uint32_t		 hiprtRayMask;
typedef _hiprtContext*	 hiprtContext;
typedef _hiprtFuncTable* hiprtFuncTable;

/** \brief Set of device data pointers for custom functions.
 *
 */
struct hiprtFuncDataSet
{
	const void* intersectFuncData;
	const void* filterFuncData;
};

/** \brief Various constants.
 *
 */
enum : uint32_t
{
	hiprtInvalidValue = ~0u,
	hiprtFullRayMask  = ~0u
};

/** \brief Ray data structure.
 *
 */
struct HIPRT_ALIGN( 32 ) hiprtRay
{
	/*!< Ray origin */
	hiprtFloat3 origin;
	/*!< Ray maximum distance */
	float minT = 0.0f;
	/*!< Ray direction */
	hiprtFloat3 direction;
	/*!< Ray maximum distance */
	float maxT = FLT_MAX;
};
static_assert( sizeof( hiprtRay ) == 32 );

/** \brief Ray hit data structure.
 *
 */
struct HIPRT_ALIGN( 32 ) hiprtHit
{
	/*!< Instance ID */
	uint32_t instanceID = hiprtInvalidValue;
	/*!< Primitive ID */
	uint32_t primID = hiprtInvalidValue;
	/*!< Texture coordinates */
	hiprtFloat2 uv;
	/*!< Geeometric normal (not normalized) */
	hiprtFloat3 normal;
	/*!< Distance */
	float t = -1.0f;
};
static_assert( sizeof( hiprtHit ) == 32 );

/** \brief A stack using (slow) local memory internally.
 *
 */
template <uint32_t PrivateStackSize>
class hiprtCustomPrivateStack
{
  public:
	HIPRT_DEVICE hiprtCustomPrivateStack();
	HIPRT_DEVICE int  pop();
	HIPRT_DEVICE void push( int val );
	HIPRT_DEVICE bool empty();
	HIPRT_DEVICE int  vacancy();
	HIPRT_DEVICE void reset();
};

/** \brief A stack using both (fast) shared memory and (slow) global memory.
 *
 * The stack uses shared memory if there is enough space.
 * Otherwise, it uses global memory as a backup.
 */
class hiprtCustomSharedStack
{
  public:
	HIPRT_DEVICE hiprtCustomSharedStack(
		int* globalStackBuffer, u32 globalStackSize, int* sharedStackBuffer = nullptr, u32 sharedStackSize = 0u );
	HIPRT_DEVICE int  pop();
	HIPRT_DEVICE void push( int val );
	HIPRT_DEVICE bool empty();
	HIPRT_DEVICE int  vacancy();
	HIPRT_DEVICE void reset();
};

typedef hiprtCustomPrivateStack<48>	 hiprtPrivateStack48;
typedef hiprtCustomPrivateStack<64>	 hiprtPrivateStack64;
typedef hiprtCustomPrivateStack<128> hiprtPrivateStack128;
typedef hiprtCustomSharedStack		 hiprtGlobalStack;

/** \brief A traversal object for finding the closest hit with hiprtGeometry containing triangles.
 *
 * It uses a private stack with size 64 internally.
 */
class hiprtGeomTraversalClosest
{
  public:
	HIPRT_DEVICE hiprtGeomTraversalClosest( hiprtGeometry geom, const hiprtRay& ray );
	HIPRT_DEVICE hiprtGeomTraversalClosest(
		hiprtGeometry	   geom,
		const hiprtRay&	   ray,
		hiprtTraversalHint hint		 = hiprtTraversalHintDefault,
		void*			   payload	 = nullptr,
		hiprtFuncTable	   funcTable = nullptr,
		u32				   rayType	 = 0 );
	HIPRT_DEVICE hiprtHit			 getNextHit();
	HIPRT_DEVICE hiprtTraversalState getCurrentState();
};

/** \brief A traversal object for finding the any hit with hiprtGeometry containing triangles.
 *
 * It uses a private stack with size 64 internally.
 */
class hiprtGeomTraversalAnyHit
{
  public:
	HIPRT_DEVICE hiprtGeomTraversalAnyHit( hiprtGeometry geom, const hiprtRay& ray );
	HIPRT_DEVICE hiprtGeomTraversalAnyHit(
		hiprtGeometry	   geom,
		const hiprtRay&	   ray,
		hiprtTraversalHint hint		 = hiprtTraversalHintDefault,
		void*			   payload	 = nullptr,
		hiprtFuncTable	   funcTable = nullptr,
		u32				   rayType	 = 0 );
	HIPRT_DEVICE hiprtHit			 getNextHit();
	HIPRT_DEVICE hiprtTraversalState getCurrentState();
};

/** \brief A traversal object for finding the closest hit with hiprtGeometry containing custom primitives.
 *
 * It uses a private stack with size 64 internally.
 */
class hiprtGeomCustomTraversalClosest
{
  public:
	HIPRT_DEVICE hiprtGeomCustomTraversalClosest(
		hiprtGeometry	   geom,
		const hiprtRay&	   ray,
		hiprtTraversalHint hint		 = hiprtTraversalHintDefault,
		void*			   payload	 = nullptr,
		hiprtFuncTable	   funcTable = nullptr,
		u32				   rayType	 = 0 );
	HIPRT_DEVICE hiprtHit			 getNextHit();
	HIPRT_DEVICE hiprtTraversalState getCurrentState();
};

/** \brief A traversal object for finding the any hit with hiprtGeometry containing custom primitives.
 *
 * It uses a private stack with size 64 internally.
 */
class hiprtGeomCustomTraversalAnyHit
{
  public:
	HIPRT_DEVICE hiprtGeomCustomTraversalAnyHit(
		hiprtGeometry	   geom,
		const hiprtRay&	   ray,
		hiprtTraversalHint hint		 = hiprtTraversalHintDefault,
		void*			   payload	 = nullptr,
		hiprtFuncTable	   funcTable = nullptr,
		u32				   rayType	 = 0 );
	HIPRT_DEVICE hiprtHit			 getNextHit();
	HIPRT_DEVICE hiprtTraversalState getCurrentState();
};

/** \brief A traversal object for finding the closest hit with hiprtScene.
 *
 * It uses a private stack with size 64 internally.
 */
class hiprtSceneTraversalClosest
{
  public:
	HIPRT_DEVICE hiprtSceneTraversalClosest( hiprtScene scene, const hiprtRay& ray, hiprtRayMask mask );
	HIPRT_DEVICE hiprtSceneTraversalClosest(
		hiprtScene		   scene,
		const hiprtRay&	   ray,
		hiprtRayMask	   mask		 = hiprtInvalidValue,
		hiprtTraversalHint hint		 = hiprtTraversalHintDefault,
		void*			   payload	 = nullptr,
		hiprtFuncTable	   funcTable = nullptr,
		u32				   rayType	 = 0,
		float			   time		 = 0.0f );
	HIPRT_DEVICE hiprtHit			 getNextHit();
	HIPRT_DEVICE hiprtTraversalState getCurrentState();
};

/** \brief A traversal object for finding the any hit with hiprtScene.
 *
 * It uses a private stack with size 64 internally.
 */
class hiprtSceneTraversalAnyHit
{
  public:
	HIPRT_DEVICE hiprtSceneTraversalAnyHit( hiprtScene scene, const hiprtRay& ray, hiprtRayMask mask );
	HIPRT_DEVICE hiprtSceneTraversalAnyHit(
		hiprtScene		   scene,
		const hiprtRay&	   ray,
		hiprtRayMask	   mask		 = hiprtInvalidValue,
		hiprtTraversalHint hint		 = hiprtTraversalHintDefault,
		void*			   payload	 = nullptr,
		hiprtFuncTable	   funcTable = nullptr,
		u32				   rayType	 = 0,
		float			   time		 = 0.0f );
	HIPRT_DEVICE hiprtHit			 getNextHit();
	HIPRT_DEVICE hiprtTraversalState getCurrentState();
};

/** \brief A traversal object for finding the closest hit with hiprtGeometry containing triangles.
 *
 * \tparam hiprtStack A custom stack.
 */
template <typename hiprtStack>
class hiprtGeomTraversalClosestCustomStack
{
  public:
	HIPRT_DEVICE hiprtGeomTraversalClosestCustomStack(
		hiprtGeometry	   geom,
		const hiprtRay&	   ray,
		hiprtStack&		   stack,
		hiprtTraversalHint hint		 = hiprtTraversalHintDefault,
		void*			   payload	 = nullptr,
		hiprtFuncTable	   funcTable = nullptr,
		u32				   rayType	 = 0 );
	HIPRT_DEVICE hiprtHit getNextHit();
	HIPRT_DEVICE hiprtTraversalState getCurrentState();
};

/** \brief A traversal object for finding the any hit with hiprtGeometry containing triangles.
 *
 * \tparam hiprtStack A custom stack.
 */
template <typename hiprtStack>
class hiprtGeomTraversalAnyHitCustomStack
{
  public:
	HIPRT_DEVICE hiprtGeomTraversalAnyHitCustomStack(
		hiprtGeometry	   geom,
		const hiprtRay&	   ray,
		hiprtStack&		   stack,
		hiprtTraversalHint hint		 = hiprtTraversalHintDefault,
		void*			   payload	 = nullptr,
		hiprtFuncTable	   funcTable = nullptr,
		u32				   rayType	 = 0 )
	HIPRT_DEVICE hiprtHit			 getNextHit();
	HIPRT_DEVICE hiprtTraversalState getCurrentState();
};

/** \brief A traversal object for finding the closest hit with hiprtGeometry containing custom primitives.
 *
 * \tparam hiprtStack A custom stack.
 */
template <typename hiprtStack>
class hiprtGeomCustomTraversalClosestCustomStack
{
  public:
	HIPRT_DEVICE hiprtGeomCustomTraversalClosestCustomStack(
		hiprtGeometry	   geom,
		const hiprtRay&	   ray,
		hiprtStack&		   stack,
		hiprtTraversalHint hint		 = hiprtTraversalHintDefault,
		void*			   payload	 = nullptr,
		hiprtFuncTable	   funcTable = nullptr,
		u32				   rayType	 = 0 )
	HIPRT_DEVICE hiprtHit getNextHit();
	HIPRT_DEVICE hiprtTraversalState getCurrentState();
};

/** \brief A traversal object for finding the any hit with hiprtGeometry containing custom primitives.
 *
 * \tparam hiprtStack A custom stack.
 */
template <typename hiprtStack>
class hiprtGeomCustomTraversalAnyHitCustomStack
{
  public:
	HIPRT_DEVICE hiprtGeomCustomTraversalAnyHitCustomStack(
		hiprtGeometry	   geom,
		const hiprtRay&	   ray,
		hiprtStack&		   stack,
		hiprtTraversalHint hint		 = hiprtTraversalHintDefault,
		void*			   payload	 = nullptr,
		hiprtFuncTable	   funcTable = nullptr,
		u32				   rayType	 = 0 )
	HIPRT_DEVICE hiprtHit			 getNextHit();
	HIPRT_DEVICE hiprtTraversalState getCurrentState();
};

/** \brief A traversal object for finding the closest hit with hiprtScene.
 *
 * \tparam hiprtStack A custom stack.
 */
template <typename hiprtStack>
class hiprtSceneTraversalClosestCustomStack
{
  public:
	HIPRT_DEVICE hiprtSceneTraversalClosestCustomStack(
		hiprtScene		   scene,
		const hiprtRay&	   ray,
		hiprtStack&		   stack,
		hiprtRayMask	   mask		 = hiprtInvalidValue,
		hiprtTraversalHint hint		 = hiprtTraversalHintDefault,
		void*			   payload	 = nullptr,
		hiprtFuncTable	   funcTable = nullptr,
		u32				   rayType	 = 0,
		float			   time		 = 0.0f );
	HIPRT_DEVICE hiprtHit getNextHit();
	HIPRT_DEVICE hiprtTraversalState getCurrentState();
};

/** \brief A traversal object for finding the any hit with hiprtScene.
 *
 * \tparam hiprtStack A custom stack.
 */
template <typename hiprtStack>
class hiprtSceneTraversalAnyHitCustomStack
{
  public:
	HIPRT_DEVICE hiprtSceneTraversalAnyHitCustomStack(
		hiprtScene		   scene,
		const hiprtRay&	   ray,
		hiprtStack&		   stack,
		hiprtRayMask	   mask		 = hiprtInvalidValue,
		hiprtTraversalHint hint		 = hiprtTraversalHintDefault,
		void*			   payload	 = nullptr,
		hiprtFuncTable	   funcTable = nullptr,
		u32				   rayType	 = 0,
		float			   time		 = 0.0f );
	HIPRT_DEVICE hiprtHit getNextHit();
	HIPRT_DEVICE hiprtTraversalState getCurrentState();
};
#endif
