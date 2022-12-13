#ifndef HIPRT_H
#define HIPRT_H

#define HIPRT_API_MAJOR_VERSION 0x000001
#define HIPRT_API_MINOR_VERSION 0x000002
#define HIPRT_API_PATCH_VERSION 0x000000
#define HIPRT_API_VERSION HIPRT_API_MAJOR_VERSION * 1000000 + HIPRT_API_MINOR_VERSION * 1000 + HIPRT_API_PATCH_VERSION

#include "hiprt_vec.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined( _MSC_VER )
#ifdef HIPRT_EXPORTS
#define HIPRT_API __declspec( dllexport )
#else
#define HIPRT_API __declspec( dllimport )
#endif
#elif defined( __GNUC__ )
#ifdef HIPRT_EXPORTS
#define HIPRT_API __attribute__( ( visibility( "default" ) ) )
#else
#define HIPRT_API
#endif
#else
#define HIPRT_API
#pragma warning Unknown dynamic link import / export semantics.
#endif

#ifndef HIPRT_EXPORTS
#if defined( __GNUC__ )
#define HIPRT_ALIGN( N ) __attribute__( ( aligned( N ) ) )
#elif defined( _MSC_VER )
#define HIPRT_ALIGN( N ) __declspec( align( N ) )
#else
#error "Please provide a definition for HIPRT_ALIGN!!"
#endif
#endif

struct _hiprtGeometry;
struct _hiprtGeometryCustom;
struct _hiprtScene;
struct _hiprtContext;
struct _hiprtFuncTable;

typedef void*			 hiprtDevicePtr;
typedef hiprtDevicePtr	 hiprtGeometry;
typedef hiprtDevicePtr	 hiprtScene;
typedef uint32_t		 hiprtBuildFlags;
typedef uint32_t		 hiprtRayMask;
typedef _hiprtContext*	 hiprtContext;
typedef _hiprtFuncTable* hiprtFuncTable;

typedef int	  hiprtApiDevice;	// hipDevice, cuDevice
typedef void* hiprtApiCtx;		// hipCtx, cuCtx
typedef void* hiprtApiStream;	// hipStream, cuStream
typedef void* hiprtApiFunction; // hipFunction, cuFunction

/** \brief Various constants.
 *
 */
enum : uint32_t
{
	hiprtInvalidValue		= ~0u,
	hiprtFullRayMask		= ~0u,
};

/** \brief Error codes.
 *
 */
enum hiprtError
{
	hiprtSuccess				= 0,
	hiprtErrorNotImplemented	= 1,
	hiprtErrorInternal			= 2,
	hiprtErrorOutOfHostMemory	= 3,
	hiprtErrorOutOfDeviceMemory = 4,
	hiprtErrorInvalidApiVersion = 5,
	hiprtErrorInvalidParameter	= 6
};

/** \brief Type of geometry/scene build operation.
 *
 * hiprtBuildGeometry/hiprtBuildScene can either build or update
 * an underlying acceleration structure.
 */
enum hiprtBuildOperation{
	hiprtBuildOperationBuild  = 1,
	hiprtBuildOperationUpdate = 2
};

/** \brief Hint flags for geometry/scene build functions.
 *
 * hiprtBuildGeometry/hiprtBuildScene use these flags to choose
 * an appropriate build format/algorithm.
 */
enum hiprtBuildFlagBits
{
	hiprtBuildFlagBitPreferFastBuild		= 0,
	hiprtBuildFlagBitPreferBalancedBuild	= 1,
	hiprtBuildFlagBitPreferHighQualityBuild = 2,
	hiprtBuildFlagBitCustomBvhImport		= 3,
	hiprtBuildFlagBitDisableSpatialSplits	= 1 << 2
};

/** \brief Geometric primitive type.
 *
 * hiprtGeometry can be built from multiple primitive types,
 * such as triangle meshes, AABB lists, line lists, etc. This enum
 * defines primitive type for hiprtBuildGeometry function.
 */
enum hiprtPrimitiveType
{
	hiprtPrimitiveTypeTriangleMesh,
	hiprtPrimitiveTypeAABBList
};

/** \brief Transformation frame type.
 *
 */
enum hiprtFrameType
{
	hiprtFrameTypeSRT,
	hiprtFrameTypeMatrix
};

/** \brief Bvh node type.
 *
 */
enum hiprtBvhNodeType
{
	/*!< Leaf node */
	hiprtBvhNodeTypeInternal = 0,
	/*!< Internal node */
	hiprtBvhNodeTypeLeaf = 1,
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

/** \brief Set of device data pointers for custom functions.
 *
 */
struct hiprtFuncDataSet
{
	const void* intersectFuncData = nullptr;
	const void* filterFuncData	  = nullptr;
};

/** \brief Set of custom function names.
 *
 */
struct hiprtFuncNameSet
{
	const char* intersectFuncName = nullptr;
	const char* filterFuncName	  = nullptr;
};

/** \brief Device type.
 *
 */
enum hiprtDeviceType
{
	/*!< AMD device */
	hiprtDeviceAMD,
	/*!< Nvidia device */
	hiprtDeviceNVIDIA,
};

/** \brief Context creation input.
 *
 */
struct hiprtContextCreationInput
{
	/*!< HIPRT API context */
	hiprtApiCtx ctxt;
	/*!< HIPRT API device */
	hiprtApiDevice device;
	/*!< HIPRT API device type */
	hiprtDeviceType deviceType;
};

/** \brief Various flags controlling scene/geometry build process.
 *
 */
struct hiprtBuildOptions
{
	/*!< Build flags */
	hiprtBuildFlags buildFlags;
};

/** \brief Triangle mesh primitive.
 *
 * Triangle mesh primitive is represented as an indexed vertex array.
 * Vertex and index arrays are defined using device pointers and strides.
 * Each vertex has to have 3 components: (x, y, z) coordinates.
 * Indices are organized into triples (i0, i1, i2) - one for each triangle.
 */
struct hiprtTriangleMeshPrimitive
{
	/*!< Device pointer to vertex data */
	hiprtDevicePtr vertices;
	/*!< Number of vertices in vertex array */
	uint32_t vertexCount;
	/*!< Stride in bytes between two vertices */
	uint32_t vertexStride;

	/*!< Device pointer to index data */
	hiprtDevicePtr triangleIndices;
	/*!< Number of trinagles in index array */
	uint32_t triangleCount;
	/*!< Stride in bytes between two triangles */
	uint32_t triangleStride;
};

/** \brief AABB list primitive.
 *
 * AABB list is an array of axis aligned bounding boxes, represented
 * by device memory pointer and stride between two consequetive boxes.
 * Each AABB is a pair of float4 values (xmin, ymin, zmin, unused), (xmax, ymax,
 * zmax, unused).
 */
struct hiprtAABBListPrimitive
{
	/*!< Device pointer to AABB data */
	hiprtDevicePtr aabbs;
	/*!< Number of AABBs in the array */
	uint32_t aabbCount;
	/*!< Stride in bytes between two AABBs */
	uint32_t aabbStride;
};

/** \brief Bvh node for custom import Bvh.
 *
 */
struct HIPRT_ALIGN( 64 ) hiprtBvhNode
{
	/*!< Child indices (empty slot needs to be marked by hiprtInvalidValue) */
	uint32_t childIndices[4];
	/*!< Child node types */
	hiprtBvhNodeType childNodeTypes[4];
	/*!< Node bounding box min */
	hiprtFloat3 boundingBoxMin;
	/*!< Node bounding box max */
	hiprtFloat3 boundingBoxMax;
};
static_assert( sizeof( hiprtBvhNode ) == 64 );

/** \brief Bvh node list.
 *
 */
struct hiprtBvhNodeList
{
	/*!< Array of hiprtBvhNode's */
	hiprtDevicePtr nodes;
	/*!< Number of nodes */
	uint32_t nodeCount;
};

/** \brief Input for geometry build/update operation.
 *
 * Build input defines concrete primitive type and a pointer to an actual
 * primitive description.
 */
struct hiprtGeometryBuildInput
{
	/*!< Primitive type */
	hiprtPrimitiveType type;
	/*!< Geometry type used for custom function table */
	uint32_t geomType = hiprtInvalidValue;
	/*!< Defines the following union */
	union
	{
		struct
		{
			/*!< Triangle mesh */
			hiprtTriangleMeshPrimitive* primitive;
		} triangleMesh;
		struct
		{
			/*!< Bounding boxes of custom primitives */
			hiprtAABBListPrimitive* primitive;
		} aabbList;
	};
	/*!< Custom Bvh nodes (optional) */
	hiprtBvhNodeList* nodes;
};

/** \brief Build input for the scene.
 *
 * Scene consists of a set of instances. Each of the instances is defined by:
 *  - Root pointer of the corresponding geometry
 *  - Transformation header
 *  - Mask
 *
 * Instances can refer to the same geometry but with different transformations
 * (essentially implementing instancing). Mask is used to implement ray
 * masking: ray mask is bitwise &ded with an instance mask, and no intersections
 * are evaluated with the primitive of corresponding instance if the result is
 * 0. The transformation header defines the offset and the number of consecutive 
 * transformation frames in the frame array for each instance. More than one frame 
 * is interpreted as motion blur. If the transformation headers is NULL, it 
 * assumes one frame per instance. Optionally, it is possible to import a custom 
 * BVH by setting nodes and the corresponding build flag.
 */
struct hiprtSceneBuildInput
{
	/*!< Array of instanceCount pointers to geometries */
	hiprtDevicePtr instanceGeometries;
	/*!< Array of instanceCount transform headers (optional: per object frame assumed if NULL) */
	hiprtDevicePtr instanceTransformHeaders;
	/*!< Array of frameCount frames (supposed to be ordered according to time) */
	hiprtDevicePtr instanceFrames;
	/*!< Per object bit masks for instance masking (optional: if NULL masks treated as hiprtFullRayMask) */
	hiprtDevicePtr instanceMasks;
	/*!< Custom Bvh nodes (optional) */
	hiprtBvhNodeList* nodes;
	/*!< Number of instances */
	uint32_t instanceCount;
	/*!< Number of frames (such that instanceCount <= frameCount) */
	uint32_t frameCount;
	/*!< Frame type (SRT or matrix) */
	hiprtFrameType frameType = hiprtFrameTypeSRT;
};

/** \brief SRT transformation frame.
 *
 * Represented by scale (S), rotation (R), translation (T), and frame time.
 * Object to world transformation is composed as (T * R * S) * x = y
 */
struct HIPRT_ALIGN( 16 ) hiprtFrameSRT
{
	/*!< Rotation (axis and angle) */
	hiprtFloat4 rotation;
	/*!< Scale */
	hiprtFloat3 scale;
	/*!< Translation */
	hiprtFloat3 translation;
	/*!< Frame time */
	float time;
};
static_assert( sizeof( hiprtFrameSRT ) == 48 );

/** \brief Transformation matrix frame representation.
 *
 * Represented by a 3x4 matrix and frame time.
 */
struct HIPRT_ALIGN( 64 ) hiprtFrameMatrix
{
	/*!< Matrix */
	float matrix[3][4];
	/*!< Frame time */
	float time;
};
static_assert( sizeof( hiprtFrameMatrix ) == 64 );

/** \brief Transformation header.
 *
 * Defines defines the index to the array of frames and the number of frames.
 */
struct HIPRT_ALIGN( 8 ) hiprtTransformHeader
{
	/*!< Frame index */
	uint32_t frameIndex;
	/*!< Number of frames */
	uint32_t frameCount;
};
static_assert( sizeof( hiprtTransformHeader ) == 8 );

/** \brief Create HIPRT API context.
 *
 * All HIPRT functions expect context as their first argument. Context
 * keeps global data required by HIPRT session. Calls made from different
 * threads with different HIPRT contexts are safe. Calls with the same context
 * should be externally synchronized by the client.
 *
 * \param hiprtApiVersion API version.
 * \param outContext Created context.
 * \return HIPRT error in case of a failure, hiprtSuccess otherwise.
 */
HIPRT_API hiprtError hiprtCreateContext( uint32_t hiprtApiVersion, hiprtContextCreationInput& input, hiprtContext* outContext );

/** \brief Destory HIPRT API context.
 *
 * Destroys all the global resources used by HIPRT session. Further calls
 * with this context are prohibited.
 *
 * \param context API context.
 * \return HIPRT error in case of a failure, hiprtSuccess otherwise.
 */
HIPRT_API hiprtError hiprtDestroyContext( hiprtContext context );

/** \brief Create a geometry.
 *
 * This function creates
 * hiprtGeometry representing acceleration structure topology.
 *
 * \param context HIPRT API context.
 * \param buildInput Describes input primitive to build geometry from.
 * \param buildOptions Various flags controlling build process.
 * \param outGeometry Resulting geometry.
 * \return HIPRT error in case of a failure, hiprtSuccess otherwise.
 */
HIPRT_API hiprtError hiprtCreateGeometry(
	hiprtContext				   context,
	const hiprtGeometryBuildInput* buildInput,
	const hiprtBuildOptions*	   buildOptions,
	hiprtGeometry*				   outGeometry );

/** \brief Destroy a geometry.
 *
 * This function destroys
 * hiprtGeometry representing acceleration structure topology.
 *
 * \param context HIPRT API context.
 * \param outGeometry Resulting geometry.
 * \return HIPRT error in case of a failure, hiprtSuccess otherwise.
 */
HIPRT_API hiprtError hiprtDestroyGeometry( hiprtContext context, hiprtGeometry outGeometry );

/** \brief Build or update a geometry.
 *
 * Given geometry description from the client, this function builds
 * hiprtGeometry representing acceleration structure topology (in case of a
 * build) or updates acceleration structure keeping topology intact (update).
 *
 * \param context HIPRT API context.
 * \param buildOperation Type of build operation.
 * \param buildInput Describes input primitive to build geometry from.
 * \param buildOptions Various flags controlling build process.
 * \param attributeOutputs Describes additional values written into vidmem.
 * \param attributeOutputCount Number of additional attributes, can be 0.
 * \param temporaryBuffer Temporary buffer for build operation.
 * \param stream to run acceleration structure build command.
 * \param outGeometry Resulting geometry.
 * \return HIPRT error in case of a failure, hiprtSuccess otherwise.
 */
HIPRT_API hiprtError hiprtBuildGeometry(
	hiprtContext				   context,
	hiprtBuildOperation			   buildOperation,
	const hiprtGeometryBuildInput* buildInput,
	const hiprtBuildOptions*	   buildOptions,
	hiprtDevicePtr				   temporaryBuffer,
	hiprtApiStream				   stream,
	hiprtGeometry				   outGeometry );

/** \brief Get temporary storage requirements for geometry build.
 *
 * \param context HIPRT API context.
 * \param buildInput Describes input primitive to build geometry from.
 * \param buildOptions Various flags controlling build process.
 * \param outSize Pointer to write result to.
 * \return HIPRT error in case of a failure, hiprtSuccess otherwise.
 */
HIPRT_API hiprtError hiprtGetGeometryBuildTemporaryBufferSize(
	hiprtContext context, const hiprtGeometryBuildInput* buildInput, const hiprtBuildOptions* buildOptions, size_t* outSize );

/** \brief Create a scene.
 *
 * This function creates
 * hiprtScene representing acceleration structure topology.
 *
 * \param context HIPRT API context.
 * \param buildInput Decribes input geometires to build scene for.
 * \param buildOptions Various flags controlling build process.
 * \param outScene Resulting scene.
 * \return HIPRT error in case of a failure, hiprtSuccess otherwise.
 */
HIPRT_API hiprtError hiprtCreateScene(
	hiprtContext context, const hiprtSceneBuildInput* buildInput, const hiprtBuildOptions* buildOptions, hiprtScene* outScene );

/** \brief Destroy a scene.
 *
 * This function destroys
 * hiprtScene representing acceleration structure topology.
 *
 * \param context HIPRT API context.
 * \param outScene Resulting scene.
 * \return HIPRT error in case of a failure, hiprtSuccess otherwise.
 */
HIPRT_API hiprtError hiprtDestroyScene( hiprtContext context, hiprtScene outScene );

/** \brief Build or update a scene.
 *
 * Given a number of hiprtGeometries from the client, this function builds
 * hiprtScene representing top level acceleration structure topology (in case of
 * a build) or updates acceleration structure keeping topology intact (update).
 *
 * \param context HIPRT API context.
 * \param buildOperation Type of build operation.
 * \param buildInput Decribes input geometires to build scene for.
 * \param buildOptions Various flags controlling build process.
 * \param temporaryBuffer Temporary buffer for build operation.
 * \param stream to run acceleration structure build command.
 * \param outScene Resulting scene.
 * \return HIPRT error in case of a failure, hiprtSuccess otherwise.
 */
HIPRT_API hiprtError hiprtBuildScene(
	hiprtContext				context,
	hiprtBuildOperation			buildOperation,
	const hiprtSceneBuildInput* buildInput,
	const hiprtBuildOptions*	buildOptions,
	hiprtDevicePtr				temporaryBuffer,
	hiprtApiStream				stream,
	hiprtScene					outScene );

/** \brief Get temporary storage requirements for scene build.
 *
 * \param context HIPRT API context.
 * \param buildInput Decribes input geometires to build scene for.
 * \param buildOptions Various flags controlling build process.
 * \param outSize Pointer to write result to.
 * \return HIPRT error in case of a failure, hiprtSuccess otherwise.
 */
HIPRT_API hiprtError hiprtGetSceneBuildTemporaryBufferSize(
	hiprtContext context, const hiprtSceneBuildInput* buildInput, const hiprtBuildOptions* buildOptions, size_t* outSize );

/** \brief Creates a custom function table (for custom geometry).
 *
 * \param context HIPRT API context.
 * \param numGeomTypes The number of geometry types.
 * \param numRayTypes The number of ray types.
 * \param outFuncTable The resulting table.
 * \return HIPRT error in case of a failure, hiprtSuccess otherwise.
 */
HIPRT_API hiprtError hiprtCreateFuncTable( hiprtContext context, uint32_t numGeomTypes, uint32_t numRayTypes, hiprtFuncTable* outFuncTable );

/** \brief Sets a custom function table.
 *
 * \param context HIPRT API context.
 * \param funcTable Function table.
 * \param geomType Geometry type.
 * \param rayType Ray type.
 * \param set Function set to be set.
 * \return HIPRT error in case of a failure, hiprtSuccess otherwise.
 */
HIPRT_API hiprtError hiprtSetFuncTable( hiprtContext context, hiprtFuncTable funcTable, uint32_t geomType, uint32_t rayType, hiprtFuncDataSet set ); 

/** \brief Destroys a custom function table.
 *
 * \param context HIPRT API context.
 * \param funcTable Function table.
 * \return HIPRT error in case of a failure, hiprtSuccess otherwise.
 */
HIPRT_API hiprtError hiprtDestroyFuncTable( hiprtContext context, hiprtFuncTable funcTable );

/** \brief Saves hiprtGeometry to a binary file.
 *
 * \param context HIPRT API context.
 * \param inGeometry Geometry to be saved.
 * \param filename File name with full path.
 * \return HIPRT error in case of a failure, hiprtSuccess otherwise.
 */
HIPRT_API hiprtError hiprtSaveGeometry( hiprtContext context, hiprtGeometry inGeometry, const char* filename );

/** \brief Loads hiprtGeometry to a binary file.
 *
 * \param context HIPRT API context.
 * \param outGeometry Geometry to be loaded.
 * \param filename File name with full path.
 * \return HIPRT error in case of a failure, hiprtSuccess otherwise.
 */
HIPRT_API hiprtError hiprtLoadGeometry( hiprtContext context, hiprtGeometry* outGeometry, const char* filename );

/** \brief Saves hiprtScene to a binary file.
 *
 * \param context HIPRT API context.
 * \param inScene Scene to be saved.
 * \param filename File name with full path.
 * \return HIPRT error in case of a failure, hiprtSuccess otherwise.
 */
HIPRT_API hiprtError hiprtSaveScene( hiprtContext context, hiprtScene inScene, const char* filename );

/** \brief Loads hiprtScene to a binary file.
 *
 * \param context HIPRT API context.
 * \param outScene Scene to be loaded.
 * \param filename File name with full path.
 * \return HIPRT error in case of a failure, hiprtSuccess otherwise.
 */
HIPRT_API hiprtError hiprtLoadScene( hiprtContext context, hiprtScene* outScene, const char* filename );

/** \brief Output scene's AABB.
 *
 * \param context HIPRT API context.
 * \param inGeometry Geometry to be queried.
 * \param outAabbMin The bounding box min. bound.
 * \param outAabbMax The bounding box max. bound.
 * \return HIPRT error in case of a failure, hiprtSuccess otherwise.
 */
HIPRT_API hiprtError hiprtExportGeometryAabb( hiprtContext context, hiprtGeometry inGeometry, hiprtFloat3& outAabbMin, hiprtFloat3& outAabbMax );

/** \brief Output scene's AABB.
 *
 * \param context HIPRT API context.
 * \param inScene Scene to be queried.
 * \param outAabbMin The bounding box min. bound.
 * \param outAabbMax The bounding box max. bound.
 * \return HIPRT error in case of a failure, hiprtSuccess otherwise.
 */
HIPRT_API hiprtError hiprtExportSceneAabb( hiprtContext context, hiprtScene inScene, hiprtFloat3& outAabbMin, hiprtFloat3& outAabbMax );

/** \brief Get Program instance with HIPRT routines.
 * \param context HIPRT API context.
 * \param numFunctions Number of function names.
 * \param functionNames Functions names (explicit instatiation for templated kernels) to which handle will be returned, NULL when numFunctions is 0.
 * \param src HIP program source.
 * \param name Program source filename.
 * \param numHeaders Number of headers, numHeaders must be greater than or equal to 0.
 * \param headers Sources of the headers, NULL when numHeaders is 0.
 * \param includeNames Name of each header by which they can be included in the HIP program source, includeNames can be NULL
 * when numHeaders is 0. 
 * \param numOptions Number of options.
 * \param options Compiler options, can be NULL.
 * \param numGeomTypes Number of geometry types.
 * \param numRayTypes Number of ray types.
 * \param outProg Output build program instance.
 * \param funcNameSets Table of custom function names (numRayTypes x numGeomTypes): 
 * \return HIPRT error in case of a failure, hiprtSuccess otherwise.
 */
HIPRT_API hiprtError hiprtBuildTraceProgram(
	hiprtContext	  context,
	uint32_t		  numFunctions,
	const char**	  functionNames,
	const char*		  src,
	const char*		  name,
	uint32_t		  numHeaders,
	const char**	  headersIn,
	const char**	  includeNamesIn,
	uint32_t		  numOptions,
	const char**	  options,
	uint32_t		  numGeomTypes,
	uint32_t		  numRayTypes,
	hiprtFuncNameSet* funcNameSets,
	void*			  outProg );

/** \brief Get binary with HIPRT routines.
 *
 * \param prog program instance.
 * \param size Output size of binary .
 * \param binary Output if NULL function returns size of parameter else returned binary(application should allocate for binary)..
 * \return HIPRT error in case of a failure, hiprtSuccess otherwise.
 */
HIPRT_API hiprtError hiprtBuildTraceGetBinary( void* prog, size_t* size, void* binary );

/** \brief Setting log level.
 *
 * \param path user defined path to cache kernels.
 */
HIPRT_API void hiprtSetCacheDirPath( 
	const char* path );


/** \brief Setting log level.
 *
 * \param level Desired log level.
 */
HIPRT_API void hiprtSetLogLevel( int level = 0 );

#ifdef __cplusplus
}
#endif

#endif
