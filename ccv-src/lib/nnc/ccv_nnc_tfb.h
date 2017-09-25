/**********************************************************
 * C-based/Cached/Core Computer Vision Library
 * Liu Liu, 2010-02-01
 **********************************************************/

/**********************************************************
 * CCV - Neural Network Collection
 **********************************************************/

/**
 * This header is included into main ccv.h
 * such that enables toll-free bridging between
 * ccv_nnc_tensor_t and ccv_dense_matrix_t
 * In effect, ccv_dense_matrix_t will be a specialized
 * version of ccv_nnc_tensor_t. We are taking some penalties
 * from this change though, namely, the size of ccv_dense_matrix_t
 * will be bigger now.
 */

#ifndef GUARD_ccv_nnc_tfb_h
#define GUARD_ccv_nnc_tfb_h

#include <unistd.h>
#include <stdint.h>

enum {
	CCV_TENSOR_FORMAT_NCHW = 0x01,
	CCV_TENSOR_FORMAT_NHWC = 0x02,
	CCV_TENSOR_FORMAT_CHWN = 0x04,
};

enum {
	CCV_TENSOR_CPU_MEMORY = 0x1,
	CCV_TENSOR_GPU_MEMORY = 0x2,
};

enum {
	CCV_COMPUTE_DEVICE_00 = 0x0000,
	CCV_COMPUTE_DEVICE_01 = 0x0100,
	CCV_COMPUTE_DEVICE_02 = 0x0200,
	CCV_COMPUTE_DEVICE_03 = 0x0300,
	CCV_COMPUTE_DEVICE_04 = 0x0400,
	CCV_COMPUTE_DEVICE_05 = 0x0500,
	CCV_COMPUTE_DEVICE_06 = 0x0600,
	CCV_COMPUTE_DEVICE_07 = 0x0700,
};

#define CCV_TENSOR_GET_MEMORY(type) ((type) & 0x3)
#define CCV_TENSOR_GET_DEVICE(type) ((type) & 0xff00)
#define CCV_TENSOR_GET_DEVICE_ID(type) (CCV_TENSOR_GET_DEVICE(type) >> 8)

enum {
	CCV_TENSOR_VIEW      = 0x01000000,
	CCV_TENSOR_MULTIVIEW = 0x02000000,
};

typedef union ccv_numeric_data_u {
	unsigned char* u8;
	int* i32;
	float* f32;
	int64_t* i64;
	double* f64;
	void* ptr; // Raw pointer
} ccv_numeric_data_t;

#define CCV_NNC_MAX_DIM_ALLOC (8)
#define CCV_NNC_MAX_DIM (2)

typedef struct {
	int type;
	int format;
	int datatype;
	int dim[CCV_NNC_MAX_DIM_ALLOC];
} ccv_nnc_tensor_param_t;

typedef struct {
	int type;
	int refcount;
	ccv_numeric_data_t data;
	uint64_t sig;
	ccv_nnc_tensor_param_t info;
} ccv_nnc_tensor_t;

typedef struct {
	int type;
	int refcount;
	ccv_numeric_data_t data;
	uint64_t sig;
	ccv_nnc_tensor_param_t info;
	/* tensor view and tensor shares the same data structure besides the following one. */
	int inc[CCV_NNC_MAX_DIM_ALLOC]; /**< "increment" or, length */
} ccv_nnc_tensor_view_t;

#define CCV_IS_TENSOR_VIEW(x) ((*(int*)(x)) & CCV_TENSOR_VIEW)
#define CCV_IS_TENSOR_MULTIVIEW(x) ((*(int*)(x)) & CCV_TENSOR_MULTIVIEW)

#if CCV_NNC_TENSOR_TFB
#define CCV_TENSOR_IS_DENSE_MATRIX(x) (((x) & 0xFFF) > 0) // has channel components
typedef struct {
	int type;
	int refcount;
	ccv_numeric_data_t data;
	uint64_t sig;
	// This is used for toll-free bridging between ccv_dense_matrix_t and ccv_nnc_tensor_t
	// Note that this is bigger than it is needed, we carefully structured this
	// such that bit is reused as much, but still some wasted spaces.
	union {
		struct {
			int resides;
			int format;
			int datatype;
			int rows;
			int cols;
			int channels;
			int reserved; /* This reserved bit need to be zero'ed such that later dim is not cared. */
			int step;
			union {
				unsigned char u8;
				int i32;
				float f32;
				int64_t i64;
				double f64;
				void* p;
			} tb;
		};
		ccv_nnc_tensor_param_t info;
	}; 
} ccv_dense_matrix_t;
#else
#define CCV_TENSOR_IS_DENSE_MATRIX(x) (0)
typedef struct {
	int type;
	int refcount;
	uint64_t sig;
	int cols;
	int rows;
	int step;
	union {
		unsigned char u8;
		int i32;
		float f32;
		int64_t i64;
		double f64;
		void* p;
	} tb;
	ccv_numeric_data_t data;
} ccv_dense_matrix_t;
#endif

#endif
