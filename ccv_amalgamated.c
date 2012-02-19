/*
** This file is an amalgamation of many separate C source files from CCV
** version 0.01.  By combining all the individual C code files into this
** single large file, the entire code can be compiled as a single translation
** unit.  This hopefully allows many compilers to do optimizations that would
** not be possible if the files were compiled separately.  Also, this makes
** compilation with XS much easier.
**
#line 1 "ccv/lib/ccv.h"
/**********************************************************
 * C-based/Cached/Core Computer Vision Library
 * Liu Liu, 2010-02-01
 **********************************************************/

#ifndef GUARD_ccv_h
#define GUARD_ccv_h

#ifndef _MSC_VER



#include <unistd.h>
#include <stdint.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <math.h>
#include <xmmintrin.h>
#include <assert.h>
#ifdef _WIN32
#include <malloc.h>
#else
#include <alloca.h>
#endif

#define CCV_PI (3.141592653589793)
#define ccmalloc malloc
#define ccrealloc realloc
#define ccfree free

enum {
	CCV_8U  = 0x0100,
	CCV_32S = 0x0200,
	CCV_32F = 0x0400,
	CCV_64F = 0x0800,
};

enum {
	CCV_C1 = 0x01,
	CCV_C2 = 0x02,
	CCV_C3 = 0x03,
	CCV_C4 = 0x04,
};
/**/
static const int _ccv_get_data_type_size[] = { -1, 1, 4, -1, 4, -1, -1, -1, 8 };

#define CCV_GET_DATA_TYPE(x) ((x) & 0xFF00)
#define CCV_GET_DATA_TYPE_SIZE(x) _ccv_get_data_type_size[CCV_GET_DATA_TYPE(x) >> 8]
#define CCV_GET_CHANNEL(x) ((x) & 0xFF)
#define CCV_ALL_DATA_TYPE (CCV_8U | CCV_32S | CCV_32F | CCV_64F)

enum {
	CCV_MATRIX_DENSE  = 0x010000,
	CCV_MATRIX_SPARSE = 0x020000,
	CCV_MATRIX_CSR    = 0x040000,
	CCV_MATRIX_CSC    = 0x080000,
};

#define CCV_GARBAGE (0x80000000)
#define CCV_REUSABLE (0x40000000)

typedef union {
	unsigned char* ptr;
	int* i;
	float* fl;
	double* db;
} ccv_matrix_cell_t;

typedef struct {
	int type;
	uint64_t sig;
	int refcount;
	int rows;
	int cols;
	int step;
	ccv_matrix_cell_t data;
} ccv_dense_matrix_t;

enum {
	CCV_SPARSE_VECTOR = 0x00100000,
	CCV_DENSE_VECTOR  = 0x00200000,
};

typedef struct ccv_dense_vector_t {
	int step;
	int length;
	int index;
	int prime;
	int load_factor;
	ccv_matrix_cell_t data;
	int* indice;
	struct ccv_dense_vector_t* next;
} ccv_dense_vector_t;

enum {
	CCV_SPARSE_ROW_MAJOR = 0x00,
	CCV_SPARSE_COL_MAJOR = 0x01,
};

typedef struct {
	int type;
	uint64_t sig;
	int refcount;
	int rows;
	int cols;
	int major;
	int prime;
	int load_factor;
	ccv_dense_vector_t* vector;
} ccv_sparse_matrix_t;

extern int _ccv_get_sparse_prime[];
#define CCV_GET_SPARSE_PRIME(x) _ccv_get_sparse_prime[(x)]

typedef void ccv_matrix_t;

/* the explicit cache mechanism */
/* the new cache is radix tree based, but has a strict memory usage upper bound
 * so that you don't have to explicitly call ccv_drain_cache() every time */
typedef void(*ccv_cache_index_free_f)(void*);

typedef union {
	struct {
		uint64_t bitmap;
		uint64_t set;
		uint32_t age;
	} branch;
	struct {
		uint64_t sign;
		uint64_t off;
		uint64_t age_and_size;
	} terminal;
} ccv_cache_index_t;

typedef struct {
	ccv_cache_index_t origin;
	uint32_t rnum;
	uint32_t age;
	size_t up;
	size_t size;
	ccv_cache_index_free_f ffree;
} ccv_cache_t;

#define ccv_cache_return(x, retval) { \
	if ((x)->type & CCV_GARBAGE) { \
		(x)->type &= ~CCV_GARBAGE; \
		return retval; } }

/* I made it as generic as possible */
void ccv_cache_init(ccv_cache_t* cache, ccv_cache_index_free_f ffree, size_t up);
void* ccv_cache_get(ccv_cache_t* cache, uint64_t sign);
int ccv_cache_put(ccv_cache_t* cache, uint64_t sign, void* x, uint32_t size);
void* ccv_cache_out(ccv_cache_t* cache, uint64_t sign);
int ccv_cache_delete(ccv_cache_t* cache, uint64_t sign);
void ccv_cache_cleanup(ccv_cache_t* cache);
void ccv_cache_close(ccv_cache_t* cache);

/* deprecated methods, often these implemented in another way and no longer suitable for newer computer architecture */
/* 0 */

typedef struct {
	int type;
	uint64_t sig;
	int refcount;
	int rows;
	int cols;
	int nnz;
	int* index;
	int* offset;
	ccv_matrix_cell_t data;
} ccv_compressed_sparse_matrix_t;

#define ccv_clamp(x, a, b) (((x) < (a)) ? (a) : (((x) > (b)) ? (b) : (x)))
#define ccv_min(a, b) (((a) < (b)) ? (a) : (b))
#define ccv_max(a, b) (((a) > (b)) ? (a) : (b))

/* matrix operations */
ccv_dense_matrix_t* ccv_dense_matrix_renew(ccv_dense_matrix_t* x, int rows, int cols, int types, int prefer_type, uint64_t sig);
ccv_dense_matrix_t* ccv_dense_matrix_new(int rows, int cols, int type, void* data, uint64_t sig);
ccv_dense_matrix_t ccv_dense_matrix(int rows, int cols, int type, void* data, uint64_t sig);
ccv_sparse_matrix_t* ccv_sparse_matrix_new(int rows, int cols, int type, int major, uint64_t sig);
uint64_t ccv_matrix_generate_signature(const char* msg, int len, uint64_t sig_start, ...);
void ccv_matrix_free_immediately(ccv_matrix_t* mat);
void ccv_matrix_free(ccv_matrix_t* mat);

#define CCV_DEFAULT_CACHE_SIZE (1024 * 1024 * 64)

void ccv_drain_cache(void);
void ccv_disable_cache(void);
void ccv_enable_default_cache(void);
void ccv_enable_cache(size_t size);

#define ccv_get_dense_matrix_cell(x, row, col) \
	((((x)->type) & CCV_32S) ? (void*)((x)->data.i + (row) * (x)->cols + (col)) : \
	((((x)->type) & CCV_32F) ? (void*)((x)->data.fl+ (row) * (x)->cols + (col)) : \
	((((x)->type) & CCV_64F) ? (void*)((x)->data.db + (row) * (x)->cols + (col)) : \
	(void*)((x)->data.ptr + (row) * (x)->step + (col)))))

#define ccv_get_dense_matrix_cell_value(x, row, col) \
	((((x)->type) & CCV_32S) ? (x)->data.i[(row) * (x)->cols + (col)] : \
	((((x)->type) & CCV_32F) ? (x)->data.fl[(row) * (x)->cols + (col)] : \
	((((x)->type) & CCV_64F) ? (x)->data.db[(row) * (x)->cols + (col)] : \
	(x)->data.ptr[(row) * (x)->step + (col)])))

#define ccv_get_value(type, ptr, i) \
	(((type) & CCV_32S) ? ((int*)(ptr))[(i)] : \
	(((type) & CCV_32F) ? ((float*)(ptr))[(i)] : \
	(((type) & CCV_64F) ? ((double*)(ptr))[(i)] : \
	((unsigned char*)(ptr))[(i)])))

#define ccv_set_value(type, ptr, i, value, factor) switch (CCV_GET_DATA_TYPE((type))) { \
	case CCV_32S: ((int*)(ptr))[(i)] = (int)(value) >> factor; break; \
	case CCV_32F: ((float*)(ptr))[(i)] = (float)value; break; \
	case CCV_64F: ((double*)(ptr))[(i)] = (double)value; break; \
	default: ((unsigned char*)(ptr))[(i)] = ccv_clamp((int)(value) >> factor, 0, 255); }


/* unswitch for loop macros */
/* the new added macro in order to do for loop expansion in a way that, you can
 * expand a for loop by inserting different code snippet */
#define ccv_unswitch_block(param, block, ...) { block(__VA_ARGS__, param); }
#define ccv_unswitch_block_a(param, block, ...) { block(__VA_ARGS__, param); }
#define ccv_unswitch_block_b(param, block, ...) { block(__VA_ARGS__, param); }
/* the factor used to provide higher accuracy in integer type (all integer
 * computation in some cases) */
#define _ccv_get_32s_value(ptr, i, factor) (((int*)(ptr))[(i)] << factor)
#define _ccv_get_32f_value(ptr, i, factor) ((float*)(ptr))[(i)]
#define _ccv_get_64f_value(ptr, i, factor) ((double*)(ptr))[(i)]
#define _ccv_get_8u_value(ptr, i, factor) (((unsigned char*)(ptr))[(i)] << factor)
#define ccv_matrix_getter(type, block, ...) { switch (CCV_GET_DATA_TYPE(type)) { \
	case CCV_32S: { block(__VA_ARGS__, _ccv_get_32s_value); break; } \
	case CCV_32F: { block(__VA_ARGS__, _ccv_get_32f_value); break; } \
	case CCV_64F: { block(__VA_ARGS__, _ccv_get_64f_value); break; } \
	default: { block(__VA_ARGS__, _ccv_get_8u_value); } } }

#define ccv_matrix_getter_a(type, block, ...) { switch (CCV_GET_DATA_TYPE(type)) { \
	case CCV_32S: { block(__VA_ARGS__, _ccv_get_32s_value); break; } \
	case CCV_32F: { block(__VA_ARGS__, _ccv_get_32f_value); break; } \
	case CCV_64F: { block(__VA_ARGS__, _ccv_get_64f_value); break; } \
	default: { block(__VA_ARGS__, _ccv_get_8u_value); } } }

#define ccv_matrix_getter_b(type, block, ...) { switch (CCV_GET_DATA_TYPE(type)) { \
	case CCV_32S: { block(__VA_ARGS__, _ccv_get_32s_value); break; } \
	case CCV_32F: { block(__VA_ARGS__, _ccv_get_32f_value); break; } \
	case CCV_64F: { block(__VA_ARGS__, _ccv_get_64f_value); break; } \
	default: { block(__VA_ARGS__, _ccv_get_8u_value); } } }

#define ccv_matrix_typeof(type, block, ...) { switch (CCV_GET_DATA_TYPE(type)) { \
	case CCV_32S: { block(__VA_ARGS__, int); break; } \
	case CCV_32F: { block(__VA_ARGS__, float); break; } \
	case CCV_64F: { block(__VA_ARGS__, double); break; } \
	default: { block(__VA_ARGS__, unsigned char); } } }

#define ccv_matrix_typeof_a(type, block, ...) { switch (CCV_GET_DATA_TYPE(type)) { \
	case CCV_32S: { block(__VA_ARGS__, int); break; } \
	case CCV_32F: { block(__VA_ARGS__, float); break; } \
	case CCV_64F: { block(__VA_ARGS__, double); break; } \
	default: { block(__VA_ARGS__, unsigned char); } } }

#define ccv_matrix_typeof_b(type, block, ...) { switch (CCV_GET_DATA_TYPE(type)) { \
	case CCV_32S: { block(__VA_ARGS__, int); break; } \
	case CCV_32F: { block(__VA_ARGS__, float); break; } \
	case CCV_64F: { block(__VA_ARGS__, double); break; } \
	default: { block(__VA_ARGS__, unsigned char); } } }

#define _ccv_set_32s_value(ptr, i, value, factor) (((int*)(ptr))[(i)] = (int)(value) >> factor)
#define _ccv_set_32f_value(ptr, i, value, factor) (((float*)(ptr))[(i)] = (float)(value))
#define _ccv_set_64f_value(ptr, i, value, factor) (((double*)(ptr))[(i)] = (double)(value))
#define _ccv_set_8u_value(ptr, i, value, factor) (((unsigned char*)(ptr))[(i)] = ccv_clamp((int)(value) >> factor, 0, 255))
#define ccv_matrix_setter(type, block, ...) { switch (CCV_GET_DATA_TYPE(type)) { \
	case CCV_32S: { block(__VA_ARGS__, _ccv_set_32s_value); break; } \
	case CCV_32F: { block(__VA_ARGS__, _ccv_set_32f_value); break; } \
	case CCV_64F: { block(__VA_ARGS__, _ccv_set_64f_value); break; } \
	default: { block(__VA_ARGS__, _ccv_set_8u_value); } } }

#define ccv_matrix_setter_a(type, block, ...) { switch (CCV_GET_DATA_TYPE(type)) { \
	case CCV_32S: { block(__VA_ARGS__, _ccv_set_32s_value); break; } \
	case CCV_32F: { block(__VA_ARGS__, _ccv_set_32f_value); break; } \
	case CCV_64F: { block(__VA_ARGS__, _ccv_set_64f_value); break; } \
	default: { block(__VA_ARGS__, _ccv_set_8u_value); } } }

#define ccv_matrix_setter_b(type, block, ...) { switch (CCV_GET_DATA_TYPE(type)) { \
	case CCV_32S: { block(__VA_ARGS__, _ccv_set_32s_value); break; } \
	case CCV_32F: { block(__VA_ARGS__, _ccv_set_32f_value); break; } \
	case CCV_64F: { block(__VA_ARGS__, _ccv_set_64f_value); break; } \
	default: { block(__VA_ARGS__, _ccv_set_8u_value); } } }

#define ccv_matrix_setter_getter(type, block, ...) { switch (CCV_GET_DATA_TYPE(type)) { \
	case CCV_32S: { block(__VA_ARGS__, _ccv_set_32s_value, _ccv_get_32s_value); break; } \
	case CCV_32F: { block(__VA_ARGS__, _ccv_set_32f_value, _ccv_get_32f_value); break; } \
	case CCV_64F: { block(__VA_ARGS__, _ccv_set_64f_value, _ccv_get_64f_value); break; } \
	default: { block(__VA_ARGS__, _ccv_set_8u_value, _ccv_get_8u_value); } } }

#define ccv_matrix_setter_getter_a(type, block, ...) { switch (CCV_GET_DATA_TYPE(type)) { \
	case CCV_32S: { block(__VA_ARGS__, _ccv_set_32s_value, _ccv_get_32s_value); break; } \
	case CCV_32F: { block(__VA_ARGS__, _ccv_set_32f_value, _ccv_get_32f_value); break; } \
	case CCV_64F: { block(__VA_ARGS__, _ccv_set_64f_value, _ccv_get_64f_value); break; } \
	default: { block(__VA_ARGS__, _ccv_set_8u_value, _ccv_get_8u_value); } } }

#define ccv_matrix_setter_getter_b(type, block, ...) { switch (CCV_GET_DATA_TYPE(type)) { \
	case CCV_32S: { block(__VA_ARGS__, _ccv_set_32s_value, _ccv_get_32s_value); break; } \
	case CCV_32F: { block(__VA_ARGS__, _ccv_set_32f_value, _ccv_get_32f_value); break; } \
	case CCV_64F: { block(__VA_ARGS__, _ccv_set_64f_value, _ccv_get_64f_value); break; } \
	default: { block(__VA_ARGS__, _ccv_set_8u_value, _ccv_get_8u_value); } } }

#define ccv_matrix_typeof_getter(type, block, ...) { switch (CCV_GET_DATA_TYPE(type)) { \
	case CCV_32S: { block(__VA_ARGS__, int, _ccv_get_32s_value); break; } \
	case CCV_32F: { block(__VA_ARGS__, float, _ccv_get_32f_value); break; } \
	case CCV_64F: { block(__VA_ARGS__, double, _ccv_get_64f_value); break; } \
	default: { block(__VA_ARGS__, unsigned char, _ccv_get_8u_value); } } }

#define ccv_matrix_typeof_getter_a(type, block, ...) { switch (CCV_GET_DATA_TYPE(type)) { \
	case CCV_32S: { block(__VA_ARGS__, int, _ccv_get_32s_value); break; } \
	case CCV_32F: { block(__VA_ARGS__, float, _ccv_get_32f_value); break; } \
	case CCV_64F: { block(__VA_ARGS__, double, _ccv_get_64f_value); break; } \
	default: { block(__VA_ARGS__, unsigned char, _ccv_get_8u_value); } } }

#define ccv_matrix_typeof_getter_b(type, block, ...) { switch (CCV_GET_DATA_TYPE(type)) { \
	case CCV_32S: { block(__VA_ARGS__, int, _ccv_get_32s_value); break; } \
	case CCV_32F: { block(__VA_ARGS__, float, _ccv_get_32f_value); break; } \
	case CCV_64F: { block(__VA_ARGS__, double, _ccv_get_64f_value); break; } \
	default: { block(__VA_ARGS__, unsigned char, _ccv_get_8u_value); } } }

#define ccv_matrix_typeof_setter(type, block, ...) { switch (CCV_GET_DATA_TYPE(type)) { \
	case CCV_32S: { block(__VA_ARGS__, int, _ccv_set_32s_value); break; } \
	case CCV_32F: { block(__VA_ARGS__, float, _ccv_set_32f_value); break; } \
	case CCV_64F: { block(__VA_ARGS__, double, _ccv_set_64f_value); break; } \
	default: { block(__VA_ARGS__, unsigned char, _ccv_set_8u_value); } } }

#define ccv_matrix_typeof_setter_a(type, block, ...) { switch (CCV_GET_DATA_TYPE(type)) { \
	case CCV_32S: { block(__VA_ARGS__, int, _ccv_set_32s_value); break; } \
	case CCV_32F: { block(__VA_ARGS__, float, _ccv_set_32f_value); break; } \
	case CCV_64F: { block(__VA_ARGS__, double, _ccv_set_64f_value); break; } \
	default: { block(__VA_ARGS__, unsigned char, _ccv_set_8u_value); } } }

#define ccv_matrix_typeof_setter_b(type, block, ...) { switch (CCV_GET_DATA_TYPE(type)) { \
	case CCV_32S: { block(__VA_ARGS__, int, _ccv_set_32s_value); break; } \
	case CCV_32F: { block(__VA_ARGS__, float, _ccv_set_32f_value); break; } \
	case CCV_64F: { block(__VA_ARGS__, double, _ccv_set_64f_value); break; } \
	default: { block(__VA_ARGS__, unsigned char, _ccv_set_8u_value); } } }

#define ccv_matrix_typeof_setter_getter(type, block, ...) { switch (CCV_GET_DATA_TYPE(type)) { \
	case CCV_32S: { block(__VA_ARGS__, int, _ccv_set_32s_value, _ccv_get_32s_value); break; } \
	case CCV_32F: { block(__VA_ARGS__, float, _ccv_set_32f_value, _ccv_get_32f_value); break; } \
	case CCV_64F: { block(__VA_ARGS__, double, _ccv_set_64f_value, _ccv_get_64f_value); break; } \
	default: { block(__VA_ARGS__, unsigned char, _ccv_set_8u_value, _ccv_get_8u_value); } } }

#define ccv_matrix_typeof_setter_getter_a(type, block, ...) { switch (CCV_GET_DATA_TYPE(type)) { \
	case CCV_32S: { block(__VA_ARGS__, int, _ccv_set_32s_value, _ccv_get_32s_value); break; } \
	case CCV_32F: { block(__VA_ARGS__, float, _ccv_set_32f_value, _ccv_get_32f_value); break; } \
	case CCV_64F: { block(__VA_ARGS__, double, _ccv_set_64f_value, _ccv_get_64f_value); break; } \
	default: { block(__VA_ARGS__, unsigned char, _ccv_set_8u_value, _ccv_get_8u_value); } } }

#define ccv_matrix_typeof_setter_getter_b(type, block, ...) { switch (CCV_GET_DATA_TYPE(type)) { \
	case CCV_32S: { block(__VA_ARGS__, int, _ccv_set_32s_value, _ccv_get_32s_value); break; } \
	case CCV_32F: { block(__VA_ARGS__, float, _ccv_set_32f_value, _ccv_get_32f_value); break; } \
	case CCV_64F: { block(__VA_ARGS__, double, _ccv_set_64f_value, _ccv_get_64f_value); break; } \
	default: { block(__VA_ARGS__, unsigned char, _ccv_set_8u_value, _ccv_get_8u_value); } } }

/* basic io */
enum {
	CCV_SERIAL_GRAY           = 0x100,
	CCV_SERIAL_COLOR          = 0x300,
	CCV_SERIAL_ANY_STREAM     = 0x010,
	CCV_SERIAL_PLAIN_STREAM   = 0x011,
	CCV_SERIAL_DEFLATE_STREAM = 0x012,
	CCV_SERIAL_JPEG_STREAM    = 0x013,
	CCV_SERIAL_PNG_STREAM     = 0x014,
	CCV_SERIAL_ANY_FILE       = 0x020,
	CCV_SERIAL_BMP_FILE       = 0x021,
	CCV_SERIAL_JPEG_FILE      = 0x022,
	CCV_SERIAL_PNG_FILE       = 0x023,
	CCV_SERIAL_BINARY_FILE    = 0x024,
};

enum {
	CCV_SERIAL_CONTINUE = 0x01,
	CCV_SERIAL_FINAL,
	CCV_SERIAL_ERROR,
};

void ccv_unserialize(const char* in, ccv_dense_matrix_t** x, int type);
int ccv_serialize(ccv_dense_matrix_t* mat, char* out, int* len, int type, void* conf);

/* basic algebra algorithm */
double ccv_trace(ccv_matrix_t* mat);

enum {
	CCV_L2_NORM = 0x01,
	CCV_L1_NORM = 0x02,
};

double ccv_norm(ccv_matrix_t* mat, int type);
double ccv_normalize(ccv_matrix_t* a, ccv_matrix_t** b, int btype, int l_type);
double ccv_dot(ccv_matrix_t* a, ccv_matrix_t* b);
double ccv_sum(ccv_matrix_t* mat);
void ccv_zero(ccv_matrix_t* mat);
void ccv_shift(ccv_matrix_t* a, ccv_matrix_t** b, int type, int lr, int rr);
void ccv_substract(ccv_matrix_t* a, ccv_matrix_t* b, ccv_matrix_t** c, int type);

enum {
	CCV_A_TRANSPOSE = 0x01,
	CCV_B_TRANSPOSE = 0X02,
	CCV_C_TRANSPOSE = 0X04,
};

void ccv_gemm(ccv_matrix_t* a, ccv_matrix_t* b, double alpha, ccv_matrix_t* c, double beta, int transpose, ccv_matrix_t** d, int type);

/* matrix build blocks */
ccv_dense_matrix_t* ccv_get_dense_matrix(ccv_matrix_t* mat);
ccv_sparse_matrix_t* ccv_get_sparse_matrix(ccv_matrix_t* mat);
ccv_dense_vector_t* ccv_get_sparse_matrix_vector(ccv_sparse_matrix_t* mat, int index);
ccv_matrix_cell_t ccv_get_sparse_matrix_cell(ccv_sparse_matrix_t* mat, int row, int col);
void ccv_set_sparse_matrix_cell(ccv_sparse_matrix_t* mat, int row, int col, void* data);
void ccv_compress_sparse_matrix(ccv_sparse_matrix_t* mat, ccv_compressed_sparse_matrix_t** csm);
void ccv_decompress_sparse_matrix(ccv_compressed_sparse_matrix_t* csm, ccv_sparse_matrix_t** smt);
void ccv_move(ccv_matrix_t* a, ccv_matrix_t** b, int btype, int y, int x);
int ccv_matrix_eq(ccv_matrix_t* a, ccv_matrix_t* b);
void ccv_slice(ccv_matrix_t* a, ccv_matrix_t** b, int type, int y, int x, int rows, int cols);

/* basic data structures */

typedef struct {
	int width;
	int height;
} ccv_size_t;

inline static ccv_size_t ccv_size(int width, int height)
{
	ccv_size_t size;
	size.width = width;
	size.height = height;
	return size;
}

typedef struct {
	int x;
	int y;
	int width;
	int height;
} ccv_rect_t;

inline static ccv_rect_t ccv_rect(int x, int y, int width, int height)
{
	ccv_rect_t rect;
	rect.x = x;
	rect.y = y;
	rect.width = width;
	rect.height = height;
	return rect;
}

typedef struct {
	int rnum;
	int size;
	int rsize;
	void* data;
} ccv_array_t;

ccv_array_t* ccv_array_new(int rnum, int rsize);
void ccv_array_push(ccv_array_t* array, void* r);
typedef int(*ccv_array_group_f)(const void*, const void*, void*);
int ccv_array_group(ccv_array_t* array, ccv_array_t** index, ccv_array_group_f gfunc, void* data);
void ccv_array_zero(ccv_array_t* array);
void ccv_array_clear(ccv_array_t* array);
void ccv_array_free(ccv_array_t* array);

#define ccv_array_get(a, i) (((char*)((a)->data)) + (a)->rsize * (i))

typedef struct {
	int x, y;
} ccv_point_t;

inline static ccv_point_t ccv_point(int x, int y)
{
	ccv_point_t point;
	point.x = x;
	point.y = y;
	return point;
}

typedef struct {
	ccv_rect_t rect;
	int size;
	ccv_array_t* set;
	long m10, m01, m11, m20, m02;
} ccv_contour_t;

ccv_contour_t* ccv_contour_new(int set);
void ccv_contour_push(ccv_contour_t* contour, ccv_point_t point);
void ccv_contour_free(ccv_contour_t* contour);
/* range: exlusive, return value: inclusive (i.e., threshold = 5, 0~5 is background, 6~range-1 is foreground */
int ccv_otsu(ccv_dense_matrix_t* a, double* outvar, int range);

/* numerical algorithms */
/* clarification about algebra and numerical algorithms:
 * when using the word "algebra", I assume the operation is well established in Mathematic sense
 * and can be calculated with a straight-forward, finite sequence of operation. The "numerical"
 * in other word, refer to a class of algorithm that can only approximate/or iteratively found the
 * solution. Thus, "invert" would be classified as numercial because of the sense that in some case,
 * it can only be "approximate" (in least-square sense), so to "solve". */
void ccv_invert(ccv_matrix_t* a, ccv_matrix_t** b, int type);
void ccv_solve(ccv_matrix_t* a, ccv_matrix_t* b, ccv_matrix_t** d, int type);
void ccv_eigen(ccv_matrix_t* a, ccv_matrix_t* b, ccv_matrix_t** d, int type);

typedef struct {
	double interp;
	double extrap;
	int max_iter;
	double ratio;
	double rho;
	double sig;
} ccv_minimize_param_t;

typedef int(*ccv_minimize_f)(const ccv_dense_matrix_t* x, double* f, ccv_dense_matrix_t* df, void*);
void ccv_minimize(ccv_dense_matrix_t* x, int length, double red, ccv_minimize_f func, ccv_minimize_param_t params, void* data);

void ccv_filter(ccv_matrix_t* a, ccv_matrix_t* b, ccv_matrix_t** d, int type);
typedef double(*ccv_filter_kernel_f)(double x, double y, void*);
void ccv_filter_kernel(ccv_dense_matrix_t* x, ccv_filter_kernel_f func, void* data);

/* modern numerical algorithms */
void ccv_sparse_coding(ccv_matrix_t* x, int k, ccv_matrix_t** A, int typeA, ccv_matrix_t** y, int typey);
void ccv_compressive_sensing_reconstruct(ccv_matrix_t* a, ccv_matrix_t* x, ccv_matrix_t** y, int type);

/* basic computer vision algorithms / or build blocks */
void ccv_sobel(ccv_dense_matrix_t* a, ccv_dense_matrix_t** b, int type, int dx, int dy);
void ccv_gradient(ccv_dense_matrix_t* a, ccv_dense_matrix_t** theta, int ttype, ccv_dense_matrix_t** m, int mtype, int dx, int dy);
void ccv_hog(ccv_dense_matrix_t* a, ccv_dense_matrix_t** b, int type, int size);
void ccv_canny(ccv_dense_matrix_t* a, ccv_dense_matrix_t** b, int type, int size, double low_thresh, double high_thresh);

enum {
	CCV_INTER_AREA   = 0x01,
	CCV_INTER_LINEAR = 0X02,
	CCV_INTER_CUBIC  = 0X03,
	CCV_INTER_LACZOS = 0X04,
};

void ccv_resample(ccv_dense_matrix_t* a, ccv_dense_matrix_t** b, int btype, int rows, int cols, int type);
void ccv_sample_down(ccv_dense_matrix_t* a, ccv_dense_matrix_t** b, int type, int src_x, int src_y);
void ccv_sample_up(ccv_dense_matrix_t* a, ccv_dense_matrix_t** b, int type, int src_x, int src_y);

enum {
	CCV_FLIP_X = 0x01,
	CCV_FLIP_Y = 0x02,
};

void ccv_flip(ccv_dense_matrix_t* a, ccv_dense_matrix_t** b, int btype, int type);
void ccv_blur(ccv_dense_matrix_t* a, ccv_dense_matrix_t** b, int type, double sigma);

/* modern computer vision algorithms */
/* SIFT, DAISY, SWT, MSER, DPM, BBF, SGF, SSD, FAST */

/* daisy related methods */
typedef struct {
	double radius;
	int rad_q_no;
	int th_q_no;
	int hist_th_q_no;
	float normalize_threshold;
	int normalize_method;
} ccv_daisy_param_t;

enum {
	CCV_DAISY_NORMAL_PARTIAL = 0x01,
	CCV_DAISY_NORMAL_FULL    = 0x02,
	CCV_DAISY_NORMAL_SIFT    = 0x03,
};

void ccv_daisy(ccv_dense_matrix_t* a, ccv_dense_matrix_t** b, int type, ccv_daisy_param_t params);

/* sift related methods */
typedef struct {
	float x, y;
	int octave;
	int level;
	union {
		struct {
			double a, b;
			double c, d;
		} affine;
		struct {
			double scale;
			double angle;
		} regular;
	};
} ccv_keypoint_t;

typedef struct {
	int up2x;
	int noctaves;
	int nlevels;
	float edge_threshold;
	float peak_threshold;
	float norm_threshold;
} ccv_sift_param_t;

void ccv_sift(ccv_dense_matrix_t* a, ccv_array_t** keypoints, ccv_dense_matrix_t** desc, int type, ccv_sift_param_t params);

/* swt related method: stroke width transform is relatively new, typically used in text detection */
typedef struct {
	int direction;
	/* canny parameters */
	int size;
	double low_thresh;
	double high_thresh;
	/* geometry filtering parameters */
	int max_height;
	int min_height;
	double aspect_ratio;
	double variance_ratio;
	/* grouping parameters */
	double thickness_ratio;
	double height_ratio;
	int intensity_thresh;
	double distance_ratio;
	double intersect_ratio;
	double elongate_ratio;
	int letter_thresh;
	/* break textline into words */
	int breakdown;
	double breakdown_ratio;
} ccv_swt_param_t;

void ccv_swt(ccv_dense_matrix_t* a, ccv_dense_matrix_t** b, int type, ccv_swt_param_t params);
ccv_array_t* ccv_swt_detect_words(ccv_dense_matrix_t* a, ccv_swt_param_t params);

/* I'd like to include Deformable Part Models as a general object detection method in here
 * The difference between BBF and DPM:
 * ~ BBF is for rigid object detection: banners, box, faces etc.
 * ~ DPM is more generalized, can detect people, car, bike (larger inner-class difference) etc.
 * ~ BBF is blazing fast (few milliseconds), DPM is relatively slow (around 1 seconds or so) */

typedef struct {
	ccv_rect_t rect;
	int neighbors;
	int id;
	float confidence;
} ccv_comp_t;

typedef struct {
	float* w;
	float d[4];
	int count;
	int x, y, z;
	ccv_size_t size;
} ccv_dpm_part_classifier_t;

typedef struct {
	int count;
	ccv_dpm_part_classifier_t root;
	ccv_dpm_part_classifier_t* part;
	float beta;
} ccv_dpm_root_classifier_t;

typedef struct {
	int interval;
	int min_neighbors;
	int flags;
	ccv_size_t size;
} ccv_dpm_param_t;

typedef struct {
} ccv_dpm_new_param_t;

ccv_dpm_root_classifier_t* ccv_load_dpm_root_classifier(const char* directory);
ccv_array_t* ccv_dpm_detect_objects(ccv_dense_matrix_t* a, ccv_dpm_root_classifier_t** classifier, int count, ccv_dpm_param_t params);

/* this is open source implementation of object detection algorithm: brightness binary feature
 * it is an extension/modification of original HAAR-like feature with Adaboost, featured faster
 * computation and higher accuracy (current highest accuracy close-source face detector is based
 * on the same algorithm) */

#define CCV_BBF_POINT_MAX (8)
#define CCV_BBF_POINT_MIN (3)

typedef struct {
	int size;
	int px[CCV_BBF_POINT_MAX];
	int py[CCV_BBF_POINT_MAX];
	int pz[CCV_BBF_POINT_MAX];
	int nx[CCV_BBF_POINT_MAX];
	int ny[CCV_BBF_POINT_MAX];
	int nz[CCV_BBF_POINT_MAX];
} ccv_bbf_feature_t;

typedef struct {
	int count;
	float threshold;
	ccv_bbf_feature_t* feature;
	float* alpha;
} ccv_bbf_stage_classifier_t;

typedef struct {
	int count;
	ccv_size_t size;
	ccv_bbf_stage_classifier_t* stage_classifier;
} ccv_bbf_classifier_cascade_t;

enum {
	CCV_BBF_GENETIC_OPT = 0x01,
	CCV_BBF_FLOAT_OPT = 0x02
};

typedef struct {
	int interval;
	int min_neighbors;
	int flags;
	ccv_size_t size;
} ccv_bbf_param_t;

typedef struct {
	double pos_crit;
	double neg_crit;
	double balance_k;
	int layer;
	int feature_number;
	int optimizer;
	ccv_bbf_param_t detector;
} ccv_bbf_new_param_t;

enum {
	CCV_BBF_NO_NESTED = 0x10000000,
};

void ccv_bbf_classifier_cascade_new(ccv_dense_matrix_t** posimg, int posnum, char** bgfiles, int bgnum, int negnum, ccv_size_t size, const char* dir, ccv_bbf_new_param_t params);
ccv_array_t* ccv_bbf_detect_objects(ccv_dense_matrix_t* a, ccv_bbf_classifier_cascade_t** cascade, int count, ccv_bbf_param_t params);
ccv_bbf_classifier_cascade_t* ccv_load_bbf_classifier_cascade(const char* directory);
ccv_bbf_classifier_cascade_t* ccv_bbf_classifier_cascade_read_binary(char* s);
int ccv_bbf_classifier_cascade_write_binary(ccv_bbf_classifier_cascade_t* cascade, char* s, int slen);
void ccv_bbf_classifier_cascade_free(ccv_bbf_classifier_cascade_t* cascade);

/* following is proprietary implementation of sparse gradient feature, another object detection algorithm
 * which should have better accuracy to shape focused object (pedestrian, vehicle etc.) but still as fast as bbf */

#define CCV_SGF_POINT_MAX (5)
#define CCV_SGF_POINT_MIN (3)

typedef struct {
	int size;
	int px[CCV_SGF_POINT_MAX];
	int py[CCV_SGF_POINT_MAX];
	int pz[CCV_SGF_POINT_MAX];
	int nx[CCV_SGF_POINT_MAX];
	int ny[CCV_SGF_POINT_MAX];
	int nz[CCV_SGF_POINT_MAX];
} ccv_sgf_feature_t;

typedef struct {
	int count;
	float threshold;
	ccv_sgf_feature_t* feature;
	float* alpha;
} ccv_sgf_stage_classifier_t;

typedef struct {
	int count;
	ccv_size_t size;
	ccv_sgf_stage_classifier_t* stage_classifier;
} ccv_sgf_classifier_cascade_t;

typedef struct {
	double pos_crit;
	double neg_crit;
	double balance_k;
	int layer;
	int feature_number;
} ccv_sgf_param_t;

typedef struct {
	ccv_rect_t rect;
	int neighbors;
	int id;
	float confidence;
} ccv_sgf_comp_t;

enum {
	CCV_SGF_NO_NESTED = 0x10000000,
};

void ccv_sgf_classifier_cascade_new(ccv_dense_matrix_t** posimg, int posnum, char** bgfiles, int bgnum, int negnum, ccv_size_t size, const char* dir, ccv_sgf_param_t params);
ccv_array_t* ccv_sgf_detect_objects(ccv_dense_matrix_t* a, ccv_sgf_classifier_cascade_t** _cascade, int count, int min_neighbors, int flags, ccv_size_t min_size);
ccv_sgf_classifier_cascade_t* ccv_load_sgf_classifier_cascade(const char* directory);
ccv_sgf_classifier_cascade_t* ccv_sgf_classifier_cascade_read_binary(char* s);
int ccv_sgf_classifier_cascade_write_binary(ccv_sgf_classifier_cascade_t* cascade, char* s, int slen);
void ccv_sgf_classifier_cascade_free(ccv_sgf_classifier_cascade_t* cascade);

/* modern machine learning algorithms */
/* RBM, LLE, APCluster */

/****************************************************************************************\

  Generic implementation of QuickSort algorithm.
  ----------------------------------------------
  Using this macro user can declare customized sort function that can be much faster
  than built-in qsort function because of lower overhead on elements
  comparison and exchange. The macro takes less_than (or LT) argument - a macro or function
  that takes 2 arguments returns non-zero if the first argument should be before the second
  one in the sorted sequence and zero otherwise.

  Example:

    Suppose that the task is to sort points by ascending of y coordinates and if
    y's are equal x's should ascend.

    The code is:
    ------------------------------------------------------------------------------
           #define cmp_pts( pt1, pt2 ) \
               ((pt1).y < (pt2).y || ((pt1).y < (pt2).y && (pt1).x < (pt2).x))

           [static] CV_IMPLEMENT_QSORT( icvSortPoints, CvPoint, cmp_pts )
    ------------------------------------------------------------------------------

    After that the function "void icvSortPoints( CvPoint* array, size_t total, int aux );"
    is available to user.

  aux is an additional parameter, which can be used when comparing elements.
  The current implementation was derived from *BSD system qsort():

    * Copyright (c) 1992, 1993
    *  The Regents of the University of California.  All rights reserved.
    *
    * Redistribution and use in source and binary forms, with or without
    * modification, are permitted provided that the following conditions
    * are met:
    * 1. Redistributions of source code must retain the above copyright
    *    notice, this list of conditions and the following disclaimer.
    * 2. Redistributions in binary form must reproduce the above copyright
    *    notice, this list of conditions and the following disclaimer in the
    *    documentation and/or other materials provided with the distribution.
    * 3. All advertising materials mentioning features or use of this software
    *    must display the following acknowledgement:
    *  This product includes software developed by the University of
    *  California, Berkeley and its contributors.
    * 4. Neither the name of the University nor the names of its contributors
    *    may be used to endorse or promote products derived from this software
    *    without specific prior written permission.
    *
    * THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS ``AS IS'' AND
    * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
    * ARE DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE
    * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
    * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
    * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
    * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
    * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
    * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
    * SUCH DAMAGE.

\****************************************************************************************/

#define CCV_SWAP(a,b,t) ((t) = (a), (a) = (b), (b) = (t))

#define CCV_IMPLEMENT_QSORT_EX(func_name, T, LT, swap_func, user_data_type)                     \
void func_name(T *array, size_t total, user_data_type aux)                                      \
{                                                                                               \
    int isort_thresh = 7;                                                                       \
    T t;                                                                                        \
    int sp = 0;                                                                                 \
                                                                                                \
    struct                                                                                      \
    {                                                                                           \
        T *lb;                                                                                  \
        T *ub;                                                                                  \
    }                                                                                           \
    stack[48];                                                                                  \
                                                                                                \
    if( total <= 1 )                                                                            \
        return;                                                                                 \
                                                                                                \
    stack[0].lb = array;                                                                        \
    stack[0].ub = array + (total - 1);                                                          \
                                                                                                \
    while( sp >= 0 )                                                                            \
    {                                                                                           \
        T* left = stack[sp].lb;                                                                 \
        T* right = stack[sp--].ub;                                                              \
                                                                                                \
        for(;;)                                                                                 \
        {                                                                                       \
            int i, n = (int)(right - left) + 1, m;                                              \
            T* ptr;                                                                             \
            T* ptr2;                                                                            \
                                                                                                \
            if( n <= isort_thresh )                                                             \
            {                                                                                   \
            insert_sort:                                                                        \
                for( ptr = left + 1; ptr <= right; ptr++ )                                      \
                {                                                                               \
                    for( ptr2 = ptr; ptr2 > left && LT(ptr2[0],ptr2[-1], aux); ptr2--)          \
                        swap_func( ptr2[0], ptr2[-1], array, aux, t );                          \
                }                                                                               \
                break;                                                                          \
            }                                                                                   \
            else                                                                                \
            {                                                                                   \
                T* left0;                                                                       \
                T* left1;                                                                       \
                T* right0;                                                                      \
                T* right1;                                                                      \
                T* pivot;                                                                       \
                T* a;                                                                           \
                T* b;                                                                           \
                T* c;                                                                           \
                int swap_cnt = 0;                                                               \
                                                                                                \
                left0 = left;                                                                   \
                right0 = right;                                                                 \
                pivot = left + (n/2);                                                           \
                                                                                                \
                if( n > 40 )                                                                    \
                {                                                                               \
                    int d = n / 8;                                                              \
                    a = left, b = left + d, c = left + 2*d;                                     \
                    left = LT(*a, *b, aux) ? (LT(*b, *c, aux) ? b : (LT(*a, *c, aux) ? c : a))  \
                                      : (LT(*c, *b, aux) ? b : (LT(*a, *c, aux) ? a : c));      \
                                                                                                \
                    a = pivot - d, b = pivot, c = pivot + d;                                    \
                    pivot = LT(*a, *b, aux) ? (LT(*b, *c, aux) ? b : (LT(*a, *c, aux) ? c : a)) \
                                      : (LT(*c, *b, aux) ? b : (LT(*a, *c, aux) ? a : c));      \
                                                                                                \
                    a = right - 2*d, b = right - d, c = right;                                  \
                    right = LT(*a, *b, aux) ? (LT(*b, *c, aux) ? b : (LT(*a, *c, aux) ? c : a)) \
                                      : (LT(*c, *b, aux) ? b : (LT(*a, *c, aux) ? a : c));      \
                }                                                                               \
                                                                                                \
                a = left, b = pivot, c = right;                                                 \
                pivot = LT(*a, *b, aux) ? (LT(*b, *c, aux) ? b : (LT(*a, *c, aux) ? c : a))     \
                                   : (LT(*c, *b, aux) ? b : (LT(*a, *c, aux) ? a : c));         \
                if( pivot != left0 )                                                            \
                {                                                                               \
                    swap_func( *pivot, *left0, array, aux, t );                                 \
                    pivot = left0;                                                              \
                }                                                                               \
                left = left1 = left0 + 1;                                                       \
                right = right1 = right0;                                                        \
                                                                                                \
                for(;;)                                                                         \
                {                                                                               \
                    while( left <= right && !LT(*pivot, *left, aux) )                           \
                    {                                                                           \
                        if( !LT(*left, *pivot, aux) )                                           \
                        {                                                                       \
                            if( left > left1 )                                                  \
                                swap_func( *left1, *left, array, aux, t );                      \
                            swap_cnt = 1;                                                       \
                            left1++;                                                            \
                        }                                                                       \
                        left++;                                                                 \
                    }                                                                           \
                                                                                                \
                    while( left <= right && !LT(*right, *pivot, aux) )                          \
                    {                                                                           \
                        if( !LT(*pivot, *right, aux) )                                          \
                        {                                                                       \
                            if( right < right1 )                                                \
                                swap_func( *right1, *right, array, aux, t );                    \
                            swap_cnt = 1;                                                       \
                            right1--;                                                           \
                        }                                                                       \
                        right--;                                                                \
                    }                                                                           \
                                                                                                \
                    if( left > right )                                                          \
                        break;                                                                  \
                    swap_func( *left, *right, array, aux, t );                                  \
                    swap_cnt = 1;                                                               \
                    left++;                                                                     \
                    right--;                                                                    \
                }                                                                               \
                                                                                                \
                if( swap_cnt == 0 )                                                             \
                {                                                                               \
                    left = left0, right = right0;                                               \
                    goto insert_sort;                                                           \
                }                                                                               \
                                                                                                \
                n = ccv_min( (int)(left1 - left0), (int)(left - left1) );                       \
                for( i = 0; i < n; i++ )                                                        \
                    swap_func( left0[i], left[i-n], array, aux, t );                            \
                                                                                                \
                n = ccv_min( (int)(right0 - right1), (int)(right1 - right) );                   \
                for( i = 0; i < n; i++ )                                                        \
                    swap_func( left[i], right0[i-n+1], array, aux, t );                         \
                n = (int)(left - left1);                                                        \
                m = (int)(right1 - right);                                                      \
                if( n > 1 )                                                                     \
                {                                                                               \
                    if( m > 1 )                                                                 \
                    {                                                                           \
                        if( n > m )                                                             \
                        {                                                                       \
                            stack[++sp].lb = left0;                                             \
                            stack[sp].ub = left0 + n - 1;                                       \
                            left = right0 - m + 1, right = right0;                              \
                        }                                                                       \
                        else                                                                    \
                        {                                                                       \
                            stack[++sp].lb = right0 - m + 1;                                    \
                            stack[sp].ub = right0;                                              \
                            left = left0, right = left0 + n - 1;                                \
                        }                                                                       \
                    }                                                                           \
                    else                                                                        \
                        left = left0, right = left0 + n - 1;                                    \
                }                                                                               \
                else if( m > 1 )                                                                \
                    left = right0 - m + 1, right = right0;                                      \
                else                                                                            \
                    break;                                                                      \
            }                                                                                   \
        }                                                                                       \
    }                                                                                           \
}

#define _ccv_qsort_default_swap(a, b, array, aux, t) CCV_SWAP((a), (b), (t))

#define CCV_IMPLEMENT_QSORT(func_name, T, cmp)  \
    CCV_IMPLEMENT_QSORT_EX(func_name, T, cmp, _ccv_qsort_default_swap, int)

#endif

/* End of ccv/lib/ccv.h */
#line 1 "ccv/lib/3rdparty/sha1.h"
/*
 * SHA1 routine optimized to do word accesses rather than byte accesses,
 * and to avoid unnecessary copies into the context array.
 *
 * This was initially based on the Mozilla SHA1 implementation, although
 * none of the original Mozilla code remains.
 */

typedef struct {
	unsigned long long size;
	unsigned int H[5];
	unsigned int W[16];
} blk_SHA_CTX;

void blk_SHA1_Init(blk_SHA_CTX *ctx);
void blk_SHA1_Update(blk_SHA_CTX *ctx, const void *dataIn, unsigned long len);
void blk_SHA1_Final(unsigned char hashout[20], blk_SHA_CTX *ctx);

#define git_SHA_CTX	blk_SHA_CTX
#define git_SHA1_Init	blk_SHA1_Init
#define git_SHA1_Update	blk_SHA1_Update
#define git_SHA1_Final	blk_SHA1_Final

/* End of ccv/lib/3rdparty/sha1.h */
#line 1 "ccv/lib/3rdparty/sha1.c"
/*
 * SHA1 routine optimized to do word accesses rather than byte accesses,
 * and to avoid unnecessary copies into the context array.
 *
 * This was initially based on the Mozilla SHA1 implementation, although
 * none of the original Mozilla code remains.
 */

/* this is only to get definitions for memcpy(), ntohl() and htonl() */
#include <string.h>
#include <stdint.h>
#ifdef WIN32
#include <winsock.h>
#else
#include <arpa/inet.h>
#endif
/* sha1.h already included */


#if defined(__GNUC__) && (defined(__i386__) || defined(__x86_64__))

/*
 * Force usage of rol or ror by selecting the one with the smaller constant.
 * It _can_ generate slightly smaller code (a constant of 1 is special), but
 * perhaps more importantly it's possibly faster on any uarch that does a
 * rotate with a loop.
 */

#define SHA_ASM(op, x, n) ({ unsigned int __res; __asm__(op " %1,%0":"=r" (__res):"i" (n), "0" (x)); __res; })
#define SHA_ROL(x,n)	SHA_ASM("rol", x, n)
#define SHA_ROR(x,n)	SHA_ASM("ror", x, n)

#else

#define SHA_ROT(X,l,r)	(((X) << (l)) | ((X) >> (r)))
#define SHA_ROL(X,n)	SHA_ROT(X,n,32-(n))
#define SHA_ROR(X,n)	SHA_ROT(X,32-(n),n)

#endif

/*
 * If you have 32 registers or more, the compiler can (and should)
 * try to change the array[] accesses into registers. However, on
 * machines with less than ~25 registers, that won't really work,
 * and at least gcc will make an unholy mess of it.
 *
 * So to avoid that mess which just slows things down, we force
 * the stores to memory to actually happen (we might be better off
 * with a 'W(t)=(val);asm("":"+m" (W(t))' there instead, as
 * suggested by Artur Skawina - that will also make gcc unable to
 * try to do the silly "optimize away loads" part because it won't
 * see what the value will be).
 *
 * Ben Herrenschmidt reports that on PPC, the C version comes close
 * to the optimized asm with this (ie on PPC you don't want that
 * 'volatile', since there are lots of registers).
 *
 * On ARM we get the best code generation by forcing a full memory barrier
 * between each SHA_ROUND, otherwise gcc happily get wild with spilling and
 * the stack frame size simply explode and performance goes down the drain.
 */

#if defined(__i386__) || defined(__x86_64__)
  #define setW(x, val) (*(volatile unsigned int *)&W(x) = (val))
#elif defined(__GNUC__) && defined(__arm__)
  #define setW(x, val) do { W(x) = (val); __asm__("":::"memory"); } while (0)
#else
  #define setW(x, val) (W(x) = (val))
#endif

/*
 * Performance might be improved if the CPU architecture is OK with
 * unaligned 32-bit loads and a fast ntohl() is available.
 * Otherwise fall back to byte loads and shifts which is portable,
 * and is faster on architectures with memory alignment issues.
 */

#if defined(__i386__) || defined(__x86_64__) || \
    defined(_M_IX86) || defined(_M_X64) || \
    defined(__ppc__) || defined(__ppc64__) || \
    defined(__powerpc__) || defined(__powerpc64__) || \
    defined(__s390__) || defined(__s390x__)

#define get_be32(p)	ntohl(*(unsigned int *)(p))
#define put_be32(p, v)	do { *(unsigned int *)(p) = htonl(v); } while (0)

#else

#define get_be32(p)	( \
	(*((unsigned char *)(p) + 0) << 24) | \
	(*((unsigned char *)(p) + 1) << 16) | \
	(*((unsigned char *)(p) + 2) <<  8) | \
	(*((unsigned char *)(p) + 3) <<  0) )
#define put_be32(p, v)	do { \
	unsigned int __v = (v); \
	*((unsigned char *)(p) + 0) = __v >> 24; \
	*((unsigned char *)(p) + 1) = __v >> 16; \
	*((unsigned char *)(p) + 2) = __v >>  8; \
	*((unsigned char *)(p) + 3) = __v >>  0; } while (0)

#endif

/* This "rolls" over the 512-bit array */
#define W(x) (array[(x)&15])

/*
 * Where do we get the source from? The first 16 iterations get it from
 * the input data, the next mix it from the 512-bit array.
 */
#define SHA_SRC(t) get_be32(data + t)
#define SHA_MIX(t) SHA_ROL(W(t+13) ^ W(t+8) ^ W(t+2) ^ W(t), 1)

#define SHA_ROUND(t, input, fn, constant, A, B, C, D, E) do { \
	unsigned int TEMP = input(t); setW(t, TEMP); \
	E += TEMP + SHA_ROL(A,5) + (fn) + (constant); \
	B = SHA_ROR(B, 2); } while (0)

#define T_0_15(t, A, B, C, D, E)  SHA_ROUND(t, SHA_SRC, (((C^D)&B)^D) , 0x5a827999, A, B, C, D, E )
#define T_16_19(t, A, B, C, D, E) SHA_ROUND(t, SHA_MIX, (((C^D)&B)^D) , 0x5a827999, A, B, C, D, E )
#define T_20_39(t, A, B, C, D, E) SHA_ROUND(t, SHA_MIX, (B^C^D) , 0x6ed9eba1, A, B, C, D, E )
#define T_40_59(t, A, B, C, D, E) SHA_ROUND(t, SHA_MIX, ((B&C)+(D&(B^C))) , 0x8f1bbcdc, A, B, C, D, E )
#define T_60_79(t, A, B, C, D, E) SHA_ROUND(t, SHA_MIX, (B^C^D) ,  0xca62c1d6, A, B, C, D, E )

static void blk_SHA1_Block(blk_SHA_CTX *ctx, const unsigned int *data)
{
	unsigned int A,B,C,D,E;
	unsigned int array[16];

	A = ctx->H[0];
	B = ctx->H[1];
	C = ctx->H[2];
	D = ctx->H[3];
	E = ctx->H[4];

	/* Round 1 - iterations 0-16 take their input from 'data' */
	T_0_15( 0, A, B, C, D, E);
	T_0_15( 1, E, A, B, C, D);
	T_0_15( 2, D, E, A, B, C);
	T_0_15( 3, C, D, E, A, B);
	T_0_15( 4, B, C, D, E, A);
	T_0_15( 5, A, B, C, D, E);
	T_0_15( 6, E, A, B, C, D);
	T_0_15( 7, D, E, A, B, C);
	T_0_15( 8, C, D, E, A, B);
	T_0_15( 9, B, C, D, E, A);
	T_0_15(10, A, B, C, D, E);
	T_0_15(11, E, A, B, C, D);
	T_0_15(12, D, E, A, B, C);
	T_0_15(13, C, D, E, A, B);
	T_0_15(14, B, C, D, E, A);
	T_0_15(15, A, B, C, D, E);

	/* Round 1 - tail. Input from 512-bit mixing array */
	T_16_19(16, E, A, B, C, D);
	T_16_19(17, D, E, A, B, C);
	T_16_19(18, C, D, E, A, B);
	T_16_19(19, B, C, D, E, A);

	/* Round 2 */
	T_20_39(20, A, B, C, D, E);
	T_20_39(21, E, A, B, C, D);
	T_20_39(22, D, E, A, B, C);
	T_20_39(23, C, D, E, A, B);
	T_20_39(24, B, C, D, E, A);
	T_20_39(25, A, B, C, D, E);
	T_20_39(26, E, A, B, C, D);
	T_20_39(27, D, E, A, B, C);
	T_20_39(28, C, D, E, A, B);
	T_20_39(29, B, C, D, E, A);
	T_20_39(30, A, B, C, D, E);
	T_20_39(31, E, A, B, C, D);
	T_20_39(32, D, E, A, B, C);
	T_20_39(33, C, D, E, A, B);
	T_20_39(34, B, C, D, E, A);
	T_20_39(35, A, B, C, D, E);
	T_20_39(36, E, A, B, C, D);
	T_20_39(37, D, E, A, B, C);
	T_20_39(38, C, D, E, A, B);
	T_20_39(39, B, C, D, E, A);

	/* Round 3 */
	T_40_59(40, A, B, C, D, E);
	T_40_59(41, E, A, B, C, D);
	T_40_59(42, D, E, A, B, C);
	T_40_59(43, C, D, E, A, B);
	T_40_59(44, B, C, D, E, A);
	T_40_59(45, A, B, C, D, E);
	T_40_59(46, E, A, B, C, D);
	T_40_59(47, D, E, A, B, C);
	T_40_59(48, C, D, E, A, B);
	T_40_59(49, B, C, D, E, A);
	T_40_59(50, A, B, C, D, E);
	T_40_59(51, E, A, B, C, D);
	T_40_59(52, D, E, A, B, C);
	T_40_59(53, C, D, E, A, B);
	T_40_59(54, B, C, D, E, A);
	T_40_59(55, A, B, C, D, E);
	T_40_59(56, E, A, B, C, D);
	T_40_59(57, D, E, A, B, C);
	T_40_59(58, C, D, E, A, B);
	T_40_59(59, B, C, D, E, A);

	/* Round 4 */
	T_60_79(60, A, B, C, D, E);
	T_60_79(61, E, A, B, C, D);
	T_60_79(62, D, E, A, B, C);
	T_60_79(63, C, D, E, A, B);
	T_60_79(64, B, C, D, E, A);
	T_60_79(65, A, B, C, D, E);
	T_60_79(66, E, A, B, C, D);
	T_60_79(67, D, E, A, B, C);
	T_60_79(68, C, D, E, A, B);
	T_60_79(69, B, C, D, E, A);
	T_60_79(70, A, B, C, D, E);
	T_60_79(71, E, A, B, C, D);
	T_60_79(72, D, E, A, B, C);
	T_60_79(73, C, D, E, A, B);
	T_60_79(74, B, C, D, E, A);
	T_60_79(75, A, B, C, D, E);
	T_60_79(76, E, A, B, C, D);
	T_60_79(77, D, E, A, B, C);
	T_60_79(78, C, D, E, A, B);
	T_60_79(79, B, C, D, E, A);

	ctx->H[0] += A;
	ctx->H[1] += B;
	ctx->H[2] += C;
	ctx->H[3] += D;
	ctx->H[4] += E;
}

void blk_SHA1_Init(blk_SHA_CTX *ctx)
{
	ctx->size = 0;

	/* Initialize H with the magic constants (see FIPS180 for constants) */
	ctx->H[0] = 0x67452301;
	ctx->H[1] = 0xefcdab89;
	ctx->H[2] = 0x98badcfe;
	ctx->H[3] = 0x10325476;
	ctx->H[4] = 0xc3d2e1f0;
}

void blk_SHA1_Update(blk_SHA_CTX *ctx, const void *data, unsigned long len)
{
	unsigned int lenW = ctx->size & 63;

	ctx->size += len;

	/* Read the data into W and process blocks as they get full */
	if (lenW) {
		unsigned int left = 64 - lenW;
		if (len < left)
			left = len;
		memcpy(lenW + (char *)ctx->W, data, left);
		lenW = (lenW + left) & 63;
		len -= left;
		data = ((const char *)data + left);
		if (lenW)
			return;
		blk_SHA1_Block(ctx, ctx->W);
	}
	while (len >= 64) {
		blk_SHA1_Block(ctx, data);
		data = ((const char *)data + 64);
		len -= 64;
	}
	if (len)
		memcpy(ctx->W, data, len);
}

void blk_SHA1_Final(unsigned char hashout[20], blk_SHA_CTX *ctx)
{
	static const unsigned char pad[64] = { 0x80 };
	unsigned int padlen[2];
	int i;

	/* Pad with a binary 1 (ie 0x80), then zeroes, then length */
	padlen[0] = htonl((uint32_t)(ctx->size >> 29));
	padlen[1] = htonl((uint32_t)(ctx->size << 3));

	i = ctx->size & 63;
	blk_SHA1_Update(ctx, pad, 1+ (63 & (55 - i)));
	blk_SHA1_Update(ctx, padlen, 8);

	/* Output hash */
	for (i = 0; i < 5; i++)
		put_be32(hashout + i*4, ctx->H[i]);
}

/* End of ccv/lib/3rdparty/sha1.c */
#line 1 "ccv/lib/ccv_io.c"
/* ccv.h already included */

#ifdef HAVE_LIBJPEG
#include <jpeglib.h>
#endif
#ifdef HAVE_LIBPNG
#include <png.h>
#endif
#ifdef HAVE_LIBJPEG
#line 1 "io/_ccv_io_libjpeg.c"
typedef struct ccv_jpeg_error_mgr_t
{
	struct jpeg_error_mgr pub;
	jmp_buf setjmp_buffer;
} ccv_jpeg_error_mgr_t;

METHODDEF(void) error_exit(j_common_ptr cinfo)
{
	ccv_jpeg_error_mgr_t* err_mgr = (ccv_jpeg_error_mgr_t*)(cinfo->err);

	/* Return control to the setjmp point */
	longjmp(err_mgr->setjmp_buffer, 1);
}

/***************************************************************************
 * following code is for supporting MJPEG image files
 * based on a message of Laurent Pinchart on the video4linux mailing list
 ***************************************************************************/

/* JPEG DHT Segment for YCrCb omitted from MJPEG data */
static
unsigned char _ccv_jpeg_odml_dht[0x1a4] = {
	0xff, 0xc4, 0x01, 0xa2,

	0x00, 0x00, 0x01, 0x05, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00,
	0x00, 0x00, 0x00, 0x00, 0x00,
	0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b,

	0x01, 0x00, 0x03, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
	0x00, 0x00, 0x00, 0x00, 0x00,
	0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b,

	0x10, 0x00, 0x02, 0x01, 0x03, 0x03, 0x02, 0x04, 0x03, 0x05, 0x05, 0x04,
	0x04, 0x00, 0x00, 0x01, 0x7d,
	0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06,
	0x13, 0x51, 0x61, 0x07,
	0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xa1, 0x08, 0x23, 0x42, 0xb1, 0xc1,
	0x15, 0x52, 0xd1, 0xf0,
	0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0a, 0x16, 0x17, 0x18, 0x19, 0x1a,
	0x25, 0x26, 0x27, 0x28,
	0x29, 0x2a, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x43, 0x44, 0x45,
	0x46, 0x47, 0x48, 0x49,
	0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64, 0x65,
	0x66, 0x67, 0x68, 0x69,
	0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x83, 0x84, 0x85,
	0x86, 0x87, 0x88, 0x89,
	0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3,
	0xa4, 0xa5, 0xa6, 0xa7,
	0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba,
	0xc2, 0xc3, 0xc4, 0xc5,
	0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8,
	0xd9, 0xda, 0xe1, 0xe2,
	0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf1, 0xf2, 0xf3, 0xf4,
	0xf5, 0xf6, 0xf7, 0xf8,
	0xf9, 0xfa,

	0x11, 0x00, 0x02, 0x01, 0x02, 0x04, 0x04, 0x03, 0x04, 0x07, 0x05, 0x04,
	0x04, 0x00, 0x01, 0x02, 0x77,
	0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21, 0x31, 0x06, 0x12, 0x41,
	0x51, 0x07, 0x61, 0x71,
	0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91, 0xa1, 0xb1, 0xc1, 0x09,
	0x23, 0x33, 0x52, 0xf0,
	0x15, 0x62, 0x72, 0xd1, 0x0a, 0x16, 0x24, 0x34, 0xe1, 0x25, 0xf1, 0x17,
	0x18, 0x19, 0x1a, 0x26,
	0x27, 0x28, 0x29, 0x2a, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x43, 0x44,
	0x45, 0x46, 0x47, 0x48,
	0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64,
	0x65, 0x66, 0x67, 0x68,
	0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x82, 0x83,
	0x84, 0x85, 0x86, 0x87,
	0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a,
	0xa2, 0xa3, 0xa4, 0xa5,
	0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8,
	0xb9, 0xba, 0xc2, 0xc3,
	0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6,
	0xd7, 0xd8, 0xd9, 0xda,
	0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf2, 0xf3, 0xf4,
	0xf5, 0xf6, 0xf7, 0xf8,
	0xf9, 0xfa
};

/*
 * Parse the DHT table.
 * This code comes from jpeg6b (jdmarker.c).
 */
static int _ccv_jpeg_load_dht(struct jpeg_decompress_struct *info, unsigned char *dht, JHUFF_TBL *ac_tables[], JHUFF_TBL *dc_tables[])
{
	unsigned int length = (dht[2] << 8) + dht[3] - 2;
	unsigned int pos = 4;
	unsigned int count, i;
	int index;

	JHUFF_TBL **hufftbl;
	unsigned char bits[17];
	unsigned char huffval[256];

	while (length > 16)
	{
		bits[0] = 0;
		index = dht[pos++];
		count = 0;
		for (i = 1; i <= 16; ++i)
		{
			bits[i] = dht[pos++];
			count += bits[i];
		}
		length -= 17;

		if (count > 256 || count > length)
			return -1;

		for (i = 0; i < count; ++i)
			huffval[i] = dht[pos++];
		length -= count;

		if (index & 0x10)
		{
			index -= 0x10;
			hufftbl = &ac_tables[index];
		}
		else
			hufftbl = &dc_tables[index];

		if (index < 0 || index >= NUM_HUFF_TBLS)
			return -1;

		if (*hufftbl == 0)
			*hufftbl = jpeg_alloc_huff_table ((j_common_ptr)info);
		if (*hufftbl == 0)
			return -1;

		memcpy((*hufftbl)->bits, bits, sizeof (*hufftbl)->bits);
		memcpy((*hufftbl)->huffval, huffval, sizeof (*hufftbl)->huffval);
	}

	if (length != 0)
		return -1;

	return 0;
}

/***************************************************************************
 * end of code for supportting MJPEG image files
 * based on a message of Laurent Pinchart on the video4linux mailing list
 ***************************************************************************/

static void _ccv_unserialize_jpeg_fd(FILE* in, ccv_dense_matrix_t** x, int type)
{
	struct jpeg_decompress_struct cinfo;
	struct ccv_jpeg_error_mgr_t jerr;
	JSAMPARRAY buffer;
	int row_stride;
	cinfo.err = jpeg_std_error(&jerr.pub);
	jerr.pub.error_exit = error_exit;
	if (setjmp(jerr.setjmp_buffer))
	{
		jpeg_destroy_decompress(&cinfo);
		return;
	}
	jpeg_create_decompress(&cinfo);

	jpeg_stdio_src(&cinfo, in);

	jpeg_read_header(&cinfo, TRUE);
	
	ccv_dense_matrix_t* im = *x;
	if (im == 0)
		*x = im = ccv_dense_matrix_new(cinfo.image_height, cinfo.image_width, (type) ? type : CCV_8U | ((cinfo.num_components > 1) ? CCV_C3 : CCV_C1), 0, 0);

	/* yes, this is a mjpeg image format, so load the correct huffman table */
	if (cinfo.ac_huff_tbl_ptrs[0] == 0 && cinfo.ac_huff_tbl_ptrs[1] == 0 && cinfo.dc_huff_tbl_ptrs[0] == 0 && cinfo.dc_huff_tbl_ptrs[1] == 0)
		_ccv_jpeg_load_dht(&cinfo, _ccv_jpeg_odml_dht, cinfo.ac_huff_tbl_ptrs, cinfo.dc_huff_tbl_ptrs);

	if(cinfo.num_components != 4)
	{
		if (cinfo.num_components > 1)
		{
			cinfo.out_color_space = JCS_RGB;
			cinfo.out_color_components = 3;
		} else {
			cinfo.out_color_space = JCS_GRAYSCALE;
			cinfo.out_color_components = 1;
		}
	} else {
		cinfo.out_color_space = JCS_CMYK;
		cinfo.out_color_components = 4;
	}

	jpeg_start_decompress(&cinfo);
	row_stride = cinfo.output_width * 4;
	buffer = (*cinfo.mem->alloc_sarray)((j_common_ptr) &cinfo, JPOOL_IMAGE, row_stride, 1);

	unsigned char* ptr = im->data.ptr;
	if(cinfo.num_components != 4)
	{
		if ((cinfo.num_components > 1 && CCV_GET_CHANNEL(im->type) == CCV_C3) || (cinfo.num_components == 1 && CCV_GET_CHANNEL(im->type) == CCV_C1))
		{
			/* no format coversion, direct copy */
			while (cinfo.output_scanline < cinfo.output_height)
			{
				jpeg_read_scanlines(&cinfo, buffer, 1);
				memcpy(ptr, buffer[0], im->step);
				ptr += im->step;
			}
		} else {
			if (cinfo.num_components > 1 && CCV_GET_CHANNEL(im->type) == CCV_C1)
			{
				/* RGB to gray */
				while (cinfo.output_scanline < cinfo.output_height)
				{
					jpeg_read_scanlines(&cinfo, buffer, 1);
					int i;
					unsigned char* g = ptr;
					unsigned char* rgb = (unsigned char*)buffer[0];
					for(i = 0; i < im->cols; i++, rgb += 3, g++)
						*g = (unsigned char)((rgb[0] * 6969 + rgb[1] * 23434 + rgb[2] * 2365) >> 15);
					ptr += im->step;
				}
			} else if (cinfo.num_components == 1 && CCV_GET_CHANNEL(im->type) == CCV_C3) {
				/* gray to RGB */
				while (cinfo.output_scanline < cinfo.output_height)
				{
					jpeg_read_scanlines(&cinfo, buffer, 1);
					int i;
					unsigned char* g = (unsigned char*)buffer[0];
					unsigned char* rgb = ptr;
					for(i = 0; i < im->cols; i++, rgb += 3, g++)
						rgb[0] = rgb[1] = rgb[2] = *g;
					ptr += im->step;
				}
			}
		}
	} else {
		if (CCV_GET_CHANNEL(im->type) == CCV_C1)
		{
			/* CMYK to gray */
			while (cinfo.output_scanline < cinfo.output_height)
			{
				jpeg_read_scanlines(&cinfo, buffer, 1);
				int i;
				unsigned char* cmyk = (unsigned char*)buffer[0];
				unsigned char* g = ptr;
				for(i = 0; i < im->cols; i++, g++, cmyk += 4)
				{
					int c = cmyk[0], m = cmyk[1], y = cmyk[2], k = cmyk[3];
					c = k - ((255 - c) * k >> 8);
					m = k - ((255 - m) * k >> 8);
					y = k - ((255 - y) * k >> 8);
					*g = (unsigned char)((c * 6969 + m * 23434 + y * 2365) >> 15);
				}
				ptr += im->step;
			}
		} else if (CCV_GET_CHANNEL(im->type) == CCV_C3) {
			/* CMYK to RGB */
			while (cinfo.output_scanline < cinfo.output_height)
			{
				jpeg_read_scanlines(&cinfo, buffer, 1);
				int i;
				unsigned char* cmyk = (unsigned char*)buffer[0];
				unsigned char* rgb = ptr;
				for(i = 0; i < im->cols; i++, rgb += 3, cmyk += 4)
				{
					int c = cmyk[0], m = cmyk[1], y = cmyk[2], k = cmyk[3];
					c = k - ((255 - c) * k >> 8);
					m = k - ((255 - m) * k >> 8);
					y = k - ((255 - y) * k >> 8);
					rgb[0] = (unsigned char)c;
					rgb[1] = (unsigned char)m;
					rgb[2] = (unsigned char)y;
				}
				ptr += im->step;
			}
		}
	}

	jpeg_finish_decompress(&cinfo);
	jpeg_destroy_decompress(&cinfo);
}

static void _ccv_serialize_jpeg_fd(ccv_dense_matrix_t* mat, FILE* fd, void* conf)
{
	struct jpeg_compress_struct cinfo;
	struct ccv_jpeg_error_mgr_t jerr;
	jpeg_create_compress(&cinfo);
    cinfo.err = jpeg_std_error(&jerr.pub);
	jerr.pub.error_exit = error_exit;
	jpeg_stdio_dest(&cinfo, fd);
	if (setjmp(jerr.setjmp_buffer))
	{
		jpeg_destroy_compress(&cinfo);
		return;
	}
	cinfo.image_width = mat->cols;
	cinfo.image_height = mat->rows;
	cinfo.input_components = (CCV_GET_CHANNEL(mat->type) == CCV_C1) ? 1 : 3;
	cinfo.in_color_space = (CCV_GET_CHANNEL(mat->type) == CCV_C1) ? JCS_GRAYSCALE : JCS_RGB;
	jpeg_set_defaults(&cinfo);
	if (conf == 0)
		jpeg_set_quality(&cinfo, 95, TRUE);
	else
		jpeg_set_quality(&cinfo, *(int*)conf, TRUE);
	jpeg_start_compress(&cinfo, TRUE);
	int i;
	unsigned char* ptr = mat->data.ptr;
	for (i = 0; i < mat->rows; i++)
	{
		jpeg_write_scanlines(&cinfo, &ptr, 1);
		ptr += mat->step;
	}
	jpeg_finish_compress(&cinfo);
	jpeg_destroy_compress(&cinfo);
}

/* End of io/_ccv_io_libjpeg.c */

#endif
#ifdef HAVE_LIBPNG
#line 1 "io/_ccv_io_libpng.c"
static void _ccv_unserialize_png_fd(FILE* in, ccv_dense_matrix_t** x, int type)
{
	png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, 0, 0, 0);
	png_infop info_ptr = png_create_info_struct(png_ptr);
	png_infop end_info = png_create_info_struct(png_ptr);
	if (setjmp(png_jmpbuf(png_ptr)))
	{
		png_destroy_read_struct(&png_ptr, &info_ptr, &end_info);
		return;
	}
	png_init_io(png_ptr, in);
	png_read_info(png_ptr, info_ptr);
	png_uint_32 width, height;
	int bit_depth, color_type;
	png_get_IHDR(png_ptr, info_ptr, &width, &height, &bit_depth, &color_type, 0, 0, 0);
	
	ccv_dense_matrix_t* im = *x;
	if (im == 0)
		*x = im = ccv_dense_matrix_new((int) height, (int) width, (type) ? type : CCV_8U | ((color_type == PNG_COLOR_TYPE_GRAY) ? CCV_C1 : CCV_C3), 0, 0);

	png_set_strip_alpha(png_ptr);
	if (color_type == PNG_COLOR_TYPE_PALETTE)
		png_set_palette_to_rgb(png_ptr);
	if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8)
		png_set_expand_gray_1_2_4_to_8(png_ptr);
	if (CCV_GET_CHANNEL(im->type) == CCV_C3)
		png_set_gray_to_rgb(png_ptr);
	else if (CCV_GET_CHANNEL(im->type) == CCV_C1)
		png_set_rgb_to_gray(png_ptr, 1, -1, -1);

	unsigned char** row_vectors = (unsigned char**)alloca(im->rows * sizeof(unsigned char*));
	int i;
	for (i = 0; i < im->rows; i++)
		row_vectors[i] = im->data.ptr + i * im->step;
	png_read_image(png_ptr, row_vectors);
	png_read_end(png_ptr, end_info);

	png_destroy_read_struct(&png_ptr, &info_ptr, &end_info);
}

static void _ccv_serialize_png_fd(ccv_dense_matrix_t* mat, FILE* fd, void* conf)
{
	png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, 0, 0, 0);
	png_infop info_ptr = png_create_info_struct(png_ptr);
	if (setjmp(png_jmpbuf(png_ptr)))
	{
		png_destroy_write_struct(&png_ptr, &info_ptr);
		return;
	}
	png_init_io(png_ptr, fd);
	int compression_level = 0;
	if (conf != 0)
		compression_level = ccv_clamp(*(int*)conf, 0, MAX_MEM_LEVEL);
	if(compression_level > 0)
	{
		png_set_compression_mem_level(png_ptr, compression_level);
	} else {
		// tune parameters for speed
		// (see http://wiki.linuxquestions.org/wiki/Libpng)
		png_set_filter(png_ptr, PNG_FILTER_TYPE_BASE, PNG_FILTER_SUB);
		png_set_compression_level(png_ptr, Z_BEST_SPEED);
	}
	png_set_compression_strategy(png_ptr, Z_HUFFMAN_ONLY);
	png_set_IHDR(png_ptr, info_ptr, mat->cols, mat->rows, (mat->type & CCV_8U) ? 8 : 16, (mat->type & CCV_C1) ? PNG_COLOR_TYPE_GRAY : PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
	
	unsigned char** row_vectors = (unsigned char**)alloca(mat->rows * sizeof(unsigned char*));
	int i;
	for (i = 0; i < mat->rows; i++)
		row_vectors[i] = mat->data.ptr + i * mat->step;
	png_write_info(png_ptr, info_ptr);
	png_write_image(png_ptr, row_vectors);
	png_write_end(png_ptr, info_ptr);
	png_destroy_write_struct(&png_ptr, &info_ptr);
}

/* End of io/_ccv_io_libpng.c */

#endif
#line 1 "io/_ccv_io_bmp.c"
static void _ccv_unserialize_bmp_fd(FILE* in, ccv_dense_matrix_t** x, int type)
{
	fseek(in, 10, SEEK_SET);
	int offset;
	(void) fread(&offset, 4, 1, in);
	int size;
	(void) fread(&size, 4, 1, in);
	int width = 0, height = 0, bpp = 0, rle_code = 0, clrused = 0;
	if (size >= 36)
	{
		(void) fread(&width, 4, 1, in);
		(void) fread(&height, 4, 1, in);
		(void) fread(&bpp, 4, 1, in);
		bpp = bpp >> 16;
		(void) fread(&rle_code, 4, 1, in);
		fseek(in, 12, SEEK_CUR);
		(void) fread(&clrused, 4, 1, in);
		fseek(in, size - 36, SEEK_CUR);
		/* only support 24-bit bmp */
	} else if (size == 12) {
		(void) fread(&width, 4, 1, in);
		(void) fread(&height, 4, 1, in);
		(void) fread(&bpp, 4, 1, in);
		bpp = bpp >> 16;
		/* TODO: not finished */
	}
	if (width == 0 || height == 0 || bpp == 0)
		return;
	ccv_dense_matrix_t* im = *x;
	if (im == 0)
		*x = im = ccv_dense_matrix_new(height, width, (type) ? type : CCV_8U | ((bpp > 8) ? CCV_C3 : CCV_C1), 0, 0);
	fseek(in, offset, SEEK_SET);
	int i, j;
	unsigned char* ptr = im->data.ptr + (im->rows - 1) * im->step;
	if ((bpp == 8 && CCV_GET_CHANNEL(im->type) == CCV_C1) || (bpp == 24 && CCV_GET_CHANNEL(im->type) == CCV_C3))
	{
		if (CCV_GET_CHANNEL(im->type) == CCV_C1)
		{
			for (i = 0; i < im->rows; i++)
			{
				(void) fread(ptr, 1, im->step, in);
				ptr -= im->step;
			}
		} else {
			for (i = 0; i < im->rows; i++)
			{
				(void) fread(ptr, 1, im->step, in);
				for (j = 0; j < im->cols * 3; j += 3)
				{
					unsigned char t = ptr[j];
					ptr[j] = ptr[j + 2];
					ptr[j + 2] = t;
				}
				ptr -= im->step;
			}
		}
	} else {
		if (bpp == 24 && CCV_GET_CHANNEL(im->type) == CCV_C1)
		{
			int bufstep = (im->cols * 3 + 3) & -4;
			unsigned char* buffer = (unsigned char*)alloca(bufstep);
			for (i = 0; i < im->rows; i++)
			{
				(void) fread(buffer, 1, bufstep, in);
				unsigned char* rgb = buffer;
				unsigned char* g = ptr;
				for(j = 0; j < im->cols; j++, rgb += 3, g++)
					*g = (unsigned char)((rgb[2] * 6969 + rgb[1] * 23434 + rgb[0] * 2365) >> 15);
				ptr -= im->step;
			}
		} else if (bpp == 8 && CCV_GET_CHANNEL(im->type) == CCV_C3) {
			int bufstep = (im->cols + 3) & -4;
			unsigned char* buffer = (unsigned char*)alloca(bufstep);
			for (i = 0; i < im->rows; i++)
			{
				(void) fread(buffer, 1, bufstep, in);
				unsigned char* g = buffer;
				unsigned char* rgb = ptr;
				for(j = 0; j < im->cols; j++, rgb += 3, g++)
					rgb[2] = rgb[1] = rgb[0] = *g;
				ptr -= im->step;
			}
		}
	}
}

/* End of io/_ccv_io_bmp.c */

#line 1 "io/_ccv_io_binary.c"
static void _ccv_serialize_binary_fd(ccv_dense_matrix_t* mat, FILE* fd, void* conf)
{
	fwrite("CCVBINDM", 1, 8, fd);
	int ctype = mat->type & 0xFFFFF;
	fwrite(&ctype, 1, 4, fd);
	fwrite(&(mat->rows), 1, 4, fd);
	fwrite(&(mat->cols), 1, 4, fd);
	fwrite(mat->data.ptr, 1, mat->step * mat->rows, fd);
	fflush(fd);
}

static void _ccv_unserialize_binary_fd(FILE* in, ccv_dense_matrix_t** x, int type)
{
	fseek(in, 8, SEEK_SET);
	fread(&type, 1, 4, in);
	int rows, cols;
	fread(&rows, 1, 4, in);
	fread(&cols, 1, 4, in);
	*x = ccv_dense_matrix_new(rows, cols, type, 0, 0);
	fread((*x)->data.ptr, 1, (*x)->step * (*x)->rows, in);
}

/* End of io/_ccv_io_binary.c */


void ccv_unserialize(const char* in, ccv_dense_matrix_t** x, int type)
{
	FILE* fd = 0;
	int ctype = (type & 0xF00) ? CCV_8U | ((type & 0xF00) >> 8) : 0;
	if (type & CCV_SERIAL_ANY_FILE)
	{
		fd = fopen(in, "rb");
		assert(fd != 0);
	}
	if ((type & 0XFF) == CCV_SERIAL_ANY_FILE)
	{
		unsigned char sig[8];
		(void) fread(sig, 1, 8, fd);
		if (memcmp(sig, "\x89\x50\x4e\x47\xd\xa\x1a\xa", 8) == 0)
			type = CCV_SERIAL_PNG_FILE;
		else if (memcmp(sig, "\xff\xd8\xff", 3) == 0)
			type = CCV_SERIAL_JPEG_FILE;
		else if (memcmp(sig, "BM", 2) == 0)
			type = CCV_SERIAL_BMP_FILE;
		else if (memcmp(sig, "CCVBINDM", 8) == 0)
			type = CCV_SERIAL_BINARY_FILE;
		else {
			printf("Unknown file signature in '%s'\n", in);
			exit(1); // XXX make exit/return gracefull later
		};
		fseek(fd, 0, SEEK_SET);
	}
	switch (type & 0XFF)
	{
#ifdef HAVE_LIBJPEG
		case CCV_SERIAL_JPEG_FILE:
			_ccv_unserialize_jpeg_fd(fd, x, ctype);
			break;
#endif
#ifdef HAVE_LIBPNG
		case CCV_SERIAL_PNG_FILE:
			_ccv_unserialize_png_fd(fd, x, ctype);
			break;
#endif
		case CCV_SERIAL_BMP_FILE:
			_ccv_unserialize_bmp_fd(fd, x, ctype);
			break;
		case CCV_SERIAL_BINARY_FILE:
			_ccv_unserialize_binary_fd(fd, x, ctype);
	}
	if (*x != 0)
	{
		(*x)->sig = ccv_matrix_generate_signature((char*) (*x)->data.ptr, (*x)->rows * (*x)->step, 0);
		(*x)->type &= ~CCV_REUSABLE;
	}
	if (type & CCV_SERIAL_ANY_FILE)
		fclose(fd);
}

int ccv_serialize(ccv_dense_matrix_t* mat, char* out, int* len, int type, void* conf)
{
	FILE* fd = 0;
	if (type & CCV_SERIAL_ANY_FILE)
	{
		fd = fopen(out, "wb");
		assert(fd != 0);
	}
	switch (type)
	{
#ifdef HAVE_LIBJPEG
		case CCV_SERIAL_JPEG_FILE:
			_ccv_serialize_jpeg_fd(mat, fd, conf);
			if (len != 0)
				*len = 0;
			break;
#endif
#ifdef HAVE_LIBPNG
		case CCV_SERIAL_PNG_FILE:
			_ccv_serialize_png_fd(mat, fd, conf);
			if (len != 0)
				*len = 0;
			break;
#endif
		case CCV_SERIAL_BINARY_FILE:
			_ccv_serialize_binary_fd(mat, fd, conf);
			if (len != 0)
				*len = 0;
			break;
	}
	if (type & CCV_SERIAL_ANY_FILE)
		fclose(fd);
	return CCV_SERIAL_FINAL;
}

/* End of ccv/lib/ccv_io.c */
#line 1 "ccv/lib/ccv_algebra.c"
/* ccv.h already included */

#ifdef HAVE_CBLAS
#include <cblas.h>
#endif

double ccv_trace(ccv_matrix_t* mat)
{
	return 0;
}

double ccv_norm(ccv_matrix_t* mat, int type)
{
	return 0;
}

double ccv_normalize(ccv_matrix_t* a, ccv_matrix_t** b, int btype, int l_type)
{
	ccv_dense_matrix_t* da = ccv_get_dense_matrix(a);
	assert(CCV_GET_CHANNEL(da->type) == CCV_C1);
	char identifier[20];
	memset(identifier, 0, 20);
	snprintf(identifier, 20, "ccv_normalize(%d)", l_type);
	uint64_t sig = (da->sig == 0) ? 0 : ccv_matrix_generate_signature(identifier, 20, da->sig, 0);
	btype = (btype == 0) ? CCV_GET_DATA_TYPE(da->type) | CCV_C1 : CCV_GET_DATA_TYPE(btype) | CCV_C1;
	ccv_dense_matrix_t* db = *b = ccv_dense_matrix_renew(*b, da->rows, da->cols, CCV_ALL_DATA_TYPE | CCV_C1, btype, sig);
	ccv_cache_return(db, 0);
	double sum = 0, inv;
	int i, j;
	unsigned char* a_ptr = da->data.ptr;
	unsigned char* b_ptr = db->data.ptr;
	switch (l_type)
	{
		case CCV_L1_NORM:
#define for_block(_for_set, _for_get) \
			for (i = 0; i < da->rows; i++) \
			{ \
				for (j = 0; j < da->cols; j++) \
					sum += _for_get(a_ptr, j, 0); \
				a_ptr += da->step; \
			} \
			inv = 1.0 / sum; \
			a_ptr = da->data.ptr; \
			for (i = 0; i < da->rows; i++) \
			{ \
				for (j = 0; j < da->cols; j++) \
					_for_set(b_ptr, j, _for_get(a_ptr, j, 0) * inv, 0); \
				a_ptr += da->step; \
				b_ptr += db->step; \
			}
			ccv_matrix_setter(db->type, ccv_matrix_getter, da->type, for_block);
#undef for_block
			break;
		case CCV_L2_NORM:
#define for_block(_for_set, _for_get) \
			for (i = 0; i < da->rows; i++) \
			{ \
				for (j = 0; j < da->cols; j++) \
					sum += _for_get(a_ptr, j, 0) * _for_get(a_ptr, j, 0); \
				a_ptr += da->step; \
			} \
			sum = sqrt(sum); \
			inv = 1.0 / sum; \
			a_ptr = da->data.ptr; \
			for (i = 0; i < da->rows; i++) \
			{ \
				for (j = 0; j < da->cols; j++) \
					_for_set(b_ptr, j, _for_get(a_ptr, j, 0) * inv, 0); \
				a_ptr += da->step; \
				b_ptr += db->step; \
			}
			ccv_matrix_setter(db->type, ccv_matrix_getter, da->type, for_block);
#undef for_block
			break;
	}
	return sum;
}

double ccv_sum(ccv_matrix_t* mat)
{
	ccv_dense_matrix_t* dmt = ccv_get_dense_matrix(mat);
	double sum = 0;
	unsigned char* m_ptr = dmt->data.ptr;
	int i, j;
#define for_block(_, _for_get) \
	for (i = 0; i < dmt->rows; i++) \
	{ \
		for (j = 0; j < dmt->cols; j++) \
			sum += _for_get(m_ptr, j, 0); \
		m_ptr += dmt->step; \
	}
	ccv_matrix_getter(dmt->type, for_block);
#undef for_block
	return sum;
}

void ccv_zero(ccv_matrix_t* mat)
{
	ccv_dense_matrix_t* dmt = ccv_get_dense_matrix(mat);
	memset(dmt->data.ptr, 0, dmt->step * dmt->rows);
}

void ccv_substract(ccv_matrix_t* a, ccv_matrix_t* b, ccv_matrix_t** c, int type)
{
	ccv_dense_matrix_t* da = ccv_get_dense_matrix(a);
	ccv_dense_matrix_t* db = ccv_get_dense_matrix(b);
	assert(da->rows == db->rows && da->cols == db->cols && CCV_GET_DATA_TYPE(da->type) == CCV_GET_DATA_TYPE(db->type) && CCV_GET_CHANNEL(da->type) == CCV_GET_CHANNEL(db->type));
	uint64_t sig = ccv_matrix_generate_signature("ccv_substract", 13, da->sig, db->sig, 0);
	int no_8u_type = (da->type & CCV_8U) ? CCV_32S : da->type;
	type = (type == 0) ? CCV_GET_DATA_TYPE(no_8u_type) | CCV_GET_CHANNEL(da->type) : CCV_GET_DATA_TYPE(type) | CCV_GET_CHANNEL(da->type);
	ccv_dense_matrix_t* dc = *c = ccv_dense_matrix_renew(*c, da->rows, da->cols, CCV_ALL_DATA_TYPE | CCV_GET_CHANNEL(da->type), type, sig);
	ccv_cache_return(dc, );
	int i, j;
	unsigned char* aptr = da->data.ptr;
	unsigned char* bptr = db->data.ptr;
	unsigned char* cptr = dc->data.ptr;
#define for_block(_for_get, _for_set) \
	for (i = 0; i < da->rows; i++) \
	{ \
		for (j = 0; j < da->cols; j++) \
		{ \
			_for_set(cptr, j, _for_get(aptr, j, 0) - _for_get(bptr, j, 0), 0); \
		} \
		aptr += da->step; \
		bptr += db->step; \
		cptr += dc->step; \
	}
	ccv_matrix_getter(da->type, ccv_matrix_setter, dc->type, for_block);
#undef for_block
}

void ccv_gemm(ccv_matrix_t* a, ccv_matrix_t* b, double alpha, ccv_matrix_t* c, double beta, int transpose, ccv_matrix_t** d, int type)
{
	ccv_dense_matrix_t* da = ccv_get_dense_matrix(a);
	ccv_dense_matrix_t* db = ccv_get_dense_matrix(b);
	ccv_dense_matrix_t* dc = (c == 0) ? 0 : ccv_get_dense_matrix(c);

	assert(CCV_GET_DATA_TYPE(da->type) == CCV_GET_DATA_TYPE(db->type) && CCV_GET_CHANNEL(da->type) == 1 && CCV_GET_CHANNEL(db->type) == 1 && ((transpose & CCV_A_TRANSPOSE) ? da->rows : da->cols) == ((transpose & CCV_B_TRANSPOSE) ? db->cols : db->rows));

	if (dc != 0)
		assert(CCV_GET_DATA_TYPE(dc->type) == CCV_GET_DATA_TYPE(da->type) && CCV_GET_CHANNEL(dc->type) == 1 && ((transpose & CCV_A_TRANSPOSE) ? da->cols : da->rows) == dc->rows && ((transpose & CCV_B_TRANSPOSE) ? db->rows : db->cols) == dc->cols);

	char identifier[20];
	memset(identifier, 0, 20);
	snprintf(identifier, 20, "ccv_gemm(%d)", transpose);
	uint64_t sig = (dc == 0) ? ((da->sig == 0 || db->sig == 0) ? 0 : ccv_matrix_generate_signature(identifier, 20, da->sig, db->sig, 0)) : ((da->sig == 0 || db->sig == 0 || dc->sig == 0) ? 0 : ccv_matrix_generate_signature(identifier, 20, da->sig, db->sig, dc->sig, 0));
	type = CCV_GET_DATA_TYPE(da->type) | CCV_GET_CHANNEL(da->type);
	ccv_dense_matrix_t* dd = *d = ccv_dense_matrix_renew(*d, (transpose & CCV_A_TRANSPOSE) ? da->cols : da->rows, (transpose & CCV_B_TRANSPOSE) ? db->rows : db->cols, type, type, sig);
	ccv_cache_return(dd, );

	if (dd != dc && dc != 0)
		memcpy(dd->data.ptr, dc->data.ptr, dc->step * dc->rows);

#ifdef HAVE_CBLAS
	switch (CCV_GET_DATA_TYPE(dd->type))
	{
		case CCV_32F:
			cblas_sgemm(CblasRowMajor, (transpose & CCV_A_TRANSPOSE) ? CblasTrans : CblasNoTrans, (transpose & CCV_B_TRANSPOSE) ? CblasTrans : CblasNoTrans, dd->rows, dd->cols, da->cols, alpha, da->data.fl, da->cols, db->data.fl, db->cols, beta, dd->data.fl, dd->cols);
			break;
		case CCV_64F:
			cblas_dgemm(CblasRowMajor, (transpose & CCV_A_TRANSPOSE) ? CblasTrans : CblasNoTrans, (transpose & CCV_B_TRANSPOSE) ? CblasTrans : CblasNoTrans, dd->rows, dd->cols, (transpose & CCV_A_TRANSPOSE) ? da->rows : da->cols, alpha, da->data.db, da->cols, db->data.db, db->cols, beta, dd->data.db, dd->cols);
			break;
	}
#endif
}

/* End of ccv/lib/ccv_algebra.c */
#line 1 "ccv/lib/ccv_basic.c"
/* ccv.h already included */


/* sobel filter is fundamental to many other high-level algorithms,
 * here includes 2 special case impl (for 1x3/3x1, 3x3) and one general impl */
void ccv_sobel(ccv_dense_matrix_t* a, ccv_dense_matrix_t** b, int type, int dx, int dy)
{
	assert(a->type & CCV_C1);
	char identifier[64];
	memset(identifier, 0, 64);
	snprintf(identifier, 64, "ccv_sobel(%d,%d)", dx, dy);
	uint64_t sig = (a->sig == 0) ? 0 : ccv_matrix_generate_signature(identifier, 64, a->sig, 0);
	type = (type == 0) ? CCV_32S | CCV_C1 : CCV_GET_DATA_TYPE(type) | CCV_C1;
	ccv_dense_matrix_t* db = *b = ccv_dense_matrix_renew(*b, a->rows, a->cols, CCV_C1 | CCV_ALL_DATA_TYPE, type, sig);
	ccv_cache_return(db, );
	int i, j, k;
	unsigned char* a_ptr = a->data.ptr;
	unsigned char* b_ptr = db->data.ptr;
	if (dx == 1 || dy == 1)
	{
		/* special case 1: 1x3 or 3x1 window */
		if (dx > dy)
		{
#define for_block(_for_get, _for_set) \
			for (i = 0; i < a->rows; i++) \
			{ \
				_for_set(b_ptr, 0, _for_get(a_ptr, 1, 0) - _for_get(a_ptr, 0, 0), 0); \
				for (j = 1; j < a->cols - 1; j++) \
					_for_set(b_ptr, j, 2 * (_for_get(a_ptr, j + 1, 0) - _for_get(a_ptr, j - 1, 0)), 0); \
				_for_set(b_ptr, a->cols - 1, _for_get(a_ptr, a->cols - 1, 0) - _for_get(a_ptr, a->cols - 2, 0), 0); \
				b_ptr += db->step; \
				a_ptr += a->step; \
			}
			ccv_matrix_getter(a->type, ccv_matrix_setter, db->type, for_block);
#undef for_block
		} else {
#define for_block(_for_get, _for_set) \
			for (j = 0; j < a->cols; j++) \
				_for_set(b_ptr, j, _for_get(a_ptr + a->step, j, 0) - _for_get(a_ptr, j, 0), 0); \
			a_ptr += a->step; \
			b_ptr += db->step; \
			for (i = 1; i < a->rows - 1; i++) \
			{ \
				for (j = 0; j < a->cols; j++) \
					_for_set(b_ptr, j, 2 * (_for_get(a_ptr + a->step, j, 0) - _for_get(a_ptr - a->step, j, 0)), 0); \
				a_ptr += a->step; \
				b_ptr += db->step; \
			} \
			for (j = 0; j < a->cols; j++) \
				_for_set(b_ptr, j, _for_get(a_ptr, j, 0) - _for_get(a_ptr - a->step, j, 0), 0);
			ccv_matrix_getter(a->type, ccv_matrix_setter, db->type, for_block);
#undef for_block
		}
	} else if (dx > 3 || dy > 3) {
		/* general case: in this case, I will generate a separable filter, and do the convolution */
		int fsz = ccv_max(dx, dy);
		assert(fsz % 2 == 1);
		int hfz = fsz / 2;
		unsigned char* df = (unsigned char*)alloca(sizeof(double) * fsz);
		unsigned char* gf = (unsigned char*)alloca(sizeof(double) * fsz);
		/* the sigma calculation is linear derviation of 3x3 - 0.85, 5x5 - 1.32 */
		double sigma = ((fsz - 1) / 2) * 0.47 + 0.38;
		double sigma2 = (2.0 * sigma * sigma);
		/* 2.5 is the factor to make the kernel "visible" in integer setting */
		double psigma3 = 2.5 / sqrt(sqrt(2 * CCV_PI) * sigma * sigma * sigma);
		for (i = 0; i < fsz; i++)
		{
			((double*)df)[i] = (i - hfz) * exp(-((i - hfz) * (i - hfz)) / sigma2) * psigma3;
			((double*)gf)[i] = exp(-((i - hfz) * (i - hfz)) / sigma2) * psigma3;
		}
		if (db->type & CCV_32S)
		{
			for (i = 0; i < fsz; i++)
			{
				// df could be negative, thus, (int)(x + 0.5) shortcut will not work
				((int*)df)[i] = (int)round(((double*)df)[i] * 256.0);
				((int*)gf)[i] = (int)(((double*)gf)[i] * 256.0 + 0.5);
			}
		} else {
			for (i = 0; i < fsz; i++)
			{
				ccv_set_value(db->type, df, i, ((double*)df)[i], 0);
				ccv_set_value(db->type, gf, i, ((double*)gf)[i], 0);
			}
		}
		if (dx < dy)
		{
			unsigned char* tf = df;
			df = gf;
			gf = tf;
		}
		unsigned char* buf = (unsigned char*)alloca(sizeof(double) * (fsz + ccv_max(a->rows, a->cols)));
#define for_block(_for_get, _for_type_b, _for_set_b, _for_get_b) \
		for (i = 0; i < a->rows; i++) \
		{ \
			for (j = 0; j < hfz; j++) \
				_for_set_b(buf, j, _for_get(a_ptr, 0, 0), 0); \
			for (j = 0; j < a->cols; j++) \
				_for_set_b(buf, j + hfz, _for_get(a_ptr, j, 0), 0); \
			for (j = a->cols; j < a->cols + hfz; j++) \
				_for_set_b(buf, j + hfz, _for_get(a_ptr, a->cols - 1, 0), 0); \
			for (j = 0; j < a->cols; j++) \
			{ \
				_for_type_b sum = 0; \
				for (k = 0; k < fsz; k++) \
					sum += _for_get_b(buf, j + k, 0) * _for_get_b(df, k, 0); \
				_for_set_b(b_ptr, j, sum, 8); \
			} \
			a_ptr += a->step; \
			b_ptr += db->step; \
		}
		ccv_matrix_getter(a->type, ccv_matrix_typeof_setter_getter, db->type, for_block);
#undef for_block
		b_ptr = db->data.ptr;
#define for_block(_, _for_type, _for_set, _for_get) \
		for (i = 0; i < a->cols; i++) \
		{ \
			for (j = 0; j < hfz; j++) \
				_for_set(buf, j, _for_get(b_ptr, i, 0), 0); \
			for (j = 0; j < a->rows; j++) \
				_for_set(buf, j + hfz, _for_get(b_ptr + j * db->step, i, 0), 0); \
			for (j = a->rows; j < a->rows + hfz; j++) \
				_for_set(buf, j + hfz, _for_get(b_ptr + (a->rows - 1) * db->step, i, 0), 0); \
			for (j = 0; j < a->rows; j++) \
			{ \
				_for_type sum = 0; \
				for (k = 0; k < fsz; k++) \
					sum += _for_get(buf, j + k, 0) * _for_get(gf, k, 0); \
				_for_set(b_ptr + j * db->step, i, sum, 8); \
			} \
		}
		ccv_matrix_typeof_setter_getter(db->type, for_block);
#undef for_block
	} else {
		/* special case 2: 3x3 window, corresponding sigma = 0.85 */
		unsigned char* buf = (unsigned char*)alloca(db->step);
		if (dx > dy)
		{
#define for_block(_for_get, _for_set) \
			for (j = 0; j < a->cols; j++) \
				_for_set(b_ptr, j, _for_get(a_ptr + a->step, j, 0) + 3 * _for_get(a_ptr, j, 0), 0); \
			a_ptr += a->step; \
			b_ptr += db->step; \
			for (i = 1; i < a->rows - 1; i++) \
			{ \
				for (j = 0; j < a->cols; j++) \
					_for_set(b_ptr, j, _for_get(a_ptr + a->step, j, 0) + 2 * _for_get(a_ptr, j, 0) + _for_get(a_ptr - a->step, j, 0), 0); \
				a_ptr += a->step; \
				b_ptr += db->step; \
			} \
			for (j = 0; j < a->cols; j++) \
				_for_set(b_ptr, j, 3 * _for_get(a_ptr, j, 0) + _for_get(a_ptr - a->step, j, 0), 0);
			ccv_matrix_getter(a->type, ccv_matrix_setter, db->type, for_block);
#undef for_block
			b_ptr = db->data.ptr;
#define for_block(_, _for_set, _for_get) \
			for (i = 0; i < a->rows; i++) \
			{ \
				_for_set(buf, 0, _for_get(b_ptr, 1, 0) - _for_get(b_ptr, 0, 0), 0); \
				for (j = 1; j < a->cols - 1; j++) \
					_for_set(buf, j, _for_get(b_ptr, j + 1, 0) - _for_get(b_ptr, j - 1, 0), 0); \
				_for_set(buf, a->cols - 1, _for_get(b_ptr, a->cols - 1, 0) - _for_get(b_ptr, a->cols - 2, 0), 0); \
				memcpy(b_ptr, buf, db->step); \
				b_ptr += db->step; \
			}
			ccv_matrix_setter_getter(db->type, for_block);
#undef for_block
		} else {
#define for_block(_for_get, _for_set) \
			for (j = 0; j < a->cols; j++) \
				_for_set(b_ptr, j, _for_get(a_ptr + a->step, j, 0) - _for_get(a_ptr, j, 0), 0); \
			a_ptr += a->step; \
			b_ptr += db->step; \
			for (i = 1; i < a->rows - 1; i++) \
			{ \
				for (j = 0; j < a->cols; j++) \
					_for_set(b_ptr, j, _for_get(a_ptr + a->step, j, 0) - _for_get(a_ptr - a->step, j, 0), 0); \
				a_ptr += a->step; \
				b_ptr += db->step; \
			} \
			for (j = 0; j < a->cols; j++) \
				_for_set(b_ptr, j, _for_get(a_ptr, j, 0) - _for_get(a_ptr - a->step, j, 0), 0);
			ccv_matrix_getter(a->type, ccv_matrix_setter, db->type, for_block);
#undef for_block
			b_ptr = db->data.ptr;
#define for_block(_, _for_set, _for_get) \
			for (i = 0; i < a->rows; i++) \
			{ \
				_for_set(buf, 0, _for_get(b_ptr, 1, 0) + 3 * _for_get(b_ptr, 0, 0), 0); \
				for (j = 1; j < a->cols - 1; j++) \
					_for_set(buf, j, _for_get(b_ptr, j + 1, 0) + 2 * _for_get(b_ptr, j, 0) + _for_get(b_ptr, j - 1, 0), 0); \
				_for_set(buf, a->cols - 1, _for_get(b_ptr, a->cols - 2, 0) + 3 * _for_get(b_ptr, a->cols - 1, 0), 0); \
				memcpy(b_ptr, buf, db->step); \
				b_ptr += db->step; \
			}
			ccv_matrix_setter_getter(db->type, for_block);
#undef for_block
		}
	}
}

/* the fast arctan function adopted from OpenCV */
static void _ccv_atan2(float* x, float* y, float* angle, float* mag, int len)
{
	int i = 0;
	float scale = (float)(180.0 / CCV_PI);
#ifndef _WIN32
	union { int i; float fl; } iabsmask; iabsmask.i = 0x7fffffff;
	__m128 eps = _mm_set1_ps((float)1e-6), absmask = _mm_set1_ps(iabsmask.fl);
	__m128 _90 = _mm_set1_ps((float)(3.141592654 * 0.5)), _180 = _mm_set1_ps((float)3.141592654), _360 = _mm_set1_ps((float)(3.141592654 * 2));
	__m128 zero = _mm_setzero_ps(), _0_28 = _mm_set1_ps(0.28f), scale4 = _mm_set1_ps(scale);
	
	for(; i <= len - 4; i += 4)
	{
		__m128 x4 = _mm_loadu_ps(x + i), y4 = _mm_loadu_ps(y + i);
		__m128 xq4 = _mm_mul_ps(x4, x4), yq4 = _mm_mul_ps(y4, y4);
		__m128 xly = _mm_cmplt_ps(xq4, yq4);
		__m128 z4 = _mm_div_ps(_mm_mul_ps(x4, y4), _mm_add_ps(_mm_add_ps(_mm_max_ps(xq4, yq4), _mm_mul_ps(_mm_min_ps(xq4, yq4), _0_28)), eps));

		// a4 <- x < y ? 90 : 0;
		__m128 a4 = _mm_and_ps(xly, _90);
		// a4 <- (y < 0 ? 360 - a4 : a4) == ((x < y ? y < 0 ? 270 : 90) : (y < 0 ? 360 : 0))
		__m128 mask = _mm_cmplt_ps(y4, zero);
		a4 = _mm_or_ps(_mm_and_ps(_mm_sub_ps(_360, a4), mask), _mm_andnot_ps(mask, a4));
		// a4 <- (x < 0 && !(x < y) ? 180 : a4)
		mask = _mm_andnot_ps(xly, _mm_cmplt_ps(x4, zero));
		a4 = _mm_or_ps(_mm_and_ps(_180, mask), _mm_andnot_ps(mask, a4));
		
		// a4 <- (x < y ? a4 - z4 : a4 + z4)
		a4 = _mm_mul_ps(_mm_add_ps(_mm_xor_ps(z4, _mm_andnot_ps(absmask, xly)), a4), scale4);
		__m128 m4 = _mm_sqrt_ps(_mm_add_ps(xq4, yq4));
		_mm_storeu_ps(angle + i, a4);
		_mm_storeu_ps(mag + i, m4);
	}
#endif
	for(; i < len; i++)
	{
		float xf = x[i], yf = y[i];
		float a, x2 = xf * xf, y2 = yf * yf;
		if(y2 <= x2)
			a = xf * yf / (x2 + 0.28f * y2 + (float)1e-6) + (float)(xf < 0 ? CCV_PI : yf >= 0 ? 0 : CCV_PI * 2);
		else
			a = (float)(yf >= 0 ? CCV_PI * 0.5 : CCV_PI * 1.5) - xf * yf / (y2 + 0.28f * x2 + (float)1e-6);
		angle[i] = a * scale;
		mag[i] = sqrt(x2 + y2);
	}
}

void ccv_gradient(ccv_dense_matrix_t* a, ccv_dense_matrix_t** theta, int ttype, ccv_dense_matrix_t** m, int mtype, int dx, int dy)
{
	assert(a->type & CCV_C1);
	uint64_t tsig = (a->sig == 0) ? 0 : ccv_matrix_generate_signature("ccv_gradient_theta", 18, a->sig, 0);
	uint64_t msig = (a->sig == 0) ? 0 : ccv_matrix_generate_signature("ccv_gradient_m", 14, a->sig, 0);
	ccv_dense_matrix_t* dtheta = *theta = ccv_dense_matrix_renew(*theta, a->rows, a->cols, CCV_32F | CCV_C1, CCV_32F | CCV_C1, tsig);
	ccv_dense_matrix_t* dm = *m = ccv_dense_matrix_renew(*m, a->rows, a->cols, CCV_32F | CCV_C1, CCV_32F | CCV_C1, msig);
	if ((dtheta->type & CCV_GARBAGE) && (dm->type & CCV_GARBAGE))
	{
		dtheta->type &= ~CCV_GARBAGE;
		dm->type &= ~CCV_GARBAGE;
		return;
	}
	dtheta->type &= ~CCV_GARBAGE;
	dm->type &= ~CCV_GARBAGE;
	ccv_dense_matrix_t* tx = 0;
	ccv_dense_matrix_t* ty = 0;
	ccv_sobel(a, &tx, CCV_32F | CCV_C1, dx, 0);
	ccv_sobel(a, &ty, CCV_32F | CCV_C1, 0, dy);
	_ccv_atan2(tx->data.fl, ty->data.fl, dtheta->data.fl, dm->data.fl, a->rows * a->cols);
	ccv_matrix_free(tx);
	ccv_matrix_free(ty);
}

void ccv_hog(ccv_dense_matrix_t* a, ccv_dense_matrix_t** b, int type, int size)
{
	assert(a->type & CCV_C1);
	int border_size = size / 2;
	uint64_t sig = (a->sig == 0) ? 0 : ccv_matrix_generate_signature("ccv_hog", 7, a->sig, 0);
	type = (type == 0) ? CCV_32S | CCV_C1 : CCV_GET_DATA_TYPE(type) | CCV_C1;
	ccv_dense_matrix_t* db = *b = ccv_dense_matrix_renew(*b, a->rows - border_size * 2, (a->cols - border_size * 2) * 8, CCV_C1 | CCV_ALL_DATA_TYPE, type, sig);
	ccv_cache_return(db, );
	ccv_dense_matrix_t* ag = 0;
	ccv_dense_matrix_t* mg = 0;
	ccv_gradient(a, &ag, 0, &mg, 0, 3, 3);
	int i, j, x, y;
	ag->type = (ag->type & ~CCV_32F) | CCV_32S;
	for (i = 0; i < a->rows * a->cols; i++)
		ag->data.i[i] = ((int)(ag->data.fl[i] / 45 + 0.5)) & 0x7;
	int* agi = ag->data.i;
	float* mgfl = mg->data.fl;
	int* bi = db->data.i;
	for (y = 0; y <= a->rows - size; y++)
	{
		float hog[] = { 0, 0, 0, 0, 0, 0, 0, 0 };
		for (i = 0; i < size; i++)
			for (j = 0; j < size; j++)
				hog[agi[i * a->cols + j]] += mgfl[i * a->cols + j];
		for (i = 0; i < 8; i++)
			bi[i] = (int)hog[i];
		bi += 8;
		mgfl++;
		agi++;
		for (x = 1; x <= a->cols - size; x++)
		{
			for (i = 0; i < size; i++)
			{
				hog[agi[i * a->cols - 1]] -= mgfl[i * a->cols - 1];
				hog[agi[i * a->cols - 1 + size]] += mgfl[i * a->cols - 1 + size];
			}
			for (i = 0; i < 8; i++)
				bi[i] = (int)hog[i];
			bi += 8;
			mgfl++;
			agi++;
		}
		agi += border_size * 2;
		mgfl += border_size * 2;
	}
	ccv_matrix_free(ag);
	ccv_matrix_free(mg);
}

/* it is a supposely cleaner and faster implementation than original OpenCV (ccv_canny_deprecated,
 * removed, since the newer implementation achieve bit accuracy with OpenCV's), after a lot
 * profiling, the current implementation still uses integer to speed up */
void ccv_canny(ccv_dense_matrix_t* a, ccv_dense_matrix_t** b, int type, int size, double low_thresh, double high_thresh)
{
	assert(a->type & CCV_C1);
	char identifier[64];
	memset(identifier, 0, 64);
	snprintf(identifier, 64, "ccv_canny(%d,%lf,%lf)", size, low_thresh, high_thresh);
	uint64_t sig = (a->sig == 0) ? 0 : ccv_matrix_generate_signature(identifier, 64, a->sig, 0);
	type = (type == 0) ? CCV_8U | CCV_C1 : CCV_GET_DATA_TYPE(type) | CCV_C1;
	ccv_dense_matrix_t* db = *b = ccv_dense_matrix_renew(*b, a->rows, a->cols, CCV_C1 | CCV_ALL_DATA_TYPE, type, sig);
	ccv_cache_return(db, );
	if ((a->type & CCV_8U) || (a->type & CCV_32S))
	{
		ccv_dense_matrix_t* dx = 0;
		ccv_dense_matrix_t* dy = 0;
		ccv_sobel(a, &dx, 0, size, 0);
		ccv_sobel(a, &dy, 0, 0, size);
		/* special case, all integer */
		int low = (int)(low_thresh + 0.5);
		int high = (int)(high_thresh + 0.5);
		int* dxi = dx->data.i;
		int* dyi = dy->data.i;
		int i, j;
		int* mbuf = (int*)alloca(3 * (a->cols + 2) * sizeof(int));
		memset(mbuf, 0, 3 * (a->cols + 2) * sizeof(int));
		int* rows[3];
		rows[0] = mbuf + 1;
		rows[1] = mbuf + (a->cols + 2) + 1;
		rows[2] = mbuf + 2 * (a->cols + 2) + 1;
		for (j = 0; j < a->cols; j++)
			rows[1][j] = abs(dxi[j]) + abs(dyi[j]);
		dxi += a->cols;
		dyi += a->cols;
		int* map = (int*)ccmalloc(sizeof(int) * (a->rows + 2) * (a->cols + 2));
		memset(map, 0, sizeof(int) * (a->cols + 2));
		int* map_ptr = map + a->cols + 2 + 1;
		int map_cols = a->cols + 2;
		int** stack = (int**)ccmalloc(sizeof(int*) * a->rows * a->cols);
		int** stack_top = stack;
		int** stack_bottom = stack;
		for (i = 1; i <= a->rows; i++)
		{
			/* the if clause should be unswitched automatically, no need to manually do so */
			if (i == a->rows)
				memset(rows[2], 0, sizeof(int) * a->cols);
			else
				for (j = 0; j < a->cols; j++)
					rows[2][j] = abs(dxi[j]) + abs(dyi[j]);
			int* _dx = dxi - a->cols;
			int* _dy = dyi - a->cols;
			map_ptr[-1] = 0;
			int suppress = 0;
			for (j = 0; j < a->cols; j++)
			{
				int f = rows[1][j];
				if (f > low)
				{
					int x = abs(_dx[j]);
					int y = abs(_dy[j]);
					int s = _dx[j] ^ _dy[j];
					/* x * tan(22.5) */
					int tg22x = x * (int)(0.4142135623730950488016887242097 * (1 << 15) + 0.5);
					/* x * tan(67.5) == 2 * x + x * tan(22.5) */
					int tg67x = tg22x + ((x + x) << 15);
					y <<= 15;
					/* it is a little different from the Canny original paper because we adopted the coordinate system of
					 * top-left corner as origin. Thus, the derivative of y convolved with matrix:
					 * |-1 -2 -1|
					 * | 0  0  0|
					 * | 1  2  1|
					 * actually is the reverse of real y. Thus, the computed angle will be mirrored around x-axis.
					 * In this case, when angle is -45 (135), we compare with north-east and south-west, and for 45,
					 * we compare with north-west and south-east (in traditional coordinate system sense, the same if we
					 * adopt top-left corner as origin for "north", "south", "east", "west" accordingly) */
#define high_block \
					{ \
						if (f > high && !suppress && map_ptr[j - map_cols] != 2) \
						{ \
							map_ptr[j] = 2; \
							suppress = 1; \
							*(stack_top++) = map_ptr + j; \
						} else { \
							map_ptr[j] = 1; \
						} \
						continue; \
					}
					/* sometimes, we end up with same f in integer domain, for that case, we will take the first occurrence
					 * suppressing the second with flag */
					if (y < tg22x)
					{
						if (f > rows[1][j - 1] && f >= rows[1][j + 1])
							high_block;
					} else if (y > tg67x) {
						if (f > rows[0][j] && f >= rows[2][j])
							high_block;
					} else {
						s = s < 0 ? -1 : 1;
						if (f > rows[0][j - s] && f > rows[2][j + s])
							high_block;
					}
#undef high_block
				}
				map_ptr[j] = 0;
				suppress = 0;
			}
			map_ptr[a->cols] = 0;
			map_ptr += map_cols;
			dxi += a->cols;
			dyi += a->cols;
			int* row = rows[0];
			rows[0] = rows[1];
			rows[1] = rows[2];
			rows[2] = row;
		}
		memset(map_ptr - map_cols - 1, 0, sizeof(int) * (a->cols + 2));
		int dr[] = {-1, 1, -map_cols - 1, -map_cols, -map_cols + 1, map_cols - 1, map_cols, map_cols + 1};
		while (stack_top > stack_bottom)
		{
			map_ptr = *(--stack_top);
			for (i = 0; i < 8; i++)
				if (map_ptr[dr[i]] == 1)
				{
					map_ptr[dr[i]] = 2;
					*(stack_top++) = map_ptr + dr[i];
				}
		}
		map_ptr = map + map_cols + 1;
		unsigned char* b_ptr = db->data.ptr;
#define for_block(_, _for_set) \
		for (i = 0; i < a->rows; i++) \
		{ \
			for (j = 0; j < a->cols; j++) \
				_for_set(b_ptr, j, (map_ptr[j] == 2), 0); \
			map_ptr += map_cols; \
			b_ptr += db->step; \
		}
		ccv_matrix_setter(db->type, for_block);
#undef for_block
		ccfree(stack);
		ccfree(map);
		ccv_matrix_free(dx);
		ccv_matrix_free(dy);
	} else {
		/* general case, use all ccv facilities to deal with it */
		ccv_dense_matrix_t* mg = 0;
		ccv_dense_matrix_t* ag = 0;
		ccv_gradient(a, &ag, 0, &mg, 0, size, size);
		ccv_matrix_free(ag);
		ccv_matrix_free(mg);
		/* FIXME: Canny implementation for general case */
	}
}

/* area interpolation resample is adopted from OpenCV */

typedef struct {
	int si, di;
	unsigned int alpha;
} ccv_int_alpha;

static void _ccv_resample_area_8u(ccv_dense_matrix_t* a, ccv_dense_matrix_t* b)
{
	ccv_int_alpha* xofs = (ccv_int_alpha*)alloca(sizeof(ccv_int_alpha) * a->cols * 2);
	int ch = ccv_clamp(CCV_GET_CHANNEL(a->type), 1, 4);
	double scale_x = (double)a->cols / b->cols;
	double scale_y = (double)a->rows / b->rows;
	// double scale = 1.f / (scale_x * scale_y);
	unsigned int inv_scale_256 = (int)(scale_x * scale_y * 0x10000);
	int dx, dy, sx, sy, i, k;
	for (dx = 0, k = 0; dx < b->cols; dx++)
	{
		double fsx1 = dx * scale_x, fsx2 = fsx1 + scale_x;
		int sx1 = (int)(fsx1 + 1.0 - 1e-6), sx2 = (int)(fsx2);
		sx1 = ccv_min(sx1, a->cols - 1);
		sx2 = ccv_min(sx2, a->cols - 1);

		if (sx1 > fsx1)
		{
			xofs[k].di = dx * ch;
			xofs[k].si = (sx1 - 1) * ch;
			xofs[k++].alpha = (unsigned int)((sx1 - fsx1) * 0x100);
		}

		for (sx = sx1; sx < sx2; sx++)
		{
			xofs[k].di = dx * ch;
			xofs[k].si = sx * ch;
			xofs[k++].alpha = 256;
		}

		if (fsx2 - sx2 > 1e-3)
		{
			xofs[k].di = dx * ch;
			xofs[k].si = sx2 * ch;
			xofs[k++].alpha = (unsigned int)((fsx2 - sx2) * 256);
		}
	}
	int xofs_count = k;
	unsigned int* buf = (unsigned int*)alloca(b->cols * ch * sizeof(unsigned int));
	unsigned int* sum = (unsigned int*)alloca(b->cols * ch * sizeof(unsigned int));
	for (dx = 0; dx < b->cols * ch; dx++)
		buf[dx] = sum[dx] = 0;
	dy = 0;
	for (sy = 0; sy < a->rows; sy++)
	{
		unsigned char* a_ptr = a->data.ptr + a->step * sy;
		for (k = 0; k < xofs_count; k++)
		{
			int dxn = xofs[k].di;
			unsigned int alpha = xofs[k].alpha;
			for (i = 0; i < ch; i++)
				buf[dxn + i] += a_ptr[xofs[k].si + i] * alpha;
		}
		if ((dy + 1) * scale_y <= sy + 1 || sy == a->rows - 1)
		{
			unsigned int beta = (int)(ccv_max(sy + 1 - (dy + 1) * scale_y, 0.f) * 256);
			unsigned int beta1 = 256 - beta;
			unsigned char* b_ptr = b->data.ptr + b->step * dy;
			if (beta <= 0)
			{
				for (dx = 0; dx < b->cols * ch; dx++)
				{
					b_ptr[dx] = ccv_clamp((sum[dx] + buf[dx] * 256) / inv_scale_256, 0, 255);
					sum[dx] = buf[dx] = 0;
				}
			} else {
				for (dx = 0; dx < b->cols * ch; dx++)
				{
					b_ptr[dx] = ccv_clamp((sum[dx] + buf[dx] * beta1) / inv_scale_256, 0, 255);
					sum[dx] = buf[dx] * beta;
					buf[dx] = 0;
				}
			}
			dy++;
		}
		else
		{
			for(dx = 0; dx < b->cols * ch; dx++)
			{
				sum[dx] += buf[dx] * 256;
				buf[dx] = 0;
			}
		}
	}
}

typedef struct {
	int si, di;
	float alpha;
} ccv_decimal_alpha;

static void _ccv_resample_area(ccv_dense_matrix_t* a, ccv_dense_matrix_t* b)
{
	ccv_decimal_alpha* xofs = (ccv_decimal_alpha*)alloca(sizeof(ccv_decimal_alpha) * a->cols * 2);
	int ch = ccv_clamp(CCV_GET_CHANNEL(a->type), 1, 4);
	double scale_x = (double)a->cols / b->cols;
	double scale_y = (double)a->rows / b->rows;
	double scale = 1.f / (scale_x * scale_y);
	int dx, dy, sx, sy, i, k;
	for (dx = 0, k = 0; dx < b->cols; dx++)
	{
		double fsx1 = dx * scale_x, fsx2 = fsx1 + scale_x;
		int sx1 = (int)(fsx1 + 1.0 - 1e-6), sx2 = (int)(fsx2);
		sx1 = ccv_min(sx1, a->cols - 1);
		sx2 = ccv_min(sx2, a->cols - 1);

		if (sx1 > fsx1)
		{
			xofs[k].di = dx * ch;
			xofs[k].si = (sx1 - 1) * ch;
			xofs[k++].alpha = (float)((sx1 - fsx1) * scale);
		}

		for (sx = sx1; sx < sx2; sx++)
		{
			xofs[k].di = dx * ch;
			xofs[k].si = sx * ch;
			xofs[k++].alpha = (float)scale;
		}

		if (fsx2 - sx2 > 1e-3)
		{
			xofs[k].di = dx * ch;
			xofs[k].si = sx2 * ch;
			xofs[k++].alpha = (float)((fsx2 - sx2) * scale);
		}
	}
	int xofs_count = k;
	float* buf = (float*)alloca(b->cols * ch * sizeof(float));
	float* sum = (float*)alloca(b->cols * ch * sizeof(float));
	for (dx = 0; dx < b->cols * ch; dx++)
		buf[dx] = sum[dx] = 0;
	dy = 0;
#define for_block(_for_get, _for_set) \
	for (sy = 0; sy < a->rows; sy++) \
	{ \
		unsigned char* a_ptr = a->data.ptr + a->step * sy; \
		for (k = 0; k < xofs_count; k++) \
		{ \
			int dxn = xofs[k].di; \
			float alpha = xofs[k].alpha; \
			for (i = 0; i < ch; i++) \
				buf[dxn + i] += _for_get(a_ptr, xofs[k].si + i, 0) * alpha; \
		} \
		if ((dy + 1) * scale_y <= sy + 1 || sy == a->rows - 1) \
		{ \
			float beta = ccv_max(sy + 1 - (dy + 1) * scale_y, 0.f); \
			float beta1 = 1 - beta; \
			unsigned char* b_ptr = b->data.ptr + b->step * dy; \
			if (fabs(beta) < 1e-3) \
			{ \
				for (dx = 0; dx < b->cols * ch; dx++) \
				{ \
					_for_set(b_ptr, dx, sum[dx] + buf[dx], 0); \
					sum[dx] = buf[dx] = 0; \
				} \
			} else { \
				for (dx = 0; dx < b->cols * ch; dx++) \
				{ \
					_for_set(b_ptr, dx, sum[dx] + buf[dx] * beta1, 0); \
					sum[dx] = buf[dx] * beta; \
					buf[dx] = 0; \
				} \
			} \
			dy++; \
		} \
		else \
		{ \
			for(dx = 0; dx < b->cols * ch; dx++) \
			{ \
				sum[dx] += buf[dx]; \
				buf[dx] = 0; \
			} \
		} \
	}
	ccv_matrix_getter(a->type, ccv_matrix_setter, b->type, for_block);
#undef for_block
}

void ccv_resample(ccv_dense_matrix_t* a, ccv_dense_matrix_t** b, int btype, int rows, int cols, int type)
{
	char identifier[64];
	memset(identifier, 0, 64);
	snprintf(identifier, 64, "ccv_resample(%d,%d,%d)", rows, cols, type);
	uint64_t sig = (a->sig == 0) ? 0 : ccv_matrix_generate_signature(identifier, 64, a->sig, 0);
	btype = (btype == 0) ? CCV_GET_DATA_TYPE(a->type) | CCV_GET_CHANNEL(a->type) : CCV_GET_DATA_TYPE(btype) | CCV_GET_CHANNEL(a->type);
	ccv_dense_matrix_t* db = *b = ccv_dense_matrix_renew(*b, rows, cols, CCV_ALL_DATA_TYPE | CCV_GET_CHANNEL(a->type), btype, sig);
	ccv_cache_return(db, );
	if (a->rows == db->rows && a->cols == db->cols)
	{
		if (CCV_GET_CHANNEL(a->type) == CCV_GET_CHANNEL(db->type) && CCV_GET_DATA_TYPE(db->type) == CCV_GET_DATA_TYPE(a->type))
			memcpy(db->data.ptr, a->data.ptr, a->rows * a->step);
		else {
			ccv_shift(a, (ccv_matrix_t**)&db, 0, 0, 0);
		}
		return;
	}
	switch (type)
	{
		case CCV_INTER_AREA:
			if (a->rows > db->rows && a->cols > db->cols)
			{
				/* using the fast alternative (fix point scale, 0x100 to avoid overflow) */
				if (CCV_GET_DATA_TYPE(a->type) == CCV_8U && CCV_GET_DATA_TYPE(db->type) == CCV_8U && a->rows * a->cols / (db->rows * db->cols) < 0x100)
					_ccv_resample_area_8u(a, db);
				else
					_ccv_resample_area(a, db);
				break;
			}
		case CCV_INTER_LINEAR:
			break;
		case CCV_INTER_CUBIC:
			break;
		case CCV_INTER_LACZOS:
			break;
	}
}

/* the following code is adopted from OpenCV cvPyrDown */
void ccv_sample_down(ccv_dense_matrix_t* a, ccv_dense_matrix_t** b, int type, int src_x, int src_y)
{
	assert(src_x >= 0 && src_y >= 0);
	char identifier[64];
	memset(identifier, 0, 64);
	snprintf(identifier, 64, "ccv_sample_down(%d,%d)", src_x, src_y);
	uint64_t sig = (a->sig == 0) ? 0 : ccv_matrix_generate_signature(identifier, 64, a->sig, 0);
	type = (type == 0) ? CCV_GET_DATA_TYPE(a->type) | CCV_GET_CHANNEL(a->type) : CCV_GET_DATA_TYPE(type) | CCV_GET_CHANNEL(a->type);
	ccv_dense_matrix_t* db = *b = ccv_dense_matrix_renew(*b, a->rows / 2, a->cols / 2, CCV_ALL_DATA_TYPE | CCV_GET_CHANNEL(a->type), type, sig);
	ccv_cache_return(db, );
	int ch = ccv_clamp(CCV_GET_CHANNEL(a->type), 1, 4);
	int cols0 = db->cols - 1 - src_x;
	int dy, sy = -2 + src_y, sx = src_x * ch, dx, k;
	int* tab = (int*)alloca((a->cols + src_x + 2) * ch * sizeof(int));
	for (dx = 0; dx < a->cols + src_x + 2; dx++)
		for (k = 0; k < ch; k++)
			tab[dx * ch + k] = ((dx >= a->cols) ? a->cols * 2 - 1 - dx : dx) * ch + k;
	unsigned char* buf = (unsigned char*)alloca(5 * db->cols * ch * ccv_max(CCV_GET_DATA_TYPE_SIZE(db->type), sizeof(int)));
	int bufstep = db->cols * ch * ccv_max(CCV_GET_DATA_TYPE_SIZE(db->type), sizeof(int));
	unsigned char* b_ptr = db->data.ptr;
	/* why is src_y * 4 in computing the offset of row?
	 * Essentially, it means sy - src_y but in a manner that doesn't result negative number.
	 * notice that we added src_y before when computing sy in the first place, however,
	 * it is not desirable to have that offset when we try to wrap it into our 5-row buffer (
	 * because in later rearrangement, we have no src_y to backup the arrangement). In
	 * such micro scope, we managed to stripe 5 addition into one shift and addition. */
#define for_block(x_block, _for_get_a, _for_get, _for_set, _for_set_b) \
	for (dy = 0; dy < db->rows; dy++) \
	{ \
		for(; sy <= dy * 2 + 2 + src_y; sy++) \
		{ \
			unsigned char* row = buf + ((sy + src_y * 4 + 2) % 5) * bufstep; \
			int _sy = (sy < 0) ? -1 - sy : (sy >= a->rows) ? a->rows * 2 - 1 - sy : sy; \
			unsigned char* a_ptr = a->data.ptr + a->step * _sy; \
			for (k = 0; k < ch; k++) \
				_for_set(row, k, _for_get_a(a_ptr, sx + k, 0) * 10 + _for_get_a(a_ptr, ch + sx + k, 0) * 5 + _for_get_a(a_ptr, 2 * ch + sx + k, 0), 0); \
			for(dx = ch; dx < cols0 * ch; dx += ch) \
				for (k = 0; k < ch; k++) \
					_for_set(row, dx + k, _for_get_a(a_ptr, dx * 2 + sx + k, 0) * 6 + (_for_get_a(a_ptr, dx * 2 + sx + k - ch, 0) + _for_get_a(a_ptr, dx * 2 + sx + k + ch, 0)) * 4 + _for_get_a(a_ptr, dx * 2 + sx + k - ch * 2, 0) + _for_get_a(a_ptr, dx * 2 + sx + k + ch * 2, 0), 0); \
			x_block(_for_get_a, _for_get, _for_set, _for_set_b); \
		} \
		unsigned char* rows[5]; \
		for(k = 0; k < 5; k++) \
			rows[k] = buf + ((dy * 2 + k) % 5) * bufstep; \
		for(dx = 0; dx < db->cols * ch; dx++) \
			_for_set_b(b_ptr, dx, (_for_get(rows[2], dx, 0) * 6 + (_for_get(rows[1], dx, 0) + _for_get(rows[3], dx, 0)) * 4 + _for_get(rows[0], dx, 0) + _for_get(rows[4], dx, 0)) / 256, 0); \
		b_ptr += db->step; \
	}
	int no_8u_type = (a->type & CCV_8U) ? CCV_32S : a->type;
	/* here is the new technique to expand for loop with condition in manual way */
	if (src_x > 0)
	{
#define x_block(_for_get_a, _for_get, _for_set, _for_set_b) \
		for (dx = cols0 * ch; dx < db->cols * ch; dx += ch) \
			for (k = 0; k < ch; k++) \
				_for_set(row, dx + k, _for_get_a(a_ptr, tab[dx * 2 + sx + k], 0) * 6 + (_for_get_a(a_ptr, tab[dx * 2 + sx + k - ch], 0) + _for_get_a(a_ptr, tab[dx * 2 + sx + k + ch], 0)) * 4 + _for_get_a(a_ptr, tab[dx * 2 + sx + k - ch * 2], 0) + _for_get_a(a_ptr, tab[dx * 2 + sx + k + ch * 2], 0), 0);
		ccv_unswitch_block(x_block, ccv_matrix_getter_a, a->type, ccv_matrix_getter, no_8u_type, ccv_matrix_setter, no_8u_type, ccv_matrix_setter_b, db->type, for_block);
#undef x_block
	} else {
#define x_block(_for_get_a, _for_get, _for_set, _for_set_b) \
		for (k = 0; k < ch; k++) \
			_for_set(row, (db->cols - 1) * ch + k, _for_get_a(a_ptr, a->cols * ch + sx - ch + k, 0) * 10 + _for_get_a(a_ptr, (a->cols - 2) * ch + sx + k, 0) * 5 + _for_get_a(a_ptr, (a->cols - 3) * ch + sx + k, 0), 0);
		ccv_unswitch_block(x_block, ccv_matrix_getter_a, a->type, ccv_matrix_getter, no_8u_type, ccv_matrix_setter, no_8u_type, ccv_matrix_setter_b, db->type, for_block);
#undef x_block
	}
#undef for_block
}

void ccv_sample_up(ccv_dense_matrix_t* a, ccv_dense_matrix_t** b, int type, int src_x, int src_y)
{
	assert(src_x >= 0 && src_y >= 0);
	char identifier[64];
	memset(identifier, 0, 64);
	snprintf(identifier, 64, "ccv_sample_up(%d,%d)", src_x, src_y);
	uint64_t sig = (a->sig == 0) ? 0 : ccv_matrix_generate_signature(identifier, 64, a->sig, 0);
	type = (type == 0) ? CCV_GET_DATA_TYPE(a->type) | CCV_GET_CHANNEL(a->type) : CCV_GET_DATA_TYPE(type) | CCV_GET_CHANNEL(a->type);
	ccv_dense_matrix_t* db = *b = ccv_dense_matrix_renew(*b, a->rows * 2, a->cols * 2, CCV_ALL_DATA_TYPE | CCV_GET_CHANNEL(a->type), type, sig);
	ccv_cache_return(db, );
	int ch = ccv_clamp(CCV_GET_CHANNEL(a->type), 1, 4);
	int cols0 = a->cols - 1 - src_x;
	int y, x, sy = -1 + src_y, sx = src_x * ch, k;
	int* tab = (int*)alloca((a->cols + src_x + 2) * ch * sizeof(int));
	for (x = 0; x < a->cols + src_x + 2; x++)
		for (k = 0; k < ch; k++)
			tab[x * ch + k] = ((x >= a->cols) ? a->cols * 2 - 1 - x : x) * ch + k;
	unsigned char* buf = (unsigned char*)alloca(3 * db->cols * ch * ccv_max(CCV_GET_DATA_TYPE_SIZE(db->type), sizeof(int)));
	int bufstep = db->cols * ch * ccv_max(CCV_GET_DATA_TYPE_SIZE(db->type), sizeof(int));
	unsigned char* b_ptr = db->data.ptr;
	/* why src_y * 2: the same argument as in ccv_sample_down */
#define for_block(x_block, _for_get_a, _for_get, _for_set, _for_set_b) \
	for (y = 0; y < a->rows; y++) \
	{ \
		for (; sy <= y + 1 + src_y; sy++) \
		{ \
			unsigned char* row = buf + ((sy + src_y * 2 + 1) % 3) * bufstep; \
			int _sy = (sy < 0) ? -1 - sy : (sy >= a->rows) ? a->rows * 2 - 1 - sy : sy; \
			unsigned char* a_ptr = a->data.ptr + a->step * _sy; \
			if (a->cols == 1) \
			{ \
				for (k = 0; k < ch; k++) \
				{ \
					_for_set(row, k, _for_get_a(a_ptr, k, 0) * 8, 0); \
					_for_set(row, k + ch, _for_get_a(a_ptr, k, 0) * 8, 0); \
				} \
				continue; \
			} \
			for (k = 0; k < ch; k++) \
			{ \
				_for_set(row, k, _for_get_a(a_ptr, k + sx, 0) * 6 + _for_get_a(a_ptr, k + sx + ch, 0) * 2, 0); \
				_for_set(row, k + ch, (_for_get_a(a_ptr, k + sx, 0) + _for_get_a(a_ptr, k + sx + ch, 0)) * 4, 0); \
			} \
			for (x = ch; x < cols0 * ch; x += ch) \
			{ \
				for (k = 0; k < ch; k++) \
				{ \
					_for_set(row, x * 2 + k, _for_get_a(a_ptr, x + sx - ch + k, 0) + _for_get_a(a_ptr, x + sx + k, 0) * 6 + _for_get_a(a_ptr, x + sx + ch + k, 0), 0); \
					_for_set(row, x * 2 + ch + k, (_for_get_a(a_ptr, x + sx + k, 0) + _for_get_a(a_ptr, x + sx + ch + k, 0)) * 4, 0); \
				} \
			} \
			x_block(_for_get_a, _for_get, _for_set, _for_set_b); \
		} \
		unsigned char* rows[3]; \
		for (k = 0; k < 3; k++) \
			rows[k] = buf + ((y + k) % 3) * bufstep; \
		for (x = 0; x < db->cols * ch; x++) \
		{ \
			_for_set_b(b_ptr, x, (_for_get(rows[0], x, 0) + _for_get(rows[1], x, 0) * 2 + _for_get(rows[2], x, 0)) / 32, 0); \
			_for_set_b(b_ptr + db->step, x, (_for_get(rows[1], x, 0) + _for_get(rows[2], x, 0)) / 16, 0); \
		} \
		b_ptr += 2 * db->step; \
	}
	int no_8u_type = (a->type & CCV_8U) ? CCV_32S : a->type;
	/* unswitch if condition in manual way */
	if (src_x > 0)
	{
#define x_block(_for_get_a, _for_get, _for_set, _for_set_b) \
		for (x = cols0 * ch; x < a->cols * ch; x += ch) \
			for (k = 0; k < ch; k++) \
			{ \
				_for_set(row, x * 2 + k, _for_get_a(a_ptr, tab[x + sx - ch + k], 0) + _for_get_a(a_ptr, tab[x + sx + k], 0) * 6 + _for_get_a(a_ptr, tab[x + sx + ch + k], 0), 0); \
				_for_set(row, x * 2 + ch + k, (_for_get_a(a_ptr, tab[x + sx + k], 0) + _for_get_a(a_ptr, tab[x + sx + ch + k], 0)) * 4, 0); \
			}
		ccv_unswitch_block(x_block, ccv_matrix_getter_a, a->type, ccv_matrix_getter, no_8u_type, ccv_matrix_setter, no_8u_type, ccv_matrix_setter_b, db->type, for_block);
#undef x_block
	} else {
#define x_block(_for_get_a, _for_get, _for_set, _for_set_b) \
		for (k = 0; k < ch; k++) \
		{ \
			_for_set(row, (a->cols - 1) * 2 * ch + k, _for_get_a(a_ptr, (a->cols - 2) * ch + k, 0) + _for_get_a(a_ptr, (a->cols - 1) * ch + k, 0) * 7, 0); \
			_for_set(row, (a->cols - 1) * 2 * ch + ch + k, _for_get_a(a_ptr, (a->cols - 1) * ch + k, 0) * 4, 0); \
		}
		ccv_unswitch_block(x_block, ccv_matrix_getter_a, a->type, ccv_matrix_getter, no_8u_type, ccv_matrix_setter, no_8u_type, ccv_matrix_setter_b, db->type, for_block);
#undef x_block
	}
#undef for_block
}

void _ccv_flip_y_self(ccv_dense_matrix_t* a)
{
	int i;
	unsigned char* buffer = (unsigned char*)alloca(a->step);
	unsigned char* a_ptr = a->data.ptr;
	unsigned char* b_ptr = a->data.ptr + (a->rows - 1) * a->step;
	for (i = 0; i < a->rows / 2; i++)
	{
		memcpy(buffer, a_ptr, a->step);
		memcpy(a_ptr, b_ptr, a->step);
		memcpy(b_ptr, buffer, a->step);
		a_ptr += a->step;
		b_ptr -= a->step;
	}
}

void _ccv_flip_x_self(ccv_dense_matrix_t* a)
{
	int i, j;
	int len = CCV_GET_DATA_TYPE_SIZE(a->type) * CCV_GET_CHANNEL(a->type);
	unsigned char* buffer = (unsigned char*)alloca(len);
	unsigned char* a_ptr = a->data.ptr;
	for (i = 0; i < a->rows; i++)
	{
		for (j = 0; j < a->cols / 2; j++)
		{
			memcpy(buffer, a_ptr + j * len, len);
			memcpy(a_ptr + j * len, a_ptr + (a->cols - 1 - j) * len, len);
			memcpy(a_ptr + (a->cols - 1 - j) * len, buffer, len);
		}
		a_ptr += a->step;
	}
}

void ccv_flip(ccv_dense_matrix_t* a, ccv_dense_matrix_t** b, int btype, int type)
{
	uint64_t sig = a->sig;
	if (type & CCV_FLIP_Y)
		sig = (a->sig == 0) ? 0 : ccv_matrix_generate_signature("ccv_flip_y", 10, sig, 0);
	if (type & CCV_FLIP_X)
		sig = (a->sig == 0) ? 0 : ccv_matrix_generate_signature("ccv_flip_x", 10, sig, 0);
	ccv_dense_matrix_t* db;
	if (b == 0)
	{
		db = a;
		if (!(a->sig == 0))
		{
			btype = CCV_GET_DATA_TYPE(a->type) | CCV_GET_CHANNEL(a->type);
			sig = ccv_matrix_generate_signature((const char*)&btype, sizeof(int), sig, 0);
			a->sig = sig;
		}
	} else {
		btype = CCV_GET_DATA_TYPE(a->type) | CCV_GET_CHANNEL(a->type);
		*b = db = ccv_dense_matrix_renew(*b, a->rows, a->cols, btype, btype, sig);
		ccv_cache_return(db, );
		memcpy(db->data.ptr, a->data.ptr, a->rows * a->step);
	}
	if (type & CCV_FLIP_Y)
		_ccv_flip_y_self(db);
	if (type & CCV_FLIP_X)
		_ccv_flip_x_self(db);
}

void ccv_blur(ccv_dense_matrix_t* a, ccv_dense_matrix_t** b, int type, double sigma)
{
	char identifier[64];
	memset(identifier, 0, 64);
	snprintf(identifier, 64, "ccv_blur(%lf)", sigma);
	uint64_t sig = (a->sig == 0) ? 0 : ccv_matrix_generate_signature(identifier, 64, a->sig, 0);
	type = (type == 0) ? CCV_GET_DATA_TYPE(a->type) | CCV_GET_CHANNEL(a->type) : CCV_GET_DATA_TYPE(type) | CCV_GET_CHANNEL(a->type);
	ccv_dense_matrix_t* db = *b = ccv_dense_matrix_renew(*b, a->rows, a->cols, CCV_ALL_DATA_TYPE | CCV_GET_CHANNEL(a->type), type, sig);
	ccv_cache_return(db, );
	int fsz = ccv_max(1, (int)(4.0 * sigma + 1.0 - 1e-8)) * 2 + 1;
	int hfz = fsz / 2;
	unsigned char* buf = (unsigned char*)alloca(sizeof(double) * (fsz + ccv_max(a->rows, a->cols * CCV_GET_CHANNEL(a->type))));
	unsigned char* filter = (unsigned char*)alloca(sizeof(double) * fsz);
	double tw = 0;
	int i, j, k, ch = CCV_GET_CHANNEL(a->type);
	for (i = 0; i < fsz; i++)
		tw += ((double*)filter)[i] = exp(-((i - hfz) * (i - hfz)) / (2.0 * sigma * sigma));
	int no_8u_type = (db->type & CCV_8U) ? CCV_32S : db->type;
	if (no_8u_type & CCV_32S)
	{
		tw = 256.0 / tw;
		for (i = 0; i < fsz; i++)
			((int*)filter)[i] = (int)(((double*)filter)[i] * tw + 0.5);
	} else {
		tw = 1.0 / tw;
		for (i = 0; i < fsz; i++)
			ccv_set_value(db->type, filter, i, ((double*)filter)[i] * tw, 0);
	}
	/* horizontal */
	unsigned char* a_ptr = a->data.ptr;
	unsigned char* b_ptr = db->data.ptr;
#define for_block(_for_type, _for_set_b, _for_get_b, _for_set_a, _for_get_a) \
	for (i = 0; i < a->rows; i++) \
	{ \
		for (j = 0; j < hfz; j++) \
			for (k = 0; k < ch; k++) \
				_for_set_b(buf, j * ch + k, _for_get_a(a_ptr, k, 0), 0); \
		for (j = 0; j < a->cols * ch; j++) \
			_for_set_b(buf, j + hfz * ch, _for_get_a(a_ptr, j, 0), 0); \
		for (j = a->cols; j < hfz + a->cols; j++) \
			for (k = 0; k < ch; k++) \
				_for_set_b(buf, j * ch + hfz * ch + k, _for_get_a(a_ptr, (a->cols - 1) * ch + k, 0), 0); \
		for (j = 0; j < a->cols * ch; j++) \
		{ \
			_for_type sum = 0; \
			for (k = 0; k < fsz; k++) \
				sum += _for_get_b(buf, k * ch + j, 0) * _for_get_b(filter, k, 0); \
			_for_set_b(buf, j, sum, 8); \
		} \
		for (j = 0; j < a->cols * ch; j++) \
			_for_set_a(b_ptr, j, _for_get_b(buf, j, 0), 0); \
		a_ptr += a->step; \
		b_ptr += db->step; \
	}
	ccv_matrix_typeof_setter_getter(no_8u_type, ccv_matrix_setter, db->type, ccv_matrix_getter, a->type, for_block);
#undef for_block
	/* vertical */
	b_ptr = db->data.ptr;
#define for_block(_for_type, _for_set_b, _for_get_b, _for_set_a, _for_get_a) \
	for (i = 0; i < a->cols * ch; i++) \
	{ \
		for (j = 0; j < hfz; j++) \
			_for_set_b(buf, j, _for_get_a(b_ptr, i, 0), 0); \
		for (j = 0; j < a->rows; j++) \
			_for_set_b(buf, j + hfz, _for_get_a(b_ptr + j * db->step, i, 0), 0); \
		for (j = a->rows; j < hfz + a->rows; j++) \
			_for_set_b(buf, j + hfz, _for_get_a(b_ptr + (a->rows - 1) * db->step, i, 0), 0); \
		for (j = 0; j < a->rows; j++) \
		{ \
			_for_type sum = 0; \
			for (k = 0; k < fsz; k++) \
				sum += _for_get_b(buf, k + j, 0) * _for_get_b(filter, k, 0); \
			_for_set_b(buf, j, sum, 8); \
		} \
		for (j = 0; j < a->rows; j++) \
			_for_set_a(b_ptr + j * db->step, i, _for_get_b(buf, j, 0), 0); \
	}
	ccv_matrix_typeof_setter_getter(no_8u_type, ccv_matrix_setter_getter, db->type, for_block);
#undef for_block
}

int ccv_otsu(ccv_dense_matrix_t* a, double* outvar, int range)
{
	assert((a->type & CCV_32S) || (a->type & CCV_8U));
	int* histogram = (int*)alloca(range * sizeof(int));
	memset(histogram, 0, sizeof(int) * range);
	int i, j;
	unsigned char* a_ptr = a->data.ptr;
#define for_block(_, _for_get) \
	for (i = 0; i < a->rows; i++) \
	{ \
		for (j = 0; j < a->cols; j++) \
			histogram[ccv_clamp((int)_for_get(a_ptr, j, 0), 0, range - 1)]++; \
		a_ptr += a->step; \
	}
	ccv_matrix_getter(a->type, for_block);
#undef for_block
	double sum = 0, sumB = 0;
	for (i = 0; i < range; i++)
		sum += i * histogram[i];
	int wB = 0, wF = 0, total = a->rows * a->cols;
	double maxVar = 0;
	int threshold = 0;
	for (i = 0; i < range; i++)
	{
		wB += histogram[i];
		if (wB == 0)
			continue;
		wF = total - wB;
		if (wF == 0)
			break;
		sumB += i * histogram[i];
		double mB = sumB / wB;
		double mF = (sum - sumB) / wF;
		double var = wB * wF * (mB - mF) * (mB - mF);
		if (var > maxVar)
		{
			maxVar = var;
			threshold = i;
		}
	}
	if (outvar != 0)
		*outvar = maxVar / total / total;
	return threshold;
}

/* End of ccv/lib/ccv_basic.c */
#line 1 "ccv/lib/ccv_bbf.c"
/* ccv.h already included */

#ifdef HAVE_GSL
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#endif
#include <sys/time.h>
#ifdef USE_OPENMP
#include <omp.h>
#endif
#ifdef USE_OPENCL
#include <CL/cl.h>
#endif

#define _ccv_width_padding(x) (((x) + 3) & -4)

static unsigned int _ccv_bbf_time_measure()
{
	struct timeval tv;
	gettimeofday(&tv, 0);
	return tv.tv_sec * 1000000 + tv.tv_usec;
}

static inline int _ccv_run_bbf_feature(ccv_bbf_feature_t* feature, int* step, unsigned char** u8)
{
#define pf_at(i) (*(u8[feature->pz[i]] + feature->px[i] + feature->py[i] * step[feature->pz[i]]))
#define nf_at(i) (*(u8[feature->nz[i]] + feature->nx[i] + feature->ny[i] * step[feature->nz[i]]))
	unsigned char pmin = pf_at(0), nmax = nf_at(0);
	/* check if every point in P > every point in N, and take a shortcut */
	if (pmin <= nmax)
		return 0;
	int i;
	for (i = 1; i < feature->size; i++)
	{
		if (feature->pz[i] >= 0)
		{
			int p = pf_at(i);
			if (p < pmin)
			{
				if (p <= nmax)
					return 0;
				pmin = p;
			}
		}
		if (feature->nz[i] >= 0)
		{
			int n = nf_at(i);
			if (n > nmax)
			{
				if (pmin <= n)
					return 0;
				nmax = n;
			}
		}
	}
#undef pf_at
#undef nf_at
	return 1;
}

#define less_than(a, b, aux) ((a) < (b))
CCV_IMPLEMENT_QSORT(_ccv_sort_32f, float, less_than)
#undef less_than

static void _ccv_bbf_eval_data(ccv_bbf_stage_classifier_t* classifier, unsigned char** posdata, int posnum, unsigned char** negdata, int negnum, ccv_size_t size, float* peval, float* neval)
{
	int i, j;
	int steps[] = { _ccv_width_padding(size.width),
					_ccv_width_padding(size.width >> 1),
					_ccv_width_padding(size.width >> 2) };
	int isizs0 = steps[0] * size.height;
	int isizs01 = isizs0 + steps[1] * (size.height >> 1);
	for (i = 0; i < posnum; i++)
	{
		unsigned char* u8[] = { posdata[i], posdata[i] + isizs0, posdata[i] + isizs01 };
		float sum = 0;
		float* alpha = classifier->alpha;
		ccv_bbf_feature_t* feature = classifier->feature;
		for (j = 0; j < classifier->count; ++j, alpha += 2, ++feature)
			sum += alpha[_ccv_run_bbf_feature(feature, steps, u8)];
		peval[i] = sum;
	}
	for (i = 0; i < negnum; i++)
	{
		unsigned char* u8[] = { negdata[i], negdata[i] + isizs0, negdata[i] + isizs01 };
		float sum = 0;
		float* alpha = classifier->alpha;
		ccv_bbf_feature_t* feature = classifier->feature;
		for (j = 0; j < classifier->count; ++j, alpha += 2, ++feature)
			sum += alpha[_ccv_run_bbf_feature(feature, steps, u8)];
		neval[i] = sum;
	}
}

static int _ccv_prune_positive_data(ccv_bbf_classifier_cascade_t* cascade, unsigned char** posdata, int posnum, ccv_size_t size)
{
	float* peval = (float*)ccmalloc(posnum * sizeof(float));
	int i, j, k, rpos = posnum;
	for (i = 0; i < cascade->count; i++)
	{
		_ccv_bbf_eval_data(cascade->stage_classifier + i, posdata, rpos, 0, 0, size, peval, 0);
		k = 0;
		for (j = 0; j < rpos; j++)
			if (peval[j] >= cascade->stage_classifier[i].threshold)
			{
				posdata[k] = posdata[j];
				++k;
			} else {
				ccfree(posdata[j]);
			}
		rpos = k;
	}
	ccfree(peval);
	return rpos;
}

static int _ccv_prepare_background_data(ccv_bbf_classifier_cascade_t* cascade, char** bgfiles, int bgnum, unsigned char** negdata, int negnum)
{
#ifdef HAVE_GSL
	int t, i, j, k, q;
	int negperbg = negnum / bgnum + 1;
	int negtotal = 0;
	int steps[] = { _ccv_width_padding(cascade->size.width),
					_ccv_width_padding(cascade->size.width >> 1),
					_ccv_width_padding(cascade->size.width >> 2) };
	int isizs0 = steps[0] * cascade->size.height;
	int isizs1 = steps[1] * (cascade->size.height >> 1);
	int isizs2 = steps[2] * (cascade->size.height >> 2);
	int* idcheck = (int*)ccmalloc(negnum * sizeof(int));

	gsl_rng_env_setup();

	gsl_rng* rng = gsl_rng_alloc(gsl_rng_default);
	gsl_rng_set(rng, (unsigned long int)idcheck);

	ccv_size_t imgsz = cascade->size;
	int rneg = negtotal;
	for (t = 0; negtotal < negnum; t++)
	{
		printf("preparing negative data ...  0%%");
		for (i = 0; i < bgnum; i++)
		{
			negperbg = (t < 2) ? (negnum - negtotal) / (bgnum - i) + 1 : negnum - negtotal;
			ccv_dense_matrix_t* image = 0;
			ccv_unserialize(bgfiles[i], &image, CCV_SERIAL_GRAY | CCV_SERIAL_ANY_FILE);
			assert((image->type & CCV_C1) && (image->type & CCV_8U));
			if (image == 0)
			{
				printf("\n%s file corrupted\n", bgfiles[i]);
				continue;
			}
			if (t % 2 != 0)
				ccv_flip(image, 0, 0, CCV_FLIP_X);
			if (t % 4 >= 2)
				ccv_flip(image, 0, 0, CCV_FLIP_Y);
			ccv_bbf_param_t params = { .interval = 3, .min_neighbors = 0, .flags = 0, .size = cascade->size };
			ccv_array_t* detected = ccv_bbf_detect_objects(image, &cascade, 1, params);
			memset(idcheck, 0, ccv_min(detected->rnum, negperbg) * sizeof(int));
			for (j = 0; j < ccv_min(detected->rnum, negperbg); j++)
			{
				int r = gsl_rng_uniform_int(rng, detected->rnum);
				int flag = 1;
				ccv_rect_t* rect = (ccv_rect_t*)ccv_array_get(detected, r);
				while (flag) {
					flag = 0;
					for (k = 0; k < j; k++)
						if (r == idcheck[k])
						{
							flag = 1;
							r = gsl_rng_uniform_int(rng, detected->rnum);
							break;
						}
					rect = (ccv_rect_t*)ccv_array_get(detected, r);
					if ((rect->x < 0) || (rect->y < 0) || (rect->width + rect->x >= image->cols) || (rect->height + rect->y >= image->rows))
					{
						flag = 1;
						r = gsl_rng_uniform_int(rng, detected->rnum);
					}
				}
				idcheck[j] = r;
				ccv_dense_matrix_t* temp = 0;
				ccv_dense_matrix_t* imgs0 = 0;
				ccv_dense_matrix_t* imgs1 = 0;
				ccv_dense_matrix_t* imgs2 = 0;
				ccv_slice(image, (ccv_matrix_t**)&temp, 0, rect->y, rect->x, rect->height, rect->width);
				ccv_resample(temp, &imgs0, 0, imgsz.height, imgsz.width, CCV_INTER_AREA);
				assert(imgs0->step == steps[0]);
				ccv_matrix_free(temp);
				ccv_sample_down(imgs0, &imgs1, 0, 0, 0);
				assert(imgs1->step == steps[1]);
				ccv_sample_down(imgs1, &imgs2, 0, 0, 0);
				assert(imgs2->step == steps[2]);

				negdata[negtotal] = (unsigned char*)ccmalloc(isizs0 + isizs1 + isizs2);
				unsigned char* u8s0 = negdata[negtotal];
				unsigned char* u8s1 = negdata[negtotal] + isizs0;
				unsigned char* u8s2 = negdata[negtotal] + isizs0 + isizs1;
				unsigned char* u8[] = { u8s0, u8s1, u8s2 };
				memcpy(u8s0, imgs0->data.ptr, imgs0->rows * imgs0->step);
				ccv_matrix_free(imgs0);
				memcpy(u8s1, imgs1->data.ptr, imgs1->rows * imgs1->step);
				ccv_matrix_free(imgs1);
				memcpy(u8s2, imgs2->data.ptr, imgs2->rows * imgs2->step);
				ccv_matrix_free(imgs2);

				flag = 1;
				ccv_bbf_stage_classifier_t* classifier = cascade->stage_classifier;
				for (k = 0; k < cascade->count; ++k, ++classifier)
				{
					float sum = 0;
					float* alpha = classifier->alpha;
					ccv_bbf_feature_t* feature = classifier->feature;
					for (q = 0; q < classifier->count; ++q, alpha += 2, ++feature)
						sum += alpha[_ccv_run_bbf_feature(feature, steps, u8)];
					if (sum < classifier->threshold)
					{
						flag = 0;
						break;
					}
				}
				if (!flag)
					ccfree(negdata[negtotal]);
				else {
					++negtotal;
					if (negtotal >= negnum)
						break;
				}
			}
			ccv_array_free(detected);
			ccv_matrix_free(image);
			ccv_drain_cache();
			printf("\rpreparing negative data ... %2d%%", 100 * negtotal / negnum);
			fflush(0);
			if (negtotal >= negnum)
				break;
		}
		if (rneg == negtotal)
			break;
		rneg = negtotal;
		printf("\nentering additional round %d\n", t + 1);
	}
	gsl_rng_free(rng);
	ccfree(idcheck);
	ccv_drain_cache();
	printf("\n");
	return negtotal;
#else
	return 0;
#endif
}

static void _ccv_prepare_positive_data(ccv_dense_matrix_t** posimg, unsigned char** posdata, ccv_size_t size, int posnum)
{
	printf("preparing positive data ...  0%%");
	int i;
	for (i = 0; i < posnum; i++)
	{
		ccv_dense_matrix_t* imgs0 = posimg[i];
		ccv_dense_matrix_t* imgs1 = 0;
		ccv_dense_matrix_t* imgs2 = 0;
		assert((imgs0->type & CCV_C1) && (imgs0->type & CCV_8U) && imgs0->rows == size.height && imgs0->cols == size.width);
		ccv_sample_down(imgs0, &imgs1, 0, 0, 0);
		ccv_sample_down(imgs1, &imgs2, 0, 0, 0);
		int isizs0 = imgs0->rows * imgs0->step;
		int isizs1 = imgs1->rows * imgs1->step;
		int isizs2 = imgs2->rows * imgs2->step;

		posdata[i] = (unsigned char*)ccmalloc(isizs0 + isizs1 + isizs2);
		memcpy(posdata[i], imgs0->data.ptr, isizs0);
		memcpy(posdata[i] + isizs0, imgs1->data.ptr, isizs1);
		memcpy(posdata[i] + isizs0 + isizs1, imgs2->data.ptr, isizs2);

		printf("\rpreparing positive data ... %2d%%", 100 * (i + 1) / posnum);
		fflush(0);

		ccv_matrix_free(imgs1);
		ccv_matrix_free(imgs2);
	}
	ccv_drain_cache();
	printf("\n");
}

typedef struct {
	double fitness;
	int pk, nk;
	int age;
	double error;
	ccv_bbf_feature_t feature;
} ccv_bbf_gene_t;

static inline void _ccv_bbf_genetic_fitness(ccv_bbf_gene_t* gene)
{
	gene->fitness = (1 - gene->error) * exp(-0.01 * gene->age) * exp((gene->pk + gene->nk) * log(1.015));
}

static inline int _ccv_bbf_exist_gene_feature(ccv_bbf_gene_t* gene, int x, int y, int z)
{
	int i;
	for (i = 0; i < gene->pk; i++)
		if (z == gene->feature.pz[i] && x == gene->feature.px[i] && y == gene->feature.py[i])
			return 1;
	for (i = 0; i < gene->nk; i++)
		if (z == gene->feature.nz[i] && x == gene->feature.nx[i] && y == gene->feature.ny[i])
			return 1;
	return 0;
}

#ifdef HAVE_GSL
static inline void _ccv_bbf_randomize_gene(gsl_rng* rng, ccv_bbf_gene_t* gene, int* rows, int* cols)
{
	int i;
	do {
		gene->pk = gsl_rng_uniform_int(rng, CCV_BBF_POINT_MAX - 1) + 1;
		gene->nk = gsl_rng_uniform_int(rng, CCV_BBF_POINT_MAX - 1) + 1;
	} while (gene->pk + gene->nk < CCV_BBF_POINT_MIN); /* a hard restriction of at least 3 points have to be examed */
	gene->feature.size = ccv_max(gene->pk, gene->nk);
	gene->age = 0;
	for (i = 0; i < CCV_BBF_POINT_MAX; i++)
	{
		gene->feature.pz[i] = -1;
		gene->feature.nz[i] = -1;
	}
	int x, y, z;
	for (i = 0; i < gene->pk; i++)
	{
		do {
			z = gsl_rng_uniform_int(rng, 3);
			x = gsl_rng_uniform_int(rng, cols[z]);
			y = gsl_rng_uniform_int(rng, rows[z]);
		} while (_ccv_bbf_exist_gene_feature(gene, x, y, z));
		gene->feature.pz[i] = z;
		gene->feature.px[i] = x;
		gene->feature.py[i] = y;
	}
	for (i = 0; i < gene->nk; i++)
	{
		do {
			z = gsl_rng_uniform_int(rng, 3);
			x = gsl_rng_uniform_int(rng, cols[z]);
			y = gsl_rng_uniform_int(rng, rows[z]);
		} while ( _ccv_bbf_exist_gene_feature(gene, x, y, z));
		gene->feature.nz[i] = z;
		gene->feature.nx[i] = x;
		gene->feature.ny[i] = y;
	}
}
#endif

static inline double _ccv_bbf_error_rate(ccv_bbf_feature_t* feature, unsigned char** posdata, int posnum, unsigned char** negdata, int negnum, ccv_size_t size, double* pw, double* nw)
{
	int i;
	int steps[] = { _ccv_width_padding(size.width),
					_ccv_width_padding(size.width >> 1),
					_ccv_width_padding(size.width >> 2) };
	int isizs0 = steps[0] * size.height;
	int isizs01 = isizs0 + steps[1] * (size.height >> 1);
	double error = 0;
	for (i = 0; i < posnum; i++)
	{
		unsigned char* u8[] = { posdata[i], posdata[i] + isizs0, posdata[i] + isizs01 };
		if (!_ccv_run_bbf_feature(feature, steps, u8))
			error += pw[i];
	}
	for (i = 0; i < negnum; i++)
	{
		unsigned char* u8[] = { negdata[i], negdata[i] + isizs0, negdata[i] + isizs01 };
		if ( _ccv_run_bbf_feature(feature, steps, u8))
			error += nw[i];
	}
	return error;
}

#define less_than(fit1, fit2, aux) ((fit1).fitness >= (fit2).fitness)
static CCV_IMPLEMENT_QSORT(_ccv_bbf_genetic_qsort, ccv_bbf_gene_t, less_than)
#undef less_than

static ccv_bbf_feature_t _ccv_bbf_genetic_optimize(unsigned char** posdata, int posnum, unsigned char** negdata, int negnum, int ftnum, ccv_size_t size, double* pw, double* nw)
{
	ccv_bbf_feature_t best;
#ifdef HAVE_GSL
	/* seed (random method) */
	gsl_rng_env_setup();
	gsl_rng* rng = gsl_rng_alloc(gsl_rng_default);
	union { unsigned long int li; double db; } dbli;
	dbli.db = pw[0] + nw[0];
	gsl_rng_set(rng, dbli.li);
	int i, j;
	int pnum = ftnum * 100;
	ccv_bbf_gene_t* gene = (ccv_bbf_gene_t*)ccmalloc(pnum * sizeof(ccv_bbf_gene_t));
	int rows[] = { size.height, size.height >> 1, size.height >> 2 };
	int cols[] = { size.width, size.width >> 1, size.width >> 2 };
	for (i = 0; i < pnum; i++)
		_ccv_bbf_randomize_gene(rng, &gene[i], rows, cols);
	unsigned int timer = _ccv_bbf_time_measure();
#ifdef USE_OPENMP
#pragma omp parallel for private(i) schedule(dynamic)
#endif
	for (i = 0; i < pnum; i++)
		gene[i].error = _ccv_bbf_error_rate(&gene[i].feature, posdata, posnum, negdata, negnum, size, pw, nw);
	timer = _ccv_bbf_time_measure() - timer;
	for (i = 0; i < pnum; i++)
		_ccv_bbf_genetic_fitness(&gene[i]);
	double best_err = 1;
	int rnum = ftnum * 39; /* number of randomize */
	int mnum = ftnum * 40; /* number of mutation */
	int hnum = ftnum * 20; /* number of hybrid */
	/* iteration stop crit : best no change in 40 iterations */
	int it = 0, t;
	for (t = 0 ; it < 40; ++it, ++t)
	{
		int min_id = 0;
		double min_err = gene[0].error;
		for (i = 1; i < pnum; i++)
			if (gene[i].error < min_err)
			{
				min_id = i;
				min_err = gene[i].error;
			}
		min_err = gene[min_id].error = _ccv_bbf_error_rate(&gene[min_id].feature, posdata, posnum, negdata, negnum, size, pw, nw);
		if (min_err < best_err)
		{
			best_err = min_err;
			memcpy(&best, &gene[min_id].feature, sizeof(best));
			printf("best bbf feature with error %f\n|-size: %d\n|-positive point: ", best_err, best.size);
			for (i = 0; i < best.size; i++)
				printf("(%d %d %d), ", best.px[i], best.py[i], best.pz[i]);
			printf("\n|-negative point: ");
			for (i = 0; i < best.size; i++)
				printf("(%d %d %d), ", best.nx[i], best.ny[i], best.nz[i]);
			printf("\n");
			it = 0;
		}
		printf("minimum error achieved in round %d(%d) : %f with %d ms\n", t, it, min_err, timer / 1000);
		_ccv_bbf_genetic_qsort(gene, pnum, 0);
		for (i = 0; i < ftnum; i++)
			++gene[i].age;
		for (i = ftnum; i < ftnum + mnum; i++)
		{
			int parent = gsl_rng_uniform_int(rng, ftnum);
			memcpy(gene + i, gene + parent, sizeof(ccv_bbf_gene_t));
			/* three mutation strategy : 1. add, 2. remove, 3. refine */
			int pnm, pn = gsl_rng_uniform_int(rng, 2);
			int* pnk[] = { &gene[i].pk, &gene[i].nk };
			int* pnx[] = { gene[i].feature.px, gene[i].feature.nx };
			int* pny[] = { gene[i].feature.py, gene[i].feature.ny };
			int* pnz[] = { gene[i].feature.pz, gene[i].feature.nz };
			int x, y, z;
			int victim, decay = 1;
			do {
				switch (gsl_rng_uniform_int(rng, 3))
				{
					case 0: /* add */
						if (gene[i].pk == CCV_SGF_POINT_MAX && gene[i].nk == CCV_SGF_POINT_MAX)
							break;
						while (*pnk[pn] + 1 > CCV_SGF_POINT_MAX)
							pn = gsl_rng_uniform_int(rng, 2);
						do {
							z = gsl_rng_uniform_int(rng, 3);
							x = gsl_rng_uniform_int(rng, cols[z]);
							y = gsl_rng_uniform_int(rng, rows[z]);
						} while (_ccv_bbf_exist_gene_feature(&gene[i], x, y, z));
						pnz[pn][*pnk[pn]] = z;
						pnx[pn][*pnk[pn]] = x;
						pny[pn][*pnk[pn]] = y;
						++(*pnk[pn]);
						gene[i].feature.size = ccv_max(gene[i].pk, gene[i].nk);
						decay = gene[i].age = 0;
						break;
					case 1: /* remove */
						if (gene[i].pk + gene[i].nk <= CCV_SGF_POINT_MIN) /* at least 3 points have to be examed */
							break;
						while (*pnk[pn] - 1 <= 0) // || *pnk[pn] + *pnk[!pn] - 1 < CCV_SGF_POINT_MIN)
							pn = gsl_rng_uniform_int(rng, 2);
						victim = gsl_rng_uniform_int(rng, *pnk[pn]);
						for (j = victim; j < *pnk[pn] - 1; j++)
						{
							pnz[pn][j] = pnz[pn][j + 1];
							pnx[pn][j] = pnx[pn][j + 1];
							pny[pn][j] = pny[pn][j + 1];
						}
						pnz[pn][*pnk[pn] - 1] = -1;
						--(*pnk[pn]);
						gene[i].feature.size = ccv_max(gene[i].pk, gene[i].nk);
						decay = gene[i].age = 0;
						break;
					case 2: /* refine */
						pnm = gsl_rng_uniform_int(rng, *pnk[pn]);
						do {
							z = gsl_rng_uniform_int(rng, 3);
							x = gsl_rng_uniform_int(rng, cols[z]);
							y = gsl_rng_uniform_int(rng, rows[z]);
						} while (_ccv_bbf_exist_gene_feature(&gene[i], x, y, z));
						pnz[pn][pnm] = z;
						pnx[pn][pnm] = x;
						pny[pn][pnm] = y;
						decay = gene[i].age = 0;
						break;
				}
			} while (decay);
		}
		for (i = ftnum + mnum; i < ftnum + mnum + hnum; i++)
		{
			/* hybrid strategy: taking positive points from dad, negative points from mum */
			int dad, mum;
			do {
				dad = gsl_rng_uniform_int(rng, ftnum);
				mum = gsl_rng_uniform_int(rng, ftnum);
			} while (dad == mum || gene[dad].pk + gene[mum].nk < CCV_SGF_POINT_MIN); /* at least 3 points have to be examed */
			for (j = 0; j < CCV_SGF_POINT_MAX; j++)
			{
				gene[i].feature.pz[j] = -1;
				gene[i].feature.nz[j] = -1;
			}
			gene[i].pk = gene[dad].pk;
			for (j = 0; j < gene[i].pk; j++)
			{
				gene[i].feature.pz[j] = gene[dad].feature.pz[j];
				gene[i].feature.px[j] = gene[dad].feature.px[j];
				gene[i].feature.py[j] = gene[dad].feature.py[j];
			}
			gene[i].nk = gene[mum].nk;
			for (j = 0; j < gene[i].nk; j++)
			{
				gene[i].feature.nz[j] = gene[mum].feature.nz[j];
				gene[i].feature.nx[j] = gene[mum].feature.nx[j];
				gene[i].feature.ny[j] = gene[mum].feature.ny[j];
			}
			gene[i].feature.size = ccv_max(gene[i].pk, gene[i].nk);
			gene[i].age = 0;
		}
		for (i = ftnum + mnum + hnum; i < ftnum + mnum + hnum + rnum; i++)
			_ccv_bbf_randomize_gene(rng, &gene[i], rows, cols);
		timer = _ccv_bbf_time_measure();
#ifdef USE_OPENMP
#pragma omp parallel for private(i) schedule(dynamic)
#endif
		for (i = 0; i < pnum; i++)
			gene[i].error = _ccv_bbf_error_rate(&gene[i].feature, posdata, posnum, negdata, negnum, size, pw, nw);
		timer = _ccv_bbf_time_measure() - timer;
		for (i = 0; i < pnum; i++)
			_ccv_bbf_genetic_fitness(&gene[i]);
	}
	ccfree(gene);
	gsl_rng_free(rng);
#endif
	return best;
}

#define less_than(fit1, fit2, aux) ((fit1).error < (fit2).error)
static CCV_IMPLEMENT_QSORT(_ccv_bbf_best_qsort, ccv_bbf_gene_t, less_than)
#undef less_than

static ccv_bbf_gene_t _ccv_bbf_best_gene(ccv_bbf_gene_t* gene, int pnum, int point_min, unsigned char** posdata, int posnum, unsigned char** negdata, int negnum, ccv_size_t size, double* pw, double* nw)
{
	int i;
	unsigned int timer = _ccv_bbf_time_measure();
#ifdef USE_OPENMP
#pragma omp parallel for private(i) schedule(dynamic)
#endif
	for (i = 0; i < pnum; i++)
		gene[i].error = _ccv_bbf_error_rate(&gene[i].feature, posdata, posnum, negdata, negnum, size, pw, nw);
	timer = _ccv_bbf_time_measure() - timer;
	_ccv_bbf_best_qsort(gene, pnum, 0);
	int min_id = 0;
	double min_err = gene[0].error;
	for (i = 0; i < pnum; i++)
		if (gene[i].nk + gene[i].pk >= point_min)
		{
			min_id = i;
			min_err = gene[i].error;
			break;
		}
	printf("local best bbf feature with error %f\n|-size: %d\n|-positive point: ", min_err, gene[min_id].feature.size);
	for (i = 0; i < gene[min_id].feature.size; i++)
		printf("(%d %d %d), ", gene[min_id].feature.px[i], gene[min_id].feature.py[i], gene[min_id].feature.pz[i]);
	printf("\n|-negative point: ");
	for (i = 0; i < gene[min_id].feature.size; i++)
		printf("(%d %d %d), ", gene[min_id].feature.nx[i], gene[min_id].feature.ny[i], gene[min_id].feature.nz[i]);
	printf("\nthe computation takes %d ms\n", timer / 1000);
	return gene[min_id];
}

static ccv_bbf_feature_t _ccv_bbf_convex_optimize(unsigned char** posdata, int posnum, unsigned char** negdata, int negnum, ccv_bbf_feature_t* best_feature, ccv_size_t size, double* pw, double* nw)
{
	ccv_bbf_gene_t best_gene;
#ifdef HAVE_GSL
	/* seed (random method) */
	gsl_rng_env_setup();
	gsl_rng* rng = gsl_rng_alloc(gsl_rng_default);
	union { unsigned long int li; double db; } dbli;
	dbli.db = pw[0] + nw[0];
	gsl_rng_set(rng, dbli.li);
	int i, j, k, q, p, g, t;
	int rows[] = { size.height, size.height >> 1, size.height >> 2 };
	int cols[] = { size.width, size.width >> 1, size.width >> 2 };
	int pnum = rows[0] * cols[0] + rows[1] * cols[1] + rows[2] * cols[2];
	ccv_bbf_gene_t* gene = (ccv_bbf_gene_t*)ccmalloc((pnum * (CCV_BBF_POINT_MAX * 2 + 1) * 2 + CCV_BBF_POINT_MAX * 2 + 1) * sizeof(ccv_bbf_gene_t));
	if (best_feature == 0)
	{
		/* bootstrapping the best feature, start from two pixels, one for positive, one for negative
		 * the bootstrapping process go like this: first, it will assign a random pixel as positive
		 * and enumerate every possible pixel as negative, and pick the best one. Then, enumerate every
		 * possible pixel as positive, and pick the best one, until it converges */
		memset(&best_gene, 0, sizeof(ccv_bbf_gene_t));
		for (i = 0; i < CCV_BBF_POINT_MAX; i++)
			best_gene.feature.pz[i] = best_gene.feature.nz[i] = -1;
		best_gene.pk = 1;
		best_gene.nk = 0;
		best_gene.feature.size = 1;
		best_gene.feature.pz[0] = gsl_rng_uniform_int(rng, 3);
		best_gene.feature.px[0] = gsl_rng_uniform_int(rng, cols[best_gene.feature.pz[0]]);
		best_gene.feature.py[0] = gsl_rng_uniform_int(rng, rows[best_gene.feature.pz[0]]);
		for (t = 0; ; ++t)
		{
			g = 0;
			if (t % 2 == 0)
			{
				for (i = 0; i < 3; i++)
					for (j = 0; j < cols[i]; j++)
						for (k = 0; k < rows[i]; k++)
							if (i != best_gene.feature.pz[0] || j != best_gene.feature.px[0] || k != best_gene.feature.py[0])
							{
								gene[g] = best_gene;
								gene[g].pk = gene[g].nk = 1;
								gene[g].feature.nz[0] = i;
								gene[g].feature.nx[0] = j;
								gene[g].feature.ny[0] = k;
								g++;
							}
			} else {
				for (i = 0; i < 3; i++)
					for (j = 0; j < cols[i]; j++)
						for (k = 0; k < rows[i]; k++)
							if (i != best_gene.feature.nz[0] || j != best_gene.feature.nx[0] || k != best_gene.feature.ny[0])
							{
								gene[g] = best_gene;
								gene[g].pk = gene[g].nk = 1;
								gene[g].feature.pz[0] = i;
								gene[g].feature.px[0] = j;
								gene[g].feature.py[0] = k;
								g++;
							}
			}
			printf("bootstrapping round : %d\n", t);
			ccv_bbf_gene_t local_gene = _ccv_bbf_best_gene(gene, g, 2, posdata, posnum, negdata, negnum, size, pw, nw);
			if (local_gene.error >= best_gene.error - 1e-10)
				break;
			best_gene = local_gene;
		}
	} else {
		best_gene.feature = *best_feature;
		best_gene.pk = best_gene.nk = best_gene.feature.size;
		for (i = 0; i < CCV_BBF_POINT_MAX; i++)
			if (best_feature->pz[i] == -1)
			{
				best_gene.pk = i;
				break;
			}
		for (i = 0; i < CCV_BBF_POINT_MAX; i++)
			if (best_feature->nz[i] == -1)
			{
				best_gene.nk = i;
				break;
			}
	}
	/* after bootstrapping, the float search technique will do the following permutations:
	 * a). add a new point to positive or negative
	 * b). remove a point from positive or negative
	 * c). move an existing point in positive or negative to another position
	 * the three rules applied exhaustively, no heuristic used. */
	for (t = 0; ; ++t)
	{
		g = 0;
		for (i = 0; i < 3; i++)
			for (j = 0; j < cols[i]; j++)
				for (k = 0; k < rows[i]; k++)
					if (!_ccv_bbf_exist_gene_feature(&best_gene, j, k, i))
					{
						/* add positive point */
						if (best_gene.pk < CCV_BBF_POINT_MAX - 1)
						{
							gene[g] = best_gene;
							gene[g].feature.pz[gene[g].pk] = i;
							gene[g].feature.px[gene[g].pk] = j;
							gene[g].feature.py[gene[g].pk] = k;
							gene[g].pk++;
							gene[g].feature.size = ccv_max(gene[g].pk, gene[g].nk);
							g++;
						}
						/* add negative point */
						if (best_gene.nk < CCV_BBF_POINT_MAX - 1)
						{
							gene[g] = best_gene;
							gene[g].feature.nz[gene[g].nk] = i;
							gene[g].feature.nx[gene[g].nk] = j;
							gene[g].feature.ny[gene[g].nk] = k;
							gene[g].nk++;
							gene[g].feature.size = ccv_max(gene[g].pk, gene[g].nk);
							g++;
						}
						/* refine positive point */
						for (q = 0; q < best_gene.pk; q++)
						{
							gene[g] = best_gene;
							gene[g].feature.pz[q] = i;
							gene[g].feature.px[q] = j;
							gene[g].feature.py[q] = k;
							g++;
						}
						/* add positive point, remove negative point */
						if (best_gene.pk < CCV_BBF_POINT_MAX - 1 && best_gene.nk > 1)
						{
							for (q = 0; q < best_gene.nk; q++)
							{
								gene[g] = best_gene;
								gene[g].feature.pz[gene[g].pk] = i;
								gene[g].feature.px[gene[g].pk] = j;
								gene[g].feature.py[gene[g].pk] = k;
								gene[g].pk++;
								for (p = q; p < best_gene.nk - 1; p++)
								{
									gene[g].feature.nz[p] = gene[g].feature.nz[p + 1];
									gene[g].feature.nx[p] = gene[g].feature.nx[p + 1];
									gene[g].feature.ny[p] = gene[g].feature.ny[p + 1];
								}
								gene[g].feature.nz[gene[g].nk - 1] = -1;
								gene[g].nk--;
								gene[g].feature.size = ccv_max(gene[g].pk, gene[g].nk);
								g++;
							}
						}
						/* refine negative point */
						for (q = 0; q < best_gene.nk; q++)
						{
							gene[g] = best_gene;
							gene[g].feature.nz[q] = i;
							gene[g].feature.nx[q] = j;
							gene[g].feature.ny[q] = k;
							g++;
						}
						/* add negative point, remove positive point */
						if (best_gene.pk > 1 && best_gene.nk < CCV_BBF_POINT_MAX - 1)
						{
							for (q = 0; q < best_gene.pk; q++)
							{
								gene[g] = best_gene;
								gene[g].feature.nz[gene[g].nk] = i;
								gene[g].feature.nx[gene[g].nk] = j;
								gene[g].feature.ny[gene[g].nk] = k;
								gene[g].nk++;
								for (p = q; p < best_gene.pk - 1; p++)
								{
									gene[g].feature.pz[p] = gene[g].feature.pz[p + 1];
									gene[g].feature.px[p] = gene[g].feature.px[p + 1];
									gene[g].feature.py[p] = gene[g].feature.py[p + 1];
								}
								gene[g].feature.pz[gene[g].pk - 1] = -1;
								gene[g].pk--;
								gene[g].feature.size = ccv_max(gene[g].pk, gene[g].nk);
								g++;
							}
						}
					}
		if (best_gene.pk > 1)
			for (q = 0; q < best_gene.pk; q++)
			{
				gene[g] = best_gene;
				for (i = q; i < best_gene.pk - 1; i++)
				{
					gene[g].feature.pz[i] = gene[g].feature.pz[i + 1];
					gene[g].feature.px[i] = gene[g].feature.px[i + 1];
					gene[g].feature.py[i] = gene[g].feature.py[i + 1];
				}
				gene[g].feature.pz[gene[g].pk - 1] = -1;
				gene[g].pk--;
				gene[g].feature.size = ccv_max(gene[g].pk, gene[g].nk);
				g++;
			}
		if (best_gene.nk > 1)
			for (q = 0; q < best_gene.nk; q++)
			{
				gene[g] = best_gene;
				for (i = q; i < best_gene.nk - 1; i++)
				{
					gene[g].feature.nz[i] = gene[g].feature.nz[i + 1];
					gene[g].feature.nx[i] = gene[g].feature.nx[i + 1];
					gene[g].feature.ny[i] = gene[g].feature.ny[i + 1];
				}
				gene[g].feature.nz[gene[g].nk - 1] = -1;
				gene[g].nk--;
				gene[g].feature.size = ccv_max(gene[g].pk, gene[g].nk);
				g++;
			}
		gene[g] = best_gene;
		g++;
		printf("float search round : %d\n", t);
		ccv_bbf_gene_t local_gene = _ccv_bbf_best_gene(gene, g, CCV_BBF_POINT_MIN, posdata, posnum, negdata, negnum, size, pw, nw);
		if (local_gene.error >= best_gene.error - 1e-10)
			break;
		best_gene = local_gene;
	}
	ccfree(gene);
	gsl_rng_free(rng);
#endif
	return best_gene.feature;
}

static int _ccv_read_bbf_stage_classifier(const char* file, ccv_bbf_stage_classifier_t* classifier)
{
	FILE* r = fopen(file, "r");
	if (r == 0) return -1;
	int stat = 0;
	stat |= fscanf(r, "%d", &classifier->count);
	union { float fl; int i; } fli;
	stat |= fscanf(r, "%d", &fli.i);
	classifier->threshold = fli.fl;
	classifier->feature = (ccv_bbf_feature_t*)ccmalloc(classifier->count * sizeof(ccv_bbf_feature_t));
	classifier->alpha = (float*)ccmalloc(classifier->count * 2 * sizeof(float));
	int i, j;
	for (i = 0; i < classifier->count; i++)
	{
		stat |= fscanf(r, "%d", &classifier->feature[i].size);
		for (j = 0; j < classifier->feature[i].size; j++)
		{
			stat |= fscanf(r, "%d %d %d", &classifier->feature[i].px[j], &classifier->feature[i].py[j], &classifier->feature[i].pz[j]);
			stat |= fscanf(r, "%d %d %d", &classifier->feature[i].nx[j], &classifier->feature[i].ny[j], &classifier->feature[i].nz[j]);
		}
		union { float fl; int i; } flia, flib;
		stat |= fscanf(r, "%d %d", &flia.i, &flib.i);
		classifier->alpha[i * 2] = flia.fl;
		classifier->alpha[i * 2 + 1] = flib.fl;
	}
	fclose(r);
	return 0;
}

static int _ccv_write_bbf_stage_classifier(const char* file, ccv_bbf_stage_classifier_t* classifier)
{
	FILE* w = fopen(file, "wb");
	if (w == 0) return -1;
	fprintf(w, "%d\n", classifier->count);
	union { float fl; int i; } fli;
	fli.fl = classifier->threshold;
	fprintf(w, "%d\n", fli.i);
	int i, j;
	for (i = 0; i < classifier->count; i++)
	{
		fprintf(w, "%d\n", classifier->feature[i].size);
		for (j = 0; j < classifier->feature[i].size; j++)
		{
			fprintf(w, "%d %d %d\n", classifier->feature[i].px[j], classifier->feature[i].py[j], classifier->feature[i].pz[j]);
			fprintf(w, "%d %d %d\n", classifier->feature[i].nx[j], classifier->feature[i].ny[j], classifier->feature[i].nz[j]);
		}
		union { float fl; int i; } flia, flib;
		flia.fl = classifier->alpha[i * 2];
		flib.fl = classifier->alpha[i * 2 + 1];
		fprintf(w, "%d %d\n", flia.i, flib.i);
	}
	fclose(w);
	return 0;
}

static int _ccv_read_background_data(const char* file, unsigned char** negdata, int* negnum, ccv_size_t size)
{
	int stat = 0;
	FILE* r = fopen(file, "rb");
	if (r == 0) return -1;
	stat |= fread(negnum, sizeof(int), 1, r);
	int i;
	int isizs012 = _ccv_width_padding(size.width) * size.height +
				   _ccv_width_padding(size.width >> 1) * (size.height >> 1) +
				   _ccv_width_padding(size.width >> 2) * (size.height >> 2);
	for (i = 0; i < *negnum; i++)
	{
		negdata[i] = (unsigned char*)ccmalloc(isizs012);
		stat |= fread(negdata[i], 1, isizs012, r);
	}
	fclose(r);
	return 0;
}

static int _ccv_write_background_data(const char* file, unsigned char** negdata, int negnum, ccv_size_t size)
{
	FILE* w = fopen(file, "w");
	if (w == 0) return -1;
	fwrite(&negnum, sizeof(int), 1, w);
	int i;
	int isizs012 = _ccv_width_padding(size.width) * size.height +
				   _ccv_width_padding(size.width >> 1) * (size.height >> 1) +
				   _ccv_width_padding(size.width >> 2) * (size.height >> 2);
	for (i = 0; i < negnum; i++)
		fwrite(negdata[i], 1, isizs012, w);
	fclose(w);
	return 0;
}

static int _ccv_resume_bbf_cascade_training_state(const char* file, int* i, int* k, int* bg, double* pw, double* nw, int posnum, int negnum)
{
	int stat = 0;
	FILE* r = fopen(file, "r");
	if (r == 0) return -1;
	stat |= fscanf(r, "%d %d %d", i, k, bg);
	int j;
	union { double db; int i[2]; } dbi;
	for (j = 0; j < posnum; j++)
	{
		stat |= fscanf(r, "%d %d", &dbi.i[0], &dbi.i[1]);
		pw[j] = dbi.db;
	}
	for (j = 0; j < negnum; j++)
	{
		stat |= fscanf(r, "%d %d", &dbi.i[0], &dbi.i[1]);
		nw[j] = dbi.db;
	}
	fclose(r);
	return 0;
}

static int _ccv_save_bbf_cacade_training_state(const char* file, int i, int k, int bg, double* pw, double* nw, int posnum, int negnum)
{
	FILE* w = fopen(file, "w");
	if (w == 0) return -1;
	fprintf(w, "%d %d %d\n", i, k, bg);
	int j;
	union { double db; int i[2]; } dbi;
	for (j = 0; j < posnum; ++j)
	{
		dbi.db = pw[j];
		fprintf(w, "%d %d ", dbi.i[0], dbi.i[1]);
	}
	fprintf(w, "\n");
	for (j = 0; j < negnum; ++j)
	{
		dbi.db = nw[j];
		fprintf(w, "%d %d ", dbi.i[0], dbi.i[1]);
	}
	fprintf(w, "\n");
	fclose(w);
	return 0;
}

void ccv_bbf_classifier_cascade_new(ccv_dense_matrix_t** posimg, int posnum, char** bgfiles, int bgnum, int negnum, ccv_size_t size, const char* dir, ccv_bbf_new_param_t params)
{
	int i, j, k;
	/* allocate memory for usage */
	ccv_bbf_classifier_cascade_t* cascade = (ccv_bbf_classifier_cascade_t*)ccmalloc(sizeof(ccv_bbf_classifier_cascade_t));
	cascade->count = 0;
	cascade->size = size;
	cascade->stage_classifier = (ccv_bbf_stage_classifier_t*)ccmalloc(sizeof(ccv_bbf_stage_classifier_t));
	unsigned char** posdata = (unsigned char**)ccmalloc(posnum * sizeof(unsigned char*));
	unsigned char** negdata = (unsigned char**)ccmalloc(negnum * sizeof(unsigned char*));
	double* pw = (double*)ccmalloc(posnum * sizeof(double));
	double* nw = (double*)ccmalloc(negnum * sizeof(double));
	float* peval = (float*)ccmalloc(posnum * sizeof(float));
	float* neval = (float*)ccmalloc(negnum * sizeof(float));
	double inv_balance_k = 1. / params.balance_k;
	/* balance factor k, and weighted with 0.01 */
	params.balance_k *= 0.01;
	inv_balance_k *= 0.01;

	int steps[] = { _ccv_width_padding(cascade->size.width),
					_ccv_width_padding(cascade->size.width >> 1),
					_ccv_width_padding(cascade->size.width >> 2) };
	int isizs0 = steps[0] * cascade->size.height;
	int isizs01 = isizs0 + steps[1] * (cascade->size.height >> 1);
	
	i = 0;
	k = 0;
	int bg = 0;
	int cacheK = 10;
	/* state resume code */
	char buf[1024];
	sprintf(buf, "%s/stat.txt", dir);
	_ccv_resume_bbf_cascade_training_state(buf, &i, &k, &bg, pw, nw, posnum, negnum);
	if (i > 0)
	{
		cascade->count = i;
		ccfree(cascade->stage_classifier);
		cascade->stage_classifier = (ccv_bbf_stage_classifier_t*)ccmalloc(i * sizeof(ccv_bbf_stage_classifier_t));
		for (j = 0; j < i; j++)
		{
			sprintf(buf, "%s/stage-%d.txt", dir, j);
			_ccv_read_bbf_stage_classifier(buf, &cascade->stage_classifier[j]);
		}
	}
	if (k > 0)
		cacheK = k;
	int rpos, rneg = 0;
	if (bg)
	{
		sprintf(buf, "%s/negs.txt", dir);
		_ccv_read_background_data(buf, negdata, &rneg, cascade->size);
	}

	for (; i < params.layer; i++)
	{
		if (!bg)
		{
			rneg = _ccv_prepare_background_data(cascade, bgfiles, bgnum, negdata, negnum);
			/* save state of background data */
			sprintf(buf, "%s/negs.txt", dir);
			_ccv_write_background_data(buf, negdata, rneg, cascade->size);
			bg = 1;
		}
		double totalw;
		/* save state of cascade : level, weight etc. */
		sprintf(buf, "%s/stat.txt", dir);
		_ccv_save_bbf_cacade_training_state(buf, i, k, bg, pw, nw, posnum, negnum);
		ccv_bbf_stage_classifier_t classifier;
		if (k > 0)
		{
			/* resume state of classifier */
			sprintf( buf, "%s/stage-%d.txt", dir, i );
			_ccv_read_bbf_stage_classifier(buf, &classifier);
		} else {
			/* initialize classifier */
			for (j = 0; j < posnum; j++)
				pw[j] = params.balance_k;
			for (j = 0; j < rneg; j++)
				nw[j] = inv_balance_k;
			classifier.count = k;
			classifier.threshold = 0;
			classifier.feature = (ccv_bbf_feature_t*)ccmalloc(cacheK * sizeof(ccv_bbf_feature_t));
			classifier.alpha = (float*)ccmalloc(cacheK * 2 * sizeof(float));
		}
		_ccv_prepare_positive_data(posimg, posdata, cascade->size, posnum);
		rpos = _ccv_prune_positive_data(cascade, posdata, posnum, cascade->size);
		printf("%d postivie data and %d negative data in training\n", rpos, rneg);
		/* reweight to 1.00 */
		totalw = 0;
		for (j = 0; j < rpos; j++)
			totalw += pw[j];
		for (j = 0; j < rneg; j++)
			totalw += nw[j];
		for (j = 0; j < rpos; j++)
			pw[j] = pw[j] / totalw;
		for (j = 0; j < rneg; j++)
			nw[j] = nw[j] / totalw;
		for (; ; k++)
		{
			/* get overall true-positive, false-positive rate and threshold */
			double tp = 0, fp = 0, etp = 0, efp = 0;
			_ccv_bbf_eval_data(&classifier, posdata, rpos, negdata, rneg, cascade->size, peval, neval);
			_ccv_sort_32f(peval, rpos, 0);
			classifier.threshold = peval[(int)((1. - params.pos_crit) * rpos)] - 1e-6;
			for (j = 0; j < rpos; j++)
			{
				if (peval[j] >= 0)
					++tp;
				if (peval[j] >= classifier.threshold)
					++etp;
			}
			tp /= rpos; etp /= rpos;
			for (j = 0; j < rneg; j++)
			{
				if (neval[j] >= 0)
					++fp;
				if (neval[j] >= classifier.threshold)
					++efp;
			}
			fp /= rneg; efp /= rneg;
			printf("stage classifier real TP rate : %f, FP rate : %f\n", tp, fp);
			printf("stage classifier TP rate : %f, FP rate : %f at threshold : %f\n", etp, efp, classifier.threshold);
			if (k > 0)
			{
				/* save classifier state */
				sprintf(buf, "%s/stage-%d.txt", dir, i);
				_ccv_write_bbf_stage_classifier(buf, &classifier);
				sprintf(buf, "%s/stat.txt", dir);
				_ccv_save_bbf_cacade_training_state(buf, i, k, bg, pw, nw, posnum, negnum);
			}
			if (etp > params.pos_crit && efp < params.neg_crit)
				break;
			/* TODO: more post-process is needed in here */

			/* select the best feature in current distribution through genetic algorithm optimization */
			ccv_bbf_feature_t best;
			if (params.optimizer == CCV_BBF_GENETIC_OPT)
			{
				best = _ccv_bbf_genetic_optimize(posdata, rpos, negdata, rneg, params.feature_number, cascade->size, pw, nw);
			} else if (params.optimizer == CCV_BBF_FLOAT_OPT) {
				best = _ccv_bbf_convex_optimize(posdata, rpos, negdata, rneg, 0, cascade->size, pw, nw);
			} else {
				best = _ccv_bbf_genetic_optimize(posdata, rpos, negdata, rneg, params.feature_number, cascade->size, pw, nw);
				best = _ccv_bbf_convex_optimize(posdata, rpos, negdata, rneg, &best, cascade->size, pw, nw);
			}
			double err = _ccv_bbf_error_rate(&best, posdata, rpos, negdata, rneg, cascade->size, pw, nw);
			double rw = (1 - err) / err;
			totalw = 0;
			/* reweight */
			for (j = 0; j < rpos; j++)
			{
				unsigned char* u8[] = { posdata[j], posdata[j] + isizs0, posdata[j] + isizs01 };
				if (!_ccv_run_bbf_feature(&best, steps, u8))
					pw[j] *= rw;
				pw[j] *= params.balance_k;
				totalw += pw[j];
			}
			for (j = 0; j < rneg; j++)
			{
				unsigned char* u8[] = { negdata[j], negdata[j] + isizs0, negdata[j] + isizs01 };
				if (_ccv_run_bbf_feature(&best, steps, u8))
					nw[j] *= rw;
				nw[j] *= inv_balance_k;
				totalw += nw[j];
			}
			for (j = 0; j < rpos; j++)
				pw[j] = pw[j] / totalw;
			for (j = 0; j < rneg; j++)
				nw[j] = nw[j] / totalw;
			double c = log(rw);
			printf("coefficient of feature %d: %f\n", k + 1, c);
			classifier.count = k + 1;
			/* resizing classifier */
			if (k >= cacheK)
			{
				ccv_bbf_feature_t* feature = (ccv_bbf_feature_t*)ccmalloc(cacheK * 2 * sizeof(ccv_bbf_feature_t));
				memcpy(feature, classifier.feature, cacheK * sizeof(ccv_bbf_feature_t));
				ccfree(classifier.feature);
				float* alpha = (float*)ccmalloc(cacheK * 4 * sizeof(float));
				memcpy(alpha, classifier.alpha, cacheK * 2 * sizeof(float));
				ccfree(classifier.alpha);
				classifier.feature = feature;
				classifier.alpha = alpha;
				cacheK *= 2;
			}
			/* setup new feature */
			classifier.feature[k] = best;
			classifier.alpha[k * 2] = -c;
			classifier.alpha[k * 2 + 1] = c;
		}
		cascade->count = i + 1;
		ccv_bbf_stage_classifier_t* stage_classifier = (ccv_bbf_stage_classifier_t*)ccmalloc(cascade->count * sizeof(ccv_bbf_stage_classifier_t));
		memcpy(stage_classifier, cascade->stage_classifier, i * sizeof(ccv_bbf_stage_classifier_t));
		ccfree(cascade->stage_classifier);
		stage_classifier[i] = classifier;
		cascade->stage_classifier = stage_classifier;
		k = 0;
		bg = 0;
		for (j = 0; j < rpos; j++)
			ccfree(posdata[j]);
		for (j = 0; j < rneg; j++)
			ccfree(negdata[j]);
	}

	ccfree(neval);
	ccfree(peval);
	ccfree(nw);
	ccfree(pw);
	ccfree(negdata);
	ccfree(posdata);
	ccfree(cascade);
}

static int _ccv_is_equal(const void* _r1, const void* _r2, void* data)
{
	const ccv_comp_t* r1 = (const ccv_comp_t*)_r1;
	const ccv_comp_t* r2 = (const ccv_comp_t*)_r2;
	int distance = (int)(r1->rect.width * 0.25 + 0.5);

	return r2->rect.x <= r1->rect.x + distance &&
		   r2->rect.x >= r1->rect.x - distance &&
		   r2->rect.y <= r1->rect.y + distance &&
		   r2->rect.y >= r1->rect.y - distance &&
		   r2->rect.width <= (int)(r1->rect.width * 1.5 + 0.5) &&
		   (int)(r2->rect.width * 1.5 + 0.5) >= r1->rect.width;
}

static int _ccv_is_equal_same_class(const void* _r1, const void* _r2, void* data)
{
	const ccv_comp_t* r1 = (const ccv_comp_t*)_r1;
	const ccv_comp_t* r2 = (const ccv_comp_t*)_r2;
	int distance = (int)(r1->rect.width * 0.25 + 0.5);

	return r2->id == r1->id &&
		   r2->rect.x <= r1->rect.x + distance &&
		   r2->rect.x >= r1->rect.x - distance &&
		   r2->rect.y <= r1->rect.y + distance &&
		   r2->rect.y >= r1->rect.y - distance &&
		   r2->rect.width <= (int)(r1->rect.width * 1.5 + 0.5) &&
		   (int)(r2->rect.width * 1.5 + 0.5) >= r1->rect.width;
}

ccv_array_t* ccv_bbf_detect_objects(ccv_dense_matrix_t* a, ccv_bbf_classifier_cascade_t** _cascade, int count, ccv_bbf_param_t params)
{
	int hr = a->rows / params.size.height;
	int wr = a->cols / params.size.width;
	double scale = pow(2., 1. / (params.interval + 1.));
	int next = params.interval + 1;
	int scale_upto = (int)(log((double)ccv_min(hr, wr)) / log(scale));
	ccv_dense_matrix_t** pyr = (ccv_dense_matrix_t**)alloca((scale_upto + next * 2) * 4 * sizeof(ccv_dense_matrix_t*));
	memset(pyr, 0, (scale_upto + next * 2) * 4 * sizeof(ccv_dense_matrix_t*));
	if (params.size.height != _cascade[0]->size.height || params.size.width != _cascade[0]->size.width)
		ccv_resample(a, &pyr[0], 0, a->rows * _cascade[0]->size.height / params.size.height, a->cols * _cascade[0]->size.width / params.size.width, CCV_INTER_AREA);
	else
		pyr[0] = a;
	int i, j, k, t, x, y, q;
	for (i = 1; i <= params.interval; i++)
		ccv_resample(pyr[0], &pyr[i * 4], 0, (int)(pyr[0]->rows / pow(scale, i)), (int)(pyr[0]->cols / pow(scale, i)), CCV_INTER_AREA);
	for (i = next; i < scale_upto + next * 2; i++)
		ccv_sample_down(pyr[i * 4 - next * 4], &pyr[i * 4], 0, 0, 0);
	for (i = next * 2; i < scale_upto + next * 2; i++)
	{
		ccv_sample_down(pyr[i * 4 - next * 4], &pyr[i * 4 + 1], 0, 1, 0);
		ccv_sample_down(pyr[i * 4 - next * 4], &pyr[i * 4 + 2], 0, 0, 1);
		ccv_sample_down(pyr[i * 4 - next * 4], &pyr[i * 4 + 3], 0, 1, 1);
	}
	ccv_array_t* idx_seq;
	ccv_array_t* seq = ccv_array_new(64, sizeof(ccv_comp_t));
	ccv_array_t* seq2 = ccv_array_new(64, sizeof(ccv_comp_t));
	ccv_array_t* result_seq = ccv_array_new(64, sizeof(ccv_comp_t));
	/* detect in multi scale */
	for (t = 0; t < count; t++)
	{
		ccv_bbf_classifier_cascade_t* cascade = _cascade[t];
		float scale_x = (float) params.size.width / (float) cascade->size.width;
		float scale_y = (float) params.size.height / (float) cascade->size.height;
		ccv_array_clear(seq);
		for (i = 0; i < scale_upto; i++)
		{
			int dx[] = {0, 1, 0, 1};
			int dy[] = {0, 0, 1, 1};
			int i_rows = pyr[i * 4 + next * 8]->rows - (cascade->size.height >> 2);
			int steps[] = { pyr[i * 4]->step, pyr[i * 4 + next * 4]->step, pyr[i * 4 + next * 8]->step };
			int i_cols = pyr[i * 4 + next * 8]->cols - (cascade->size.width >> 2);
			int paddings[] = { pyr[i * 4]->step * 4 - i_cols * 4,
							   pyr[i * 4 + next * 4]->step * 2 - i_cols * 2,
							   pyr[i * 4 + next * 8]->step - i_cols };
			for (q = 0; q < 4; q++)
			{
				unsigned char* u8[] = { pyr[i * 4]->data.ptr + dx[q] * 2 + dy[q] * pyr[i * 4]->step * 2, pyr[i * 4 + next * 4]->data.ptr + dx[q] + dy[q] * pyr[i * 4 + next * 4]->step, pyr[i * 4 + next * 8 + q]->data.ptr };
				for (y = 0; y < i_rows; y++)
				{
					for (x = 0; x < i_cols; x++)
					{
						float sum;
						int flag = 1;
						ccv_bbf_stage_classifier_t* classifier = cascade->stage_classifier;
						for (j = 0; j < cascade->count; ++j, ++classifier)
						{
							sum = 0;
							float* alpha = classifier->alpha;
							ccv_bbf_feature_t* feature = classifier->feature;
							for (k = 0; k < classifier->count; ++k, alpha += 2, ++feature)
								sum += alpha[_ccv_run_bbf_feature(feature, steps, u8)];
							if (sum < classifier->threshold)
							{
								flag = 0;
								break;
							}
						}
						if (flag)
						{
							ccv_comp_t comp;
							comp.rect = ccv_rect((int)((x * 4 + dx[q] * 2) * scale_x + 0.5), (int)((y * 4 + dy[q] * 2) * scale_y + 0.5), (int)(cascade->size.width * scale_x + 0.5), (int)(cascade->size.height * scale_y + 0.5));
							comp.id = t;
							comp.neighbors = 1;
							comp.confidence = sum;
							ccv_array_push(seq, &comp);
						}
						u8[0] += 4;
						u8[1] += 2;
						u8[2] += 1;
					}
					u8[0] += paddings[0];
					u8[1] += paddings[1];
					u8[2] += paddings[2];
				}
			}
			scale_x *= scale;
			scale_y *= scale;
		}

		/* the following code from OpenCV's haar feature implementation */
		if(params.min_neighbors == 0)
		{
			for (i = 0; i < seq->rnum; i++)
			{
				ccv_comp_t* comp = (ccv_comp_t*)ccv_array_get(seq, i);
				ccv_array_push(result_seq, comp);
			}
		} else {
			idx_seq = 0;
			ccv_array_clear(seq2);
			// group retrieved rectangles in order to filter out noise
			int ncomp = ccv_array_group(seq, &idx_seq, _ccv_is_equal_same_class, 0);
			ccv_comp_t* comps = (ccv_comp_t*)ccmalloc((ncomp + 1) * sizeof(ccv_comp_t));
			memset(comps, 0, (ncomp + 1) * sizeof(ccv_comp_t));

			// count number of neighbors
			for(i = 0; i < seq->rnum; i++)
			{
				ccv_comp_t r1 = *(ccv_comp_t*)ccv_array_get(seq, i);
				int idx = *(int*)ccv_array_get(idx_seq, i);

				if (comps[idx].neighbors == 0)
					comps[idx].confidence = r1.confidence;

				++comps[idx].neighbors;

				comps[idx].rect.x += r1.rect.x;
				comps[idx].rect.y += r1.rect.y;
				comps[idx].rect.width += r1.rect.width;
				comps[idx].rect.height += r1.rect.height;
				comps[idx].id = r1.id;
				comps[idx].confidence = ccv_max(comps[idx].confidence, r1.confidence);
			}

			// calculate average bounding box
			for(i = 0; i < ncomp; i++)
			{
				int n = comps[i].neighbors;
				if(n >= params.min_neighbors)
				{
					ccv_comp_t comp;
					comp.rect.x = (comps[i].rect.x * 2 + n) / (2 * n);
					comp.rect.y = (comps[i].rect.y * 2 + n) / (2 * n);
					comp.rect.width = (comps[i].rect.width * 2 + n) / (2 * n);
					comp.rect.height = (comps[i].rect.height * 2 + n) / (2 * n);
					comp.neighbors = comps[i].neighbors;
					comp.id = comps[i].id;
					comp.confidence = comps[i].confidence;
					ccv_array_push(seq2, &comp);
				}
			}

			// filter out small face rectangles inside large face rectangles
			for(i = 0; i < seq2->rnum; i++)
			{
				ccv_comp_t r1 = *(ccv_comp_t*)ccv_array_get(seq2, i);
				int flag = 1;

				for(j = 0; j < seq2->rnum; j++)
				{
					ccv_comp_t r2 = *(ccv_comp_t*)ccv_array_get(seq2, j);
					int distance = (int)(r2.rect.width * 0.25 + 0.5);

					if(i != j &&
					   r1.id == r2.id &&
					   r1.rect.x >= r2.rect.x - distance &&
					   r1.rect.y >= r2.rect.y - distance &&
					   r1.rect.x + r1.rect.width <= r2.rect.x + r2.rect.width + distance &&
					   r1.rect.y + r1.rect.height <= r2.rect.y + r2.rect.height + distance &&
					   (r2.neighbors > ccv_max(3, r1.neighbors) || r1.neighbors < 3))
					{
						flag = 0;
						break;
					}
				}

				if(flag)
					ccv_array_push(result_seq, &r1);
			}
			ccv_array_free(idx_seq);
			ccfree(comps);
		}
	}

	ccv_array_free(seq);
	ccv_array_free(seq2);

	ccv_array_t* result_seq2;
	/* the following code from OpenCV's haar feature implementation */
	if (params.flags & CCV_SGF_NO_NESTED)
	{
		result_seq2 = ccv_array_new(64, sizeof(ccv_comp_t));
		idx_seq = 0;
		// group retrieved rectangles in order to filter out noise
		int ncomp = ccv_array_group(result_seq, &idx_seq, _ccv_is_equal, 0);
		ccv_comp_t* comps = (ccv_comp_t*)ccmalloc((ncomp + 1) * sizeof(ccv_comp_t));
		memset(comps, 0, (ncomp + 1) * sizeof(ccv_comp_t));

		// count number of neighbors
		for(i = 0; i < result_seq->rnum; i++)
		{
			ccv_comp_t r1 = *(ccv_comp_t*)ccv_array_get(result_seq, i);
			int idx = *(int*)ccv_array_get(idx_seq, i);

			if (comps[idx].neighbors == 0 || comps[idx].confidence < r1.confidence)
			{
				comps[idx].confidence = r1.confidence;
				comps[idx].neighbors = 1;
				comps[idx].rect = r1.rect;
				comps[idx].id = r1.id;
			}
		}

		// calculate average bounding box
		for(i = 0; i < ncomp; i++)
			if(comps[i].neighbors)
				ccv_array_push(result_seq2, &comps[i]);

		ccv_array_free(result_seq);
		ccfree(comps);
	} else {
		result_seq2 = result_seq;
	}

	for (i = 1; i < scale_upto + next * 2; i++)
		ccv_matrix_free(pyr[i * 4]);
	for (i = next * 2; i < scale_upto + next * 2; i++)
	{
		ccv_matrix_free(pyr[i * 4 + 1]);
		ccv_matrix_free(pyr[i * 4 + 2]);
		ccv_matrix_free(pyr[i * 4 + 3]);
	}
	if (params.size.height != _cascade[0]->size.height || params.size.width != _cascade[0]->size.width)
		ccv_matrix_free(pyr[0]);

	return result_seq2;
}

ccv_bbf_classifier_cascade_t* ccv_load_bbf_classifier_cascade(const char* directory)
{
	char buf[1024];
	sprintf(buf, "%s/cascade.txt", directory);
	int s, i;
	FILE* r = fopen(buf, "r");
	if (r == 0)
		return 0;
	ccv_bbf_classifier_cascade_t* cascade = (ccv_bbf_classifier_cascade_t*)ccmalloc(sizeof(ccv_bbf_classifier_cascade_t));
	s = fscanf(r, "%d %d %d", &cascade->count, &cascade->size.width, &cascade->size.height);
	cascade->stage_classifier = (ccv_bbf_stage_classifier_t*)ccmalloc(cascade->count * sizeof(ccv_bbf_stage_classifier_t));
	for (i = 0; i < cascade->count; i++)
	{
		sprintf(buf, "%s/stage-%d.txt", directory, i);
		if (_ccv_read_bbf_stage_classifier(buf, &cascade->stage_classifier[i]) < 0)
		{
			cascade->count = i;
			break;
		}
	}
	fclose(r);
	return cascade;
}

ccv_bbf_classifier_cascade_t* ccv_bbf_classifier_cascade_read_binary(char* s)
{
	int i;
	ccv_bbf_classifier_cascade_t* cascade = (ccv_bbf_classifier_cascade_t*)ccmalloc(sizeof(ccv_bbf_classifier_cascade_t));
	memcpy(&cascade->count, s, sizeof(cascade->count)); s += sizeof(cascade->count);
	memcpy(&cascade->size.width, s, sizeof(cascade->size.width)); s += sizeof(cascade->size.width);
	memcpy(&cascade->size.height, s, sizeof(cascade->size.height)); s += sizeof(cascade->size.height);
	ccv_bbf_stage_classifier_t* classifier = cascade->stage_classifier = (ccv_bbf_stage_classifier_t*)ccmalloc(cascade->count * sizeof(ccv_bbf_stage_classifier_t));
	for (i = 0; i < cascade->count; i++, classifier++)
	{
		memcpy(&classifier->count, s, sizeof(classifier->count)); s += sizeof(classifier->count);
		memcpy(&classifier->threshold, s, sizeof(classifier->threshold)); s += sizeof(classifier->threshold);
		classifier->feature = (ccv_bbf_feature_t*)ccmalloc(classifier->count * sizeof(ccv_bbf_feature_t));
		classifier->alpha = (float*)ccmalloc(classifier->count * 2 * sizeof(float));
		memcpy(classifier->feature, s, classifier->count * sizeof(ccv_bbf_feature_t)); s += classifier->count * sizeof(ccv_bbf_feature_t);
		memcpy(classifier->alpha, s, classifier->count * 2 * sizeof(float)); s += classifier->count * 2 * sizeof(float);
	}
	return cascade;

}

int ccv_bbf_classifier_cascade_write_binary(ccv_bbf_classifier_cascade_t* cascade, char* s, int slen)
{
	int i;
	int len = sizeof(cascade->count) + sizeof(cascade->size.width) + sizeof(cascade->size.height);
	ccv_bbf_stage_classifier_t* classifier = cascade->stage_classifier;
	for (i = 0; i < cascade->count; i++, classifier++)
		len += sizeof(classifier->count) + sizeof(classifier->threshold) + classifier->count * sizeof(ccv_bbf_feature_t) + classifier->count * 2 * sizeof(float);
	if (slen >= len)
	{
		memcpy(s, &cascade->count, sizeof(cascade->count)); s += sizeof(cascade->count);
		memcpy(s, &cascade->size.width, sizeof(cascade->size.width)); s += sizeof(cascade->size.width);
		memcpy(s, &cascade->size.height, sizeof(cascade->size.height)); s += sizeof(cascade->size.height);
		classifier = cascade->stage_classifier;
		for (i = 0; i < cascade->count; i++, classifier++)
		{
			memcpy(s, &classifier->count, sizeof(classifier->count)); s += sizeof(classifier->count);
			memcpy(s, &classifier->threshold, sizeof(classifier->threshold)); s += sizeof(classifier->threshold);
			memcpy(s, classifier->feature, classifier->count * sizeof(ccv_bbf_feature_t)); s += classifier->count * sizeof(ccv_bbf_feature_t);
			memcpy(s, classifier->alpha, classifier->count * 2 * sizeof(float)); s += classifier->count * 2 * sizeof(float);
		}
	}
	return len;
}

void ccv_bbf_classifier_cascade_free(ccv_bbf_classifier_cascade_t* cascade)
{
	int i;
	for (i = 0; i < cascade->count; ++i)
	{
		ccfree(cascade->stage_classifier[i].feature);
		ccfree(cascade->stage_classifier[i].alpha);
	}
	ccfree(cascade->stage_classifier);
	ccfree(cascade);
}

/* End of ccv/lib/ccv_bbf.c */
#line 1 "ccv/lib/ccv_cache.c"
/* ccv.h already included */


void ccv_cache_init(ccv_cache_t* cache, ccv_cache_index_free_f ffree, size_t up)
{
	cache->rnum = 0;
	cache->age = 0;
	cache->up = up;
	cache->size = 0;
	cache->ffree = ffree;
	memset(&cache->origin, 0, sizeof(ccv_cache_index_t));
}

static int bits_in_16bits[0x1u << 16];
static int bits_in_16bits_init = 0;

static int sparse_bitcount(unsigned int n) {
	int count = 0;
	while (n) {
		count++;
		n &= (n - 1);
	}
	return count;
}

static void precomputed_16bits() {
	int i;
	for (i = 0; i < (0x1u << 16); i++)
		bits_in_16bits[i] = sparse_bitcount(i);
	bits_in_16bits_init = 1;
}

static uint32_t compute_bits(uint64_t m) {
	return (bits_in_16bits[m & 0xffff] + bits_in_16bits[(m >> 16) & 0xffff] +
			bits_in_16bits[(m >> 32) & 0xffff] + bits_in_16bits[(m >> 48) & 0xffff]);
}

/* udate age along a path in the radix tree */
static void _ccv_cache_aging(ccv_cache_index_t* branch, uint64_t sign)
{
	if (!bits_in_16bits_init)
		precomputed_16bits();
	int i;
	uint64_t j = 63;
	ccv_cache_index_t* breadcrumb[10];
	for (i = 0; i < 10; i++)
	{
		breadcrumb[i] = branch;
		int leaf = branch->terminal.off & 0x1;
		int full = branch->terminal.off & 0x2;
		if (leaf)
		{
			break;
		} else {
			ccv_cache_index_t* set = (ccv_cache_index_t*)(branch->branch.set - (branch->branch.set & 0x3));
			int dice = (sign & j) >> (i * 6);
			if (full)
			{
				branch = set + dice;
			} else {
				uint64_t k = 1;
				k = k << dice;
				if (k & branch->branch.bitmap) {
					uint64_t m = (k - 1) & branch->branch.bitmap;
					branch = set + compute_bits(m);
				} else {
					break;
				}
			}
			j <<= 6;
		}
	}
	assert(i < 10);
	for (; i >= 0; i--)
	{
		branch = breadcrumb[i];
		int leaf = branch->terminal.off & 0x1;
		if (!leaf)
		{
			ccv_cache_index_t* set = (ccv_cache_index_t*)(branch->branch.set - (branch->branch.set & 0x3));
			uint32_t total = compute_bits(branch->branch.bitmap);
			uint32_t min_age = (set[0].terminal.off & 0x1) ? (set[0].terminal.age_and_size >> 32) : set[0].branch.age;
			for (j = 1; j < total; j++)
			{
				uint32_t age = (set[j].terminal.off & 0x1) ? (set[j].terminal.age_and_size >> 32) : set[j].branch.age;
				if (age < min_age)
					min_age = age;
			}
			branch->branch.age = min_age;
		}
	}
}

static ccv_cache_index_t* _ccv_cache_seek(ccv_cache_index_t* branch, uint64_t sign, int* depth)
{
	if (!bits_in_16bits_init)
		precomputed_16bits();
	int i;
	uint64_t j = 63;
	for (i = 0; i < 10; i++)
	{
		int leaf = branch->terminal.off & 0x1;
		int full = branch->terminal.off & 0x2;
		if (leaf)
		{
			if (depth)
				*depth = i;
			return branch;
		} else {
			ccv_cache_index_t* set = (ccv_cache_index_t*)(branch->branch.set - (branch->branch.set & 0x3));
			int dice = (sign & j) >> (i * 6);
			if (full)
			{
				branch = set + dice;
			} else {
				uint64_t k = 1;
				k = k << dice;
				if (k & branch->branch.bitmap) {
					uint64_t m = (k - 1) & branch->branch.bitmap;
					branch = set + compute_bits(m);
				} else {
					if (depth)
						*depth = i;
					return branch;
				}
			}
			j <<= 6;
		}
	}
	return 0;
}

void* ccv_cache_get(ccv_cache_t* cache, uint64_t sign)
{
	ccv_cache_index_t* branch = _ccv_cache_seek(&cache->origin, sign, 0);
	if (!branch)
		return 0;
	int leaf = branch->terminal.off & 0x1;
	if (!leaf)
		return 0;
	if (branch->terminal.sign != sign)
		return 0;
	return (void*)(branch->terminal.off - (branch->terminal.off & 0x3));
}

// only call this function when the cache space is delpeted
static void _ccv_cache_lru(ccv_cache_t* cache)
{
	ccv_cache_index_t* branch = &cache->origin;
	int leaf = branch->terminal.off & 0x1;
	if (leaf)
	{
		cache->rnum = 0;
		cache->size = 0;
		return;
	}
	uint32_t min_age = branch->branch.age;
	int i, j;
	for (i = 0; i < 10; i++)
	{
		ccv_cache_index_t* old_branch = branch;
		int leaf = branch->terminal.off & 0x1;
		if (leaf)
		{
			ccv_cache_delete(cache, branch->terminal.sign);
			break;
		} else {
			ccv_cache_index_t* set = (ccv_cache_index_t*)(branch->branch.set - (branch->branch.set & 0x3));
			uint32_t total = compute_bits(branch->branch.bitmap);
			for (j = 0; j < total; j++)
			{
				uint32_t age = (set[j].terminal.off & 0x1) ? (set[j].terminal.age_and_size >> 32) : set[j].branch.age;
				assert(age >= min_age);
				if (age == min_age)
				{
					branch = set + j;
					break;
				}
			}
			assert(old_branch != branch);
		}
	}
	assert(i < 10);
}

static void _ccv_cache_depleted(ccv_cache_t* cache, size_t size)
{
	while (cache->size > size)
		_ccv_cache_lru(cache);
}

int ccv_cache_put(ccv_cache_t* cache, uint64_t sign, void* x, uint32_t size)
{
	assert(((uint64_t)x & 0x3) == 0);
	if (size > cache->up)
		return -1;
	if (size + cache->size > cache->up)
		_ccv_cache_depleted(cache, cache->up - size);
	if (cache->rnum == 0)
	{
		cache->age = 1;
		cache->origin.terminal.off = (uint64_t)x | 0x1;
		cache->origin.terminal.sign = sign;
		cache->rnum = 1;
		return 0;
	}
	++cache->age;
	int i, depth = -1;
	ccv_cache_index_t* branch = _ccv_cache_seek(&cache->origin, sign, &depth);
	if (!branch)
		return -1;
	int leaf = branch->terminal.off & 0x1;
	uint64_t on = 1;
	assert(depth >= 0);
	if (leaf)
	{
		if (sign == branch->terminal.sign)
		{
			cache->ffree((void*)(branch->terminal.off - (branch->terminal.off & 0x3)));
			branch->terminal.off = (uint64_t)x | 0x1;
			uint32_t old_size = branch->terminal.age_and_size & 0xffffffff;
			cache->size = cache->size + size - old_size;
			branch->terminal.age_and_size = ((uint64_t)cache->age << 32) | size;
			_ccv_cache_aging(&cache->origin, sign);
			return 1;
		} else {
			ccv_cache_index_t t = *branch;
			uint32_t age = branch->terminal.age_and_size >> 32;
			uint64_t j = 63;
			j = j << (depth * 6);
			int dice, udice;
			assert(depth < 10);
			for (i = depth; i < 10; i++)
			{
				dice = (t.terminal.sign & j) >> (i * 6);
				udice = (sign & j) >> (i * 6);
				if (dice == udice)
				{
					branch->branch.bitmap = on << dice;
					ccv_cache_index_t* set = (ccv_cache_index_t*)ccmalloc(sizeof(ccv_cache_index_t));
					assert(((uint64_t)set & 0x3) == 0);
					branch->branch.set = (uint64_t)set;
					branch->branch.age = age;
					branch = set;
				} else {
					break;
				}
				j <<= 6;
			}
			branch->branch.bitmap = (on << dice) | (on << udice);
			ccv_cache_index_t* set = (ccv_cache_index_t*)ccmalloc(sizeof(ccv_cache_index_t) * 2);
			assert(((uint64_t)set & 0x3) == 0);
			branch->branch.set = (uint64_t)set;
			branch->branch.age = age;
			int u = dice < udice;
			set[u].terminal.sign = sign;
			set[u].terminal.off = (uint64_t)x | 0x1;
			set[u].terminal.age_and_size = ((uint64_t)cache->age << 32) | size;
			set[1 - u] = t;
		}
	} else {
		uint64_t k = 1, j = 63;
		k = k << ((sign & (j << (depth * 6))) >> (depth * 6));
		uint64_t m = (k - 1) & branch->branch.bitmap;
		uint32_t start = compute_bits(m);
		uint32_t total = compute_bits(branch->branch.bitmap);
		ccv_cache_index_t* set = (ccv_cache_index_t*)(branch->branch.set - (branch->branch.set & 0x3));
		set = (ccv_cache_index_t*)ccrealloc(set, sizeof(ccv_cache_index_t) * (total + 1));
		assert(((uint64_t)set & 0x3) == 0);
		for (i = total; i > start; i--)
			set[i] = set[i - 1];
		set[start].terminal.off = (uint64_t)x | 0x1;
		set[start].terminal.sign = sign;
		set[start].terminal.age_and_size = ((uint64_t)cache->age << 32) | size;
		branch->branch.set = (uint64_t)set;
		branch->branch.bitmap |= k;
		if (total == 63)
			branch->branch.set |= 0x2;
	}
	cache->rnum++;
	cache->size += size;
	return 0;
}

static void _ccv_cache_cleanup(ccv_cache_index_t* branch)
{
	int leaf = branch->terminal.off & 0x1;
	if (!leaf)
	{
		int i;
		uint64_t total = compute_bits(branch->branch.bitmap);
		ccv_cache_index_t* set = (ccv_cache_index_t*)(branch->branch.set - (branch->branch.set & 0x3));
		for (i = 0; i < total; i++)
		{
			if (!(set[i].terminal.off & 0x1))
				_ccv_cache_cleanup(set + i);
		}
		ccfree(set);
	}
}

static void _ccv_cache_cleanup_and_free(ccv_cache_index_t* branch, ccv_cache_index_free_f ffree)
{
	int leaf = branch->terminal.off & 0x1;
	if (!leaf)
	{
		int i;
		uint64_t total = compute_bits(branch->branch.bitmap);
		ccv_cache_index_t* set = (ccv_cache_index_t*)(branch->branch.set - (branch->branch.set & 0x3));
		for (i = 0; i < total; i++)
			_ccv_cache_cleanup_and_free(set + i, ffree);
		ccfree(set);
	} else {
		ffree((void*)(branch->terminal.off - (branch->terminal.off & 0x3)));
	}
}

void* ccv_cache_out(ccv_cache_t* cache, uint64_t sign)
{
	if (!bits_in_16bits_init)
		precomputed_16bits();
	int i, found = 0, depth = -1;
	ccv_cache_index_t* parent = 0;
	ccv_cache_index_t* uncle = &cache->origin;
	ccv_cache_index_t* branch = &cache->origin;
	uint64_t j = 63;
	for (i = 0; i < 10; i++)
	{
		int leaf = branch->terminal.off & 0x1;
		int full = branch->terminal.off & 0x2;
		if (leaf)
		{
			found = 1;
			break;
		}
		if (parent != 0 && compute_bits(parent->branch.bitmap) > 1)
			uncle = branch;
		parent = branch;
		depth = i;
		ccv_cache_index_t* set = (ccv_cache_index_t*)(branch->branch.set - (branch->branch.set & 0x3));
		int dice = (sign & j) >> (i * 6);
		if (full)
		{
			branch = set + dice;
		} else {
			uint64_t k = 1;
			k = k << dice;
			if (k & branch->branch.bitmap)
			{
				uint64_t m = (k - 1) & branch->branch.bitmap;
				branch = set + compute_bits(m);
			} else {
				return 0;
			}
		}
		j <<= 6;
	}
	if (!found)
		return 0;
	int leaf = branch->terminal.off & 0x1;
	if (!leaf)
		return 0;
	if (branch->terminal.sign != sign)
		return 0;
	void* result = (void*)(branch->terminal.off - (branch->terminal.off & 0x3));
	uint32_t size = branch->terminal.age_and_size & 0xffffffff;
	if (branch != &cache->origin)
	{
		uint64_t k = 1, j = 63;
		int dice = (sign & (j << (depth * 6))) >> (depth * 6);
		k = k << dice;
		uint64_t m = (k - 1) & parent->branch.bitmap;
		uint32_t start = compute_bits(m);
		uint32_t total = compute_bits(parent->branch.bitmap);
		assert(total > 1);
		ccv_cache_index_t* set = (ccv_cache_index_t*)(parent->branch.set - (parent->branch.set & 0x3));
		if (total > 2 || (total == 2 && !(set[1 - start].terminal.off & 0x1)))
		{
			parent->branch.bitmap &= ~k;
			for (i = start + 1; i < total; i++)
				set[i - 1] = set[i];
			set = (ccv_cache_index_t*)ccrealloc(set, sizeof(ccv_cache_index_t) * (total - 1));
			parent->branch.set = (uint64_t)set;
		} else {
			ccv_cache_index_t t = set[1 - start];
			_ccv_cache_cleanup(uncle);
			*uncle = t;
		}
		_ccv_cache_aging(&cache->origin, sign);
	} else {
		// if I only have one item, reset age to 1
		cache->age = 1;
	}
	cache->rnum--;
	cache->size -= size;
	return result;
}

int ccv_cache_delete(ccv_cache_t* cache, uint64_t sign)
{
	void* result = ccv_cache_out(cache, sign);
	if (result != 0)
	{
		cache->ffree(result);
		return 0;
	}
	return -1;
}

void ccv_cache_cleanup(ccv_cache_t* cache)
{
	if (cache->rnum > 0)
	{
		_ccv_cache_cleanup_and_free(&cache->origin, cache->ffree);
		cache->size = 0;
		cache->age = 0;
		cache->rnum = 0;
	}
}

void ccv_cache_close(ccv_cache_t* cache)
{
	// for radix-tree based cache, close/cleanup are the same (it is not the same for cuckoo based one,
	// because for cuckoo based one, it will free up space in close whereas only cleanup space in cleanup
	ccv_cache_cleanup(cache);
}

/* End of ccv/lib/ccv_cache.c */
#line 1 "ccv/lib/ccv_daisy.c"
/* ccv.h already included */


/* the method is adopted from original author's published C++ code under BSD Licence.
 * Here is the copyright:
 * //////////////////////////////////////////////////////////////////////////
 * // Software License Agreement (BSD License)                             //
 * //                                                                      //
 * // Copyright (c) 2009                                                   //
 * // Engin Tola                                                           //
 * // web   : http://cvlab.epfl.ch/~tola                                   //
 * // email : engin.tola@epfl.ch                                           //
 * //                                                                      //
 * // All rights reserved.                                                 //
 * //                                                                      //
 * // Redistribution and use in source and binary forms, with or without   //
 * // modification, are permitted provided that the following conditions   //
 * // are met:                                                             //
 * //                                                                      //
 * //  * Redistributions of source code must retain the above copyright    //
 * //    notice, this list of conditions and the following disclaimer.     //
 * //  * Redistributions in binary form must reproduce the above           //
 * //    copyright notice, this list of conditions and the following       //
 * //    disclaimer in the documentation and/or other materials provided   //
 * //    with the distribution.                                            //
 * //  * Neither the name of the EPFL nor the names of its                 //
 * //    contributors may be used to endorse or promote products derived   //
 * //    from this software without specific prior written permission.     //
 * //                                                                      //
 * // THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS  //
 * // "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT    //
 * // LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS    //
 * // FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE       //
 * // COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,  //
 * // INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, //
 * // BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;     //
 * // LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER     //
 * // CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT   //
 * // LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN    //
 * // ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE      //
 * // POSSIBILITY OF SUCH DAMAGE.                                          //
 * //                                                                      //
 * // See licence.txt file for more details                                //
 * //////////////////////////////////////////////////////////////////////////
 */

void ccv_daisy(ccv_dense_matrix_t* a, ccv_dense_matrix_t** b, int type, ccv_daisy_param_t params)
{
	int grid_point_number = params.rad_q_no * params.th_q_no + 1;
	int desc_size = grid_point_number * params.hist_th_q_no;
	char* identifier = (char*)alloca(ccv_max(sizeof(ccv_daisy_param_t) + 9, sizeof(double) * params.rad_q_no));
	memset(identifier, 0, ccv_max(sizeof(ccv_daisy_param_t) + 9, sizeof(double) * params.rad_q_no));
	memcpy(identifier, "ccv_daisy", 9);
	memcpy(identifier + 9, &params, sizeof(ccv_daisy_param_t));
	uint64_t sig = (a->sig == 0) ? 0 : ccv_matrix_generate_signature(identifier, sizeof(ccv_daisy_param_t) + 9, a->sig, 0);
	type = (type == 0) ? CCV_32F | CCV_C1 : CCV_GET_DATA_TYPE(type) | CCV_C1;
	ccv_dense_matrix_t* db = *b = ccv_dense_matrix_renew(*b, a->rows, a->cols * desc_size, CCV_C1 | CCV_ALL_DATA_TYPE, type, sig);
	int layer_size = a->rows * a->cols;
	int cube_size = layer_size * params.hist_th_q_no;
	float* workspace_memory = (float*)ccmalloc(cube_size * (params.rad_q_no + 2) * sizeof(float));
	/* compute_cube_sigmas */
	int i, j, k, r, t;
	double* cube_sigmas = (double*)identifier;
	double r_step = params.radius / (double)params.rad_q_no;
	for (i = 0; i < params.rad_q_no; i++)
		cube_sigmas[i] = (i + 1) * r_step * 0.5;
	/* compute_grid_points */
	double t_step = 2 * 3.141592654 / params.th_q_no;
	double* grid_points = (double*)alloca(grid_point_number * 2 * sizeof(double));
	for (i = 0; i < params.rad_q_no; i++)
		for (j = 0; j < params.th_q_no; j++)
		{
			grid_points[(i * params.th_q_no + 1 + j) * 2] = sin(j * t_step) * (i + 1) * r_step;
			grid_points[(i * params.th_q_no + 1 + j) * 2 + 1] = cos(j * t_step) * (i + 1) * r_step;
		}
	/* TODO: require 0.5 gaussian smooth before gradient computing */
	/* NOTE: the default sobel already applied a sigma = 0.85 gaussian blur by using a
	 * | -1  0  1 |   |  0  0  0 |   | 1  2  1 |
	 * | -2  0  2 | = | -1  0  1 | * | 2  4  2 |
	 * | -1  0  1 |   |  0  0  0 |   | 1  2  1 | */
	ccv_dense_matrix_t* dx = ccv_dense_matrix_new(a->rows, a->cols, CCV_32F | CCV_C1, 0, 0);
	ccv_sobel(a, &dx, 0, 1, 0);
	ccv_dense_matrix_t* dy = ccv_dense_matrix_new(a->rows, a->cols, CCV_32F | CCV_C1, 0, 0);
	ccv_sobel(a, &dy, 0, 0, 1);
	double sobel_sigma = sqrt(0.5 / -log(0.5));
	double sigma_init = 1.6;
	double sigma = sqrt(sigma_init * sigma_init - sobel_sigma * sobel_sigma);
	/* layered_gradient & smooth_layers */
	for (k = params.hist_th_q_no - 1; k >= 0; k--)
	{
		float radius = k * 2 * 3.141592654 / params.th_q_no;
		float kcos = cos(radius);
		float ksin = sin(radius);
		float* w_ptr = workspace_memory + cube_size + (k - 1) * layer_size;
		for (i = 0; i < layer_size; i++)
			w_ptr[i] = ccv_max(0, kcos * dx->data.fl[i] + ksin * dy->data.fl[i]);
		ccv_dense_matrix_t src = ccv_dense_matrix(a->rows, a->cols, CCV_32F | CCV_C1, w_ptr, 0);
		ccv_dense_matrix_t des = ccv_dense_matrix(a->rows, a->cols, CCV_32F | CCV_C1, w_ptr + layer_size, 0);
		ccv_dense_matrix_t* desp = &des;
		ccv_blur(&src, &desp, 0, sigma);
	}
	ccv_matrix_free(dx);
	ccv_matrix_free(dy);
	/* compute_smoothed_gradient_layers & compute_histograms (rearrange memory) */
	for (k = 0; k < params.rad_q_no; k++)
	{
		sigma = (k == 0) ? cube_sigmas[0] : sqrt(cube_sigmas[k] * cube_sigmas[k] - cube_sigmas[k - 1] * cube_sigmas[k - 1]);
		float* src_ptr = workspace_memory + (k + 1) * cube_size;
		float* des_ptr = src_ptr + cube_size;
		for (i = 0; i < params.hist_th_q_no; i++)
		{
			ccv_dense_matrix_t src = ccv_dense_matrix(a->rows, a->cols, CCV_32F | CCV_C1, src_ptr + i * layer_size, 0);
			ccv_dense_matrix_t des = ccv_dense_matrix(a->rows, a->cols, CCV_32F | CCV_C1, des_ptr + i * layer_size, 0);
			ccv_dense_matrix_t* desp = &des;
			ccv_blur(&src, &desp, 0, sigma);
		}
		float* his_ptr = src_ptr - cube_size;
		for (i = 0; i < layer_size; i++)
			for (j = 0; j < params.hist_th_q_no; j++)
				his_ptr[i * params.hist_th_q_no + j] = src_ptr[i + j * layer_size];
	}
	/* petals of the flower */
	memset(db->data.ptr, 0, db->rows * db->step);
	for (i = 0; i < a->rows; i++)
		for (j = 0; j < a->cols; j++)
		{
			float* a_ptr = workspace_memory + i * params.hist_th_q_no * a->cols + j * params.hist_th_q_no;
			float* b_ptr = db->data.fl + i * db->cols + j * desc_size;
			memcpy(b_ptr, a_ptr, params.hist_th_q_no * sizeof(float));
			for (r = 0; r < params.rad_q_no; r++)
			{
				int rdt = r * params.th_q_no + 1;
				for (t = rdt; t < rdt + params.th_q_no; t++)
				{
					double y = i + grid_points[t * 2];
					double x = j + grid_points[t * 2 + 1];
					int iy = (int)(y + 0.5);
					int ix = (int)(x + 0.5);
					float* bh = b_ptr + t * params.hist_th_q_no;
					if (iy < 0 || iy >= a->rows || ix < 0 || ix >= a->cols)
						continue;
					// bilinear interpolation
					int jy = (int)y;
					int jx = (int)x;
					float yr = y - jy, _yr = 1 - yr;
					float xr = x - jx, _xr = 1 - xr;
					if (jy >= 0 && jy < a->rows && jx >= 0 && jx < a->cols)
					{
						float* ah = workspace_memory + (r + 1) * cube_size + jy * params.hist_th_q_no * a->cols + jx * params.hist_th_q_no;
						for (k = 0; k < params.hist_th_q_no; k++)
							bh[k] += ah[k] * _yr * _xr;
					}
					if (jy + 1 >= 0 && jy + 1 < a->rows && jx >= 0 && jx < a->cols)
					{
						float* ah = workspace_memory + (r + 1) * cube_size + (jy + 1) * params.hist_th_q_no * a->cols + jx * params.hist_th_q_no;
						for (k = 0; k < params.hist_th_q_no; k++)
							bh[k] += ah[k] * yr * _xr;
					}
					if (jy >= 0 && jy < a->rows && jx + 1 >= 0 && jx + 1 < a->cols)
					{
						float* ah = workspace_memory + (r + 1) * cube_size + jy * params.hist_th_q_no * a->cols + (jx + 1) * params.hist_th_q_no;
						for (k = 0; k < params.hist_th_q_no; k++)
							bh[k] += ah[k] * _yr * xr;
					}
					if (jy + 1 >= 0 && jy + 1 < a->rows && jx + 1 >= 0 && jx + 1 < a->cols)
					{
						float* ah = workspace_memory + (r + 1) * cube_size + (jy + 1) * params.hist_th_q_no * a->cols + (jx + 1) * params.hist_th_q_no;
						for (k = 0; k < params.hist_th_q_no; k++)
							bh[k] += ah[k] * yr * xr;
					}
				}
			}
		}
	ccfree(workspace_memory);
	for (i = 0; i < a->rows; i++)
		for (j = 0; j < a->cols; j++)
		{
			float* b_ptr = db->data.fl + i * db->cols + j * desc_size;
			float norm;
			int iter, changed;
			switch (params.normalize_method)
			{
				case CCV_DAISY_NORMAL_PARTIAL:
					for (t = 0; t < grid_point_number; t++)
					{
						norm = 0;
						float* bh = b_ptr + t * params.hist_th_q_no;
						for (k = 0; k < params.hist_th_q_no; k++)
							norm += bh[k] * bh[k];
						if (norm > 1e-6)
						{
							norm = 1.0 / sqrt(norm);
							for (k = 0; k < params.hist_th_q_no; k++)
								bh[k] *= norm;
						}
					}
					break;
				case CCV_DAISY_NORMAL_FULL:
					norm = 0;
					for (t = 0; t < desc_size; t++)
						norm += b_ptr[t] * b_ptr[t];
					if (norm > 1e-6)
					{
						norm = 1.0 / sqrt(norm);
						for (t = 0; t < desc_size; t++)
							b_ptr[t] *= norm;
					}
					break;
				case CCV_DAISY_NORMAL_SIFT:
					for (iter = 0, changed = 1; changed && iter < 5; iter++)
					{
						norm = 0;
						for (t = 0; t < desc_size; t++)
							norm += b_ptr[t] * b_ptr[t];
						changed = 0;
						if (norm > 1e-6)
						{
							norm = 1.0 / sqrt(norm);
							for (t = 0; t < desc_size; t++)
							{
								b_ptr[t] *= norm;
								if (b_ptr[t] < params.normalize_threshold)
								{
									b_ptr[t] = params.normalize_threshold;
									changed = 1;
								}
							}
						}
					}
					break;
			}
		}
}

/* End of ccv/lib/ccv_daisy.c */
#line 1 "ccv/lib/ccv_dpm.c"
/* ccv.h already included */


void ccv_dpm_classifier_lsvm_new(ccv_dense_matrix_t** posimgs, int posnum, char** bgfiles, int bgnum, int negnum, const char* dir, ccv_dpm_new_param_t params)
{
}

static inline float _ccv_lsvm_match(ccv_dense_matrix_t* a, ccv_dense_matrix_t* tpl, ccv_dense_matrix_t** b)
{
}

// this is specific HOG computation for dpm, I may later change it to ccv_hog
static void _ccv_hog(ccv_dense_matrix_t* a, ccv_dense_matrix_t** b, int b_type, int sbin, int size)
{
	ccv_dense_matrix_t* ag = 0;
	ccv_dense_matrix_t* mg = 0;
	ccv_gradient(a, &ag, 0, &mg, 0, 1, 1);
	float* agp = ag->data.fl;
	float* mgp = mg->data.fl;
	int i, j, k;
	int rows = a->rows / size;
	int cols = a->cols / size;
	ccv_dense_matrix_t* cn = ccv_dense_matrix_new(rows, cols, CCV_32F | (sbin * 2), 0, 0);
	ccv_dense_matrix_t* ca = ccv_dense_matrix_new(rows, cols, CCV_64F | CCV_C1, 0, 0);
	ccv_dense_matrix_t* db = *b = ccv_dense_matrix_new(rows, cols, CCV_32F | (4 + sbin * 3), 0, 0);
	ccv_zero(cn);
	float* cnp = cn->data.fl;
	int sizec = 0;
	for (i = 0; i < rows * size; i++)
	{
		for (j = 0; j < cols; j++)
		{
			for (k = j * size; k < j * size + size; k++)
			{
				int ag0, ag1;
				float agr;
				agr = (agp[k] / 360.0f) * (sbin * 2);
				ag0 = (int)agr;
				ag1 = (ag0 + 1 < sbin * 2) ? ag0 + 1 : 0;
				agr = agr - ag0;
				cnp[ag0] += (1.0f - agr) * mgp[k] / 255.0f;
				cnp[ag1] += agr * mgp[k] / 255.0f;
			}
			cnp += 2 * sbin;
		}
		agp += a->cols;
		mgp += a->cols;
		if (++sizec < size)
			cnp -= cn->cols;
		else
			sizec = 0;
	}
	ccv_matrix_free(ag);
	ccv_matrix_free(mg);
	cnp = cn->data.fl;
	double* cap = ca->data.db;
	for (i = 0; i < rows; i++)
	{
		for (j = 0; j < cols; j++)
		{
			*cap = 0;
			for (k = 0; k < sbin * 2; k++)
			{
				*cap += (*cnp) * (*cnp);
				cnp++;
			}
			cap++;
		}
	}
	cnp = cn->data.fl;
	cap = ca->data.db;
	ccv_zero(db);
	float* dbp = db->data.fl;
	// normalize sbin direction-sensitive and sbin * 2 insensitive over 4 normalization factor
	// accumulating them over sbin * 2 + sbin + 4 channels
	// TNA - truncation - normalization - accumulation
#define TNA(idx, a, b, c, d) \
	{ \
		float norm = 1.0f / sqrt(cap[a] + cap[b] + cap[c] + cap[d] + 1e-4f); \
		for (k = 0; k < sbin * 2; k++) \
		{ \
			float v = 0.5f * ccv_min(cnp[k] * norm, 0.2f); \
			dbp[5 + sbin + k] += v; \
			dbp[1 + idx] += v; \
		} \
		dbp[sbin * 3 + idx] *= 0.2357f; \
		for (k = 0; k < sbin; k++) \
		{ \
			float v = 0.5f * ccv_min((cnp[k] + cnp[k + sbin]) * norm, 0.2f); \
			dbp[5 + k] += v; \
		} \
	}
	TNA(0, 0, 0, 0, 0);
	TNA(1, 1, 1, 0, 0);
	TNA(2, 0, ca->cols, ca->cols, 0);
	TNA(3, 1, ca->cols + 1, ca->cols, 0);
	cnp += 2 * sbin;
	dbp += 3 * sbin + 4;
	cap++;
	for (j = 1; j < cols - 1; j++)
	{
		TNA(0, -1, -1, 0, 0);
		TNA(1, 1, 1, 0, 0);
		TNA(2, -1, ca->cols - 1, ca->cols, 0);
		TNA(3, 1, ca->cols + 1, ca->cols, 0);
		cnp += 2 * sbin;
		dbp += 3 * sbin + 4;
		cap++;
	}
	TNA(0, -1, -1, 0, 0);
	TNA(1, 0, 0, 0, 0);
	TNA(2, -1, ca->cols - 1, ca->cols, 0);
	TNA(3, 0, ca->cols, ca->cols, 0);
	cnp += 2 * sbin;
	dbp += 3 * sbin + 4;
	cap++;
	for (i = 1; i < rows - 1; i++)
	{
		TNA(0, 0, -ca->cols, -ca->cols, 0);
		TNA(1, 1, -ca->cols + 1, -ca->cols, 0);
		TNA(2, 0, ca->cols, ca->cols, 0);
		TNA(3, 1, ca->cols + 1, ca->cols, 0);
		cnp += 2 * sbin;
		dbp += 3 * sbin + 4;
		cap++;
		for (j = 1; j < cols - 1; j++)
		{
			TNA(0, -1, -ca->cols - 1, -ca->cols, 0);
			TNA(1, 1, -ca->cols + 1, -ca->cols, 0);
			TNA(2, -1, ca->cols - 1, ca->cols, 0);
			TNA(3, 1, ca->cols + 1, ca->cols, 0);
			cnp += 2 * sbin;
			dbp += 3 * sbin + 4;
			cap++;
		}
		TNA(0, -1, -ca->cols - 1, -ca->cols, 0);
		TNA(1, 0, -ca->cols, -ca->cols, 0);
		TNA(2, -1, ca->cols - 1, ca->cols, 0);
		TNA(3, 0, ca->cols, ca->cols, 0);
		cnp += 2 * sbin;
		dbp += 3 * sbin + 4;
		cap++;
	}
	TNA(0, 0, -ca->cols, -ca->cols, 0);
	TNA(1, 1, -ca->cols + 1, -ca->cols, 0);
	TNA(2, 0, 0, 0, 0);
	TNA(3, 1, 1, 0, 0);
	cnp += 2 * sbin;
	dbp += 3 * sbin + 4;
	cap++;
	for (j = 1; j < cols - 1; j++)
	{
		TNA(0, -1, -ca->cols - 1, -ca->cols, 0);
		TNA(1, 1, -ca->cols + 1, -ca->cols, 0);
		TNA(2, -1, -1, 0, 0);
		TNA(3, 1, 1, 0, 0);
		cnp += 2 * sbin;
		dbp += 3 * sbin + 4;
		cap++;
	}
	TNA(0, -1, -ca->cols - 1, -ca->cols, 0);
	TNA(1, 0, -ca->cols, -ca->cols, 0);
	TNA(2, -1, -1, 0, 0);
	TNA(3, 0, 0, 0, 0);
#undef TNA
	ccv_matrix_free(cn);
	ccv_matrix_free(ca);
}

ccv_array_t* ccv_dpm_detect_objects(ccv_dense_matrix_t* a, ccv_dpm_root_classifier_t** _classifier, int count, ccv_dpm_param_t params)
{
	int hr = a->rows / params.size.height;
	int wr = a->cols / params.size.width;
	double scale = pow(2., 1. / (params.interval + 1.));
	int next = params.interval + 1;
	int scale_upto = (int)(log((double)ccv_min(hr, wr)) / log(scale));
	ccv_dense_matrix_t** pyr = (ccv_dense_matrix_t**)alloca((scale_upto + next * 2) * sizeof(ccv_dense_matrix_t*));
	memset(pyr, 0, (scale_upto + next * 2) * sizeof(ccv_dense_matrix_t*));
	if (params.size.height != _classifier[0]->root.size.height * 8 || params.size.width != _classifier[0]->root.size.width * 8)
		ccv_resample(a, &pyr[0], 0, a->rows * _classifier[0]->root.size.height * 8 / params.size.height, a->cols * _classifier[0]->root.size.width * 8 / params.size.width, CCV_INTER_AREA);
	else
		pyr[0] = a;
	int i, j;
	for (i = 1; i <= params.interval; i++)
		ccv_resample(pyr[0], &pyr[i], 0, (int)(pyr[0]->rows / pow(scale, i)), (int)(pyr[0]->cols / pow(scale, i)), CCV_INTER_AREA);
	for (i = next; i < scale_upto + next * 2; i++)
		ccv_sample_down(pyr[i - next], &pyr[i], 0, 0, 0);
	for (i = 0; i < 1; i++) // scale_upto + next * 2; i++)
	{
		ccv_dense_matrix_t* hog = 0;
		_ccv_hog(pyr[i], &hog, 0, 9, 8);
		// ??? Syntax error?
		// _ccv_run_lsvm(hog, );
		ccv_matrix_free(hog);
	}
	/*
	ccv_dense_matrix_t* hog = 0;
	_ccv_hog(pyr[0], &hog, 0, 9, 8);
	ccv_dense_matrix_t* b = ccv_dense_matrix_new(pyr[0]->rows, pyr[0]->cols, CCV_8U | CCV_C1, 0, 0);
	unsigned char* bptr = b->data.ptr;
	for (i = 0; i < b->rows; i++)
	{
		if (i >= hog->rows * 8)
			break;
		for (j = 0; j < b->cols; j++)
		{
			int k = (i / 8) * hog->cols + (j / 8) * 31 + 2;
			bptr[j] = ccv_clamp(100 * hog->data.fl[k], 0, 255);
		}
		bptr += b->step;
	}
	ccv_serialize(b, "hog.png", 0, CCV_SERIAL_PNG_FILE, 0);
	ccv_matrix_free(hog);
	ccv_matrix_free(b);
	*/
	if (params.size.height != _classifier[0]->root.size.height * 8 || params.size.width != _classifier[0]->root.size.width * 8)
		ccv_matrix_free(pyr[0]);
	 for (i = 1; i < scale_upto + next * 2; i++)
		ccv_matrix_free(pyr[i]);
	return 0;
}

ccv_dpm_root_classifier_t* ccv_load_dpm_root_classifier(const char* directory)
{
	FILE* r = fopen(directory, "r");
	if (r == 0)
		return 0;
	ccv_dpm_root_classifier_t* root_classifier = (ccv_dpm_root_classifier_t*)ccmalloc(sizeof(ccv_dpm_root_classifier_t));
	memset(root_classifier, 0, sizeof(ccv_dpm_root_classifier_t));
	fscanf(r, "%d %d", &root_classifier->root.size.width, &root_classifier->root.size.height);
	root_classifier->root.w = (float*)ccmalloc(sizeof(float) * root_classifier->root.size.width * root_classifier->root.size.height * 32);
	int i, j, k;
	for (i = 0; i < root_classifier->root.size.width * root_classifier->root.size.height * 32; i++)
		fscanf(r, "%f", &root_classifier->root.w[i]);
	/*
	for (j = 0; j < root_classifier->root.size.width * root_classifier->root.size.height; j++)
	{
		i = 31;
		printf("%f ", root_classifier->root.w[i * root_classifier->root.size.width * root_classifier->root.size.height + j]);
		for (i = 27; i < 31; i++)
			printf("%f ", root_classifier->root.w[i * root_classifier->root.size.width * root_classifier->root.size.height + j]);
		for (i = 18; i < 27; i++)
			printf("%f ", root_classifier->root.w[i * root_classifier->root.size.width * root_classifier->root.size.height + j]);
		for (i = 0; i < 18; i++)
			printf("%f ", root_classifier->root.w[i * root_classifier->root.size.width * root_classifier->root.size.height + j]);
	}
	printf("\n");
	*/
	fclose(r);
	return root_classifier;
}

/* End of ccv/lib/ccv_dpm.c */
#line 1 "ccv/lib/ccv_memory.c"
/* ccv.h already included */

/* 3rdparty/sha1.h already included */


static ccv_cache_t ccv_cache;

/* option to enable/disable cache */
static int ccv_cache_opt = 0;

ccv_dense_matrix_t* ccv_dense_matrix_new(int rows, int cols, int type, void* data, uint64_t sig)
{
	ccv_dense_matrix_t* mat;
	if (ccv_cache_opt && sig != 0)
	{
		mat = (ccv_dense_matrix_t*)ccv_cache_out(&ccv_cache, sig);
		if (mat)
		{
			mat->type |= CCV_GARBAGE;
			mat->refcount = 1;
			return mat;
		}
	}
	mat = (ccv_dense_matrix_t*)ccmalloc((data) ? sizeof(ccv_dense_matrix_t) : (sizeof(ccv_dense_matrix_t) + ((cols * CCV_GET_DATA_TYPE_SIZE(type) * CCV_GET_CHANNEL(type) + 3) & -4) * rows));
	mat->sig = sig;
	mat->type = (type | CCV_MATRIX_DENSE) & ~CCV_GARBAGE;
	if (data == 0)
		mat->type |= CCV_REUSABLE;
	mat->rows = rows;
	mat->cols = cols;
	mat->step = (cols * CCV_GET_DATA_TYPE_SIZE(type) * CCV_GET_CHANNEL(type) + 3) & -4;
	mat->refcount = 1;
	mat->data.ptr = (data) ? (unsigned char*)data : (unsigned char*)(mat + 1);
	return mat;
}

ccv_dense_matrix_t* ccv_dense_matrix_renew(ccv_dense_matrix_t* x, int rows, int cols, int types, int prefer_type, uint64_t sig)
{
	if (x != 0 && (x->rows != rows || x->cols != cols || !(CCV_GET_DATA_TYPE(x->type) & types) || (CCV_GET_CHANNEL(x->type) != CCV_GET_CHANNEL(types))))
	{
		ccv_matrix_free(x);
		x = 0;
	} else if (x != 0) {
		prefer_type = CCV_GET_DATA_TYPE(x->type) | CCV_GET_CHANNEL(x->type);
	}
	sig = ccv_matrix_generate_signature((const char*)&prefer_type, sizeof(int), sig, 0);
	if (x == 0)
	{
		x = ccv_dense_matrix_new(rows, cols, prefer_type, 0, sig);
	} else {
		x->sig = sig;
	}
	return x;
}

ccv_dense_matrix_t ccv_dense_matrix(int rows, int cols, int type, void* data, uint64_t sig)
{
	ccv_dense_matrix_t mat;
	mat.sig = sig;
	mat.type = (type | CCV_MATRIX_DENSE) & ~CCV_GARBAGE;
	mat.rows = rows;
	mat.cols = cols;
	mat.step = (cols * CCV_GET_DATA_TYPE_SIZE(type) * CCV_GET_CHANNEL(type) + 3) & -4;
	mat.refcount = 1;
	mat.data.ptr = (unsigned char*)data;
	return mat;
}

ccv_sparse_matrix_t* ccv_sparse_matrix_new(int rows, int cols, int type, int major, uint64_t sig)
{
	ccv_sparse_matrix_t* mat;
	mat = (ccv_sparse_matrix_t*)ccmalloc(sizeof(ccv_sparse_matrix_t));
	mat->rows = rows;
	mat->cols = cols;
	mat->type = type | CCV_MATRIX_SPARSE | ((type & CCV_DENSE_VECTOR) ? CCV_DENSE_VECTOR : CCV_SPARSE_VECTOR);
	mat->major = major;
	mat->prime = 0;
	mat->load_factor = 0;
	mat->refcount = 1;
	mat->vector = (ccv_dense_vector_t*)ccmalloc(CCV_GET_SPARSE_PRIME(mat->prime) * sizeof(ccv_dense_vector_t));
	int i;
	for (i = 0; i < CCV_GET_SPARSE_PRIME(mat->prime); i++)
	{
		mat->vector[i].index = -1;
		mat->vector[i].length = 0;
		mat->vector[i].next = 0;
	}
	return mat;
}

void ccv_matrix_free_immediately(ccv_matrix_t* mat)
{
	int type = *(int*)mat;
	if (type & CCV_MATRIX_DENSE)
	{
		ccv_dense_matrix_t* dmt = (ccv_dense_matrix_t*)mat;
		dmt->refcount = 0;
		ccfree(dmt);
	} else if (type & CCV_MATRIX_SPARSE) {
		ccv_sparse_matrix_t* smt = (ccv_sparse_matrix_t*)mat;
		int i;
		for (i = 0; i < CCV_GET_SPARSE_PRIME(smt->prime); i++)
			if (smt->vector[i].index != -1)
			{
				ccv_dense_vector_t* iter = &smt->vector[i];
				ccfree(iter->data.ptr);
				iter = iter->next;
				while (iter != 0)
				{
					ccv_dense_vector_t* iter_next = iter->next;
					ccfree(iter->data.ptr);
					ccfree(iter);
					iter = iter_next;
				}
			}
		ccfree(smt->vector);
		ccfree(smt);
	} else if ((type & CCV_MATRIX_CSR) || (type & CCV_MATRIX_CSC)) {
		ccv_compressed_sparse_matrix_t* csm = (ccv_compressed_sparse_matrix_t*)mat;
		csm->refcount = 0;
		ccfree(csm);
	}
}

void ccv_matrix_free(ccv_matrix_t* mat)
{
	int type = *(int*)mat;
	if (type & CCV_MATRIX_DENSE)
	{
		ccv_dense_matrix_t* dmt = (ccv_dense_matrix_t*)mat;
		dmt->refcount = 0;
		if (!ccv_cache_opt || !(dmt->type & CCV_REUSABLE) || dmt->sig == 0)
			ccfree(dmt);
		else {
			size_t size = sizeof(ccv_dense_matrix_t) + ((dmt->cols * CCV_GET_DATA_TYPE_SIZE(dmt->type) * CCV_GET_CHANNEL(dmt->type) + 3) & -4) * dmt->rows;
			ccv_cache_put(&ccv_cache, dmt->sig, dmt, size);
		}
	} else if (type & CCV_MATRIX_SPARSE) {
		ccv_sparse_matrix_t* smt = (ccv_sparse_matrix_t*)mat;
		int i;
		for (i = 0; i < CCV_GET_SPARSE_PRIME(smt->prime); i++)
			if (smt->vector[i].index != -1)
			{
				ccv_dense_vector_t* iter = &smt->vector[i];
				ccfree(iter->data.ptr);
				iter = iter->next;
				while (iter != 0)
				{
					ccv_dense_vector_t* iter_next = iter->next;
					ccfree(iter->data.ptr);
					ccfree(iter);
					iter = iter_next;
				}
			}
		ccfree(smt->vector);
		ccfree(smt);
	} else if ((type & CCV_MATRIX_CSR) || (type & CCV_MATRIX_CSC)) {
		ccv_compressed_sparse_matrix_t* csm = (ccv_compressed_sparse_matrix_t*)mat;
		csm->refcount = 0;
		ccfree(csm);
	}
}

void ccv_drain_cache(void)
{
	if (ccv_cache.rnum > 0)
		ccv_cache_cleanup(&ccv_cache);
}

void ccv_disable_cache(void)
{
	ccv_cache_opt = 0;
	ccv_cache_close(&ccv_cache);
}

void ccv_enable_cache(size_t size)
{
	ccv_cache_opt = 1;
	ccv_cache_init(&ccv_cache, ccfree, size);
}

void ccv_enable_default_cache(void)
{
	ccv_enable_cache(CCV_DEFAULT_CACHE_SIZE);
}

uint64_t ccv_matrix_generate_signature(const char* msg, int len, uint64_t sig_start, ...)
{
	blk_SHA_CTX ctx;
	blk_SHA1_Init(&ctx);
	uint64_t sigi;
	va_list arguments;
	va_start(arguments, sig_start);
	for (sigi = sig_start; sigi != 0; sigi = va_arg(arguments, uint64_t))
		blk_SHA1_Update(&ctx, &sigi, 8);
	va_end(arguments);
	blk_SHA1_Update(&ctx, msg, len);
	union {
		uint64_t u;
		uint8_t chr[20];
	} sig;
	blk_SHA1_Final(sig.chr, &ctx);
	return sig.u;
}

/* End of ccv/lib/ccv_memory.c */
#line 1 "ccv/lib/ccv_numeric.c"
/* ccv.h already included */

#ifdef HAVE_FFTW3
#include <complex.h>
#include <fftw3.h>
#endif

void ccv_invert(ccv_matrix_t* a, ccv_matrix_t** b, int type)
{
}

void ccv_solve(ccv_matrix_t* a, ccv_matrix_t* b, ccv_matrix_t** d, int type)
{
}

void ccv_eigen(ccv_matrix_t* a, ccv_matrix_t* b, ccv_matrix_t** d, int type)
{
}

void ccv_minimize(ccv_dense_matrix_t* x, int length, double red, ccv_minimize_f func, ccv_minimize_param_t params, void* data)
{
	ccv_dense_matrix_t* df0 = ccv_dense_matrix_new(x->rows, x->cols, x->type, 0, 0);
	ccv_zero(df0);
	ccv_dense_matrix_t* df3 = ccv_dense_matrix_new(x->rows, x->cols, x->type, 0, 0);
	ccv_zero(df3);
	ccv_dense_matrix_t* dF0 = ccv_dense_matrix_new(x->rows, x->cols, x->type, 0, 0);
	ccv_zero(dF0);
	ccv_dense_matrix_t* s = ccv_dense_matrix_new(x->rows, x->cols, x->type, 0, 0);
	ccv_zero(s);
	ccv_dense_matrix_t* x0 = ccv_dense_matrix_new(x->rows, x->cols, x->type, 0, 0);
	ccv_zero(x0);
	ccv_dense_matrix_t* xn = ccv_dense_matrix_new(x->rows, x->cols, x->type, 0, 0);
	ccv_zero(xn);
	
	double F0 = 0, f0 = 0, f1 = 0, f2 = 0, f3 = 0, f4 = 0;
	double x1 = 0, x2 = 0, x3 = 0, x4 = 0;
	double d0 = 0, d1 = 0, d2 = 0, d3 = 0, d4 = 0;
	double A = 0, B = 0;
	int ls_failed = 0;

	int i, j, k;
	func(x, &f0, df0, data);
	d0 = 0;
	unsigned char* df0p = df0->data.ptr;
	unsigned char* sp = s->data.ptr;
	for (i = 0; i < x->rows; i++)
	{
		for (j = 0; j < x->cols; j++)
		{
			double ss = ccv_get_value(x->type, df0p, j);
			ccv_set_value(x->type, sp, j, -ss, 0);
			d0 += -ss * ss;
		}
		df0p += x->step;
		sp += x->step;
	}
	x3 = red / (1.0 - d0);
	int l = (length > 0) ? length : -length;
	int ls = (length > 0) ? 1 : 0;
	int eh = (length > 0) ? 0 : 1;
	for (k = 0; k < l;)
	{
		k += ls;
		memcpy(x0->data.ptr, x->data.ptr, x->rows * x->step);
		memcpy(dF0->data.ptr, df0->data.ptr, x->rows * x->step);
		F0 = f0;
		int m = ccv_min(params.max_iter, (length > 0) ? params.max_iter : l - k);
		for (;;)
		{
			x2 = 0;
			f3 = f2 = f0;
			d2 = d0;
			memcpy(df3->data.ptr, df0->data.ptr, x->rows * x->step);
			while (m > 0)
			{
				m--;
				k += eh;
				unsigned char* sp = s->data.ptr;
				unsigned char* xp = x->data.ptr;
				unsigned char* xnp = xn->data.ptr;
				for (i = 0; i < x->rows; i++)
				{
					for (j = 0; j < x->cols; j++)
						ccv_set_value(x->type, xnp, j, x3 * ccv_get_value(x->type, sp, j) + ccv_get_value(x->type, xp, j), 0);
					sp += x->step;
					xp += x->step;
					xnp += x->step;
				}
				if (func(xn, &f3, df3, data) == 0)
					break;
				else
					x3 = (x2 + x3) * 0.5;
			}
			if (f3 < F0)
			{
				memcpy(x0->data.ptr, xn->data.ptr, x->rows * x->step);
				memcpy(dF0->data.ptr, df3->data.ptr, x->rows * x->step);
				F0 = f3;
			}
			d3 = 0;
			unsigned char* df3p = df3->data.ptr;
			unsigned char* sp = s->data.ptr;
			for (i = 0; i < x->rows; i++)
			{
				for (j = 0; j < x->cols; j++)
					d3 += ccv_get_value(x->type, df3p, j) * ccv_get_value(x->type, sp, j);
				df3p += x->step;
				sp += x->step;
			}
			if ((d3 > params.sig * d0) || (f3 > f0 + x3 * params.rho * d0) || (m <= 0))
				break;
			x1 = x2; f1 = f2; d1 = d2;
			x2 = x3; f2 = f3; d2 = d3;
			A = 6.0 * (f1 - f2) + 3.0 * (d2 + d1) * (x2 - x1);
			B = 3.0 * (f2 - f1) - (2.0 * d1 + d2) * (x2 - x1);
			x3 = B * B - A * d1 * (x2 - x1);
			if (x3 < 0)
				x3 = x2 * params.extrap;
			else {
				x3 = x1 - d1 * (x2 - x1) * (x2 - x1) / (B + sqrt(x3));
				if (x3 < 0)
					x3 = x2 * params.extrap;
				else {
					if (x3 > x2 * params.extrap)
						x3 = x2 * params.extrap;
					else if (x3 < x2 + params.interp * (x2 - x1))
						x3 = x2 + params.interp * (x2 - x1);
				}
			}
		}
		while (((fabs(d3) > -params.sig * d0) || (f3 > f0 + x3 * params.rho * d0)) && (m > 0))
		{
			if ((d3 > 1e-8) || (f3 > f0 + x3 * params.rho * d0))
			{
				x4 = x3; f4 = f3; d4 = d3;
			} else {
				x2 = x3; f2 = f3; d2 = d3;
			}
			if (f4 > f0)
				x3 = x2 - (0.5 * d2 * (x4 - x2) * (x4 - x2)) / (f4 - f2 - d2 * (x4 - x2));
			else {
				A = 6.0 * (f2 - f4) / (x4 - x2) + 3.0 * (d4 + d2);
				B = 3.0 * (f4 - f2) - (2.0 * d2 + d4) * (x4 - x2);
				x3 = B * B - A * d2 * (x4 - x2) * (x4 - x2);
				x3 = (x3 < 0) ? (x2 + x4) * 0.5 : x2 + (sqrt(x3) - B) / A;
			}
			x3 = ccv_max(ccv_min(x3, x4 - params.interp * (x4 - x2)), x2 + params.interp * (x4 - x2));
			sp = s->data.ptr;
			unsigned char* xp = x->data.ptr;
			unsigned char* xnp = xn->data.ptr;
			for (i = 0; i < x->rows; i++)
			{
				for (j = 0; j < x->cols; j++)
					ccv_set_value(x->type, xnp, j, x3 * ccv_get_value(x->type, sp, j) + ccv_get_value(x->type, xp, j), 0);
				sp += x->step;
				xp += x->step;
				xnp += x->step;
			}
			func(xn, &f3, df3, data);
			if (f3 < F0)
			{
				memcpy(x0->data.ptr, xn->data.ptr, x->rows * x->step);
				memcpy(dF0->data.ptr, df3->data.ptr, x->rows * x->step);
				F0 = f3;
			}
			m--;
			k += eh;
			d3 = 0;
			sp = s->data.ptr;
			unsigned char* df3p = df3->data.ptr;
			for (i = 0; i < x->rows; i++)
			{
				for (j = 0; j < x->cols; j++)
					d3 += ccv_get_value(x->type, df3p, j) * ccv_get_value(x->type, sp, j);
				df3p += x->step;
				sp += x->step;
			}
		}
		if ((fabs(d3) < -params.sig * d0) && (f3 < f0 + x3 * params.rho * d0))
		{
			memcpy(x->data.ptr, xn->data.ptr, x->rows * x->step);
			f0 = f3;
			double df0_df3 = 0;
			double df3_df3 = 0;
			double df0_df0 = 0;
			unsigned char* df0p = df0->data.ptr;
			unsigned char* df3p = df3->data.ptr;
			for (i = 0; i < x->rows; i++)
			{
				for (j = 0; j < x->cols; j++)
				{
					df0_df0 += ccv_get_value(x->type, df0p, j) * ccv_get_value(x->type, df0p, j);
					df0_df3 += ccv_get_value(x->type, df0p, j) * ccv_get_value(x->type, df3p, j);
					df3_df3 += ccv_get_value(x->type, df3p, j) * ccv_get_value(x->type, df3p, j);
				}
				df0p += x->step;
				df3p += x->step;
			}
			double slr = (df3_df3 - df0_df3) / df0_df0;
			df3p = df3->data.ptr;
			unsigned char* sp = s->data.ptr;
			for (i = 0; i < x->rows; i++)
			{
				for (j = 0; j < x->cols; j++)
					ccv_set_value(x->type, sp, j, slr * ccv_get_value(x->type, sp, j) - ccv_get_value(x->type, df3p, j), 0);
				df3p += x->step;
				sp += x->step;
			}
			memcpy(df0->data.ptr, df3->data.ptr, x->rows * x->step);
			d3 = d0;
			d0 = 0;
			df0p = df0->data.ptr;
			sp = s->data.ptr;
			for (i = 0; i < x->rows; i++)
			{
				for (j = 0; j < x->cols; j++)
				{
					d0 += ccv_get_value(x->type, df0p, j) * ccv_get_value(x->type, sp, j);
				}
				df0p += x->step;
				sp += x->step;
			}
			if (d0 > 0)
			{
				d0 = 0;
				df0p = df0->data.ptr;
				sp = s->data.ptr;
				for (i = 0; i < x->rows; i++)
				{
					for (j = 0; j < x->cols; j++)
					{
						double ss = ccv_get_value(x->type, df0p, j);
						ccv_set_value(x->type, sp, j, -ss, 0);
						d0 += -ss * ss;
					}
					df0p += x->step;
					sp += x->step;
				}
			}
			x3 = x3 * ccv_min(params.ratio, d3 / (d0 - 1e-8));
			ls_failed = 0;
		} else {
			memcpy(x->data.ptr, x0->data.ptr, x->rows * x->step);
			memcpy(df0->data.ptr, dF0->data.ptr, x->rows * x->step);
			f0 = F0;
			if (ls_failed)
				break;
			d0 = 0;
			unsigned char* df0p = df0->data.ptr;
			unsigned char* sp = s->data.ptr;
			for (i = 0; i < x->rows; i++)
			{
				for (j = 0; j < x->cols; j++)
				{
					double ss = ccv_get_value(x->type, df0p, j);
					ccv_set_value(x->type, sp, j, -ss, 0);
					d0 += -ss * ss;
				}
				df0p += x->step;
				sp += x->step;
			}
			x3 = red / (1.0 - d0);
			ls_failed = 1;
		}
	}
	ccv_matrix_free(s);
	ccv_matrix_free(x0);
	ccv_matrix_free(xn);
	ccv_matrix_free(dF0);
	ccv_matrix_free(df0);
	ccv_matrix_free(df3);
}

/* optimal FFT size table is adopted from OpenCV */
static const int _ccv_optimal_fft_size[] = {
	1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24, 25, 27, 30, 32, 36, 40, 45, 48, 
	50, 54, 60, 64, 72, 75, 80, 81, 90, 96, 100, 108, 120, 125, 128, 135, 144, 150, 160, 
	162, 180, 192, 200, 216, 225, 240, 243, 250, 256, 270, 288, 300, 320, 324, 360, 375, 
	384, 400, 405, 432, 450, 480, 486, 500, 512, 540, 576, 600, 625, 640, 648, 675, 720, 
	729, 750, 768, 800, 810, 864, 900, 960, 972, 1000, 1024, 1080, 1125, 1152, 1200, 
	1215, 1250, 1280, 1296, 1350, 1440, 1458, 1500, 1536, 1600, 1620, 1728, 1800, 1875, 
	1920, 1944, 2000, 2025, 2048, 2160, 2187, 2250, 2304, 2400, 2430, 2500, 2560, 2592, 
	2700, 2880, 2916, 3000, 3072, 3125, 3200, 3240, 3375, 3456, 3600, 3645, 3750, 3840, 
	3888, 4000, 4050, 4096, 4320, 4374, 4500, 4608, 4800, 4860, 5000, 5120, 5184, 5400, 
	5625, 5760, 5832, 6000, 6075, 6144, 6250, 6400, 6480, 6561, 6750, 6912, 7200, 7290, 
	7500, 7680, 7776, 8000, 8100, 8192, 8640, 8748, 9000, 9216, 9375, 9600, 9720, 10000, 
	10125, 10240, 10368, 10800, 10935, 11250, 11520, 11664, 12000, 12150, 12288, 12500, 
	12800, 12960, 13122, 13500, 13824, 14400, 14580, 15000, 15360, 15552, 15625, 16000, 
	16200, 16384, 16875, 17280, 17496, 18000, 18225, 18432, 18750, 19200, 19440, 19683, 
	20000, 20250, 20480, 20736, 21600, 21870, 22500, 23040, 23328, 24000, 24300, 24576, 
	25000, 25600, 25920, 26244, 27000, 27648, 28125, 28800, 29160, 30000, 30375, 30720, 
	31104, 31250, 32000, 32400, 32768, 32805, 33750, 34560, 34992, 36000, 36450, 36864, 
	37500, 38400, 38880, 39366, 40000, 40500, 40960, 41472, 43200, 43740, 45000, 46080, 
	46656, 46875, 48000, 48600, 49152, 50000, 50625, 51200, 51840, 52488, 54000, 54675, 
	55296, 56250, 57600, 58320, 59049, 60000, 60750, 61440, 62208, 62500, 64000, 64800, 
	65536, 65610, 67500, 69120, 69984, 72000, 72900, 73728, 75000, 76800, 77760, 78125, 
	78732, 80000, 81000, 81920, 82944, 84375, 86400, 87480, 90000, 91125, 92160, 93312, 
	93750, 96000, 97200, 98304, 98415, 100000, 101250, 102400, 103680, 104976, 108000, 
	109350, 110592, 112500, 115200, 116640, 118098, 120000, 121500, 122880, 124416, 125000, 
	128000, 129600, 131072, 131220, 135000, 138240, 139968, 140625, 144000, 145800, 147456, 
	150000, 151875, 153600, 155520, 156250, 157464, 160000, 162000, 163840, 164025, 165888, 
	168750, 172800, 174960, 177147, 180000, 182250, 184320, 186624, 187500, 192000, 194400, 
	196608, 196830, 200000, 202500, 204800, 207360, 209952, 216000, 218700, 221184, 225000, 
	230400, 233280, 234375, 236196, 240000, 243000, 245760, 248832, 250000, 253125, 256000, 
	259200, 262144, 262440, 270000, 273375, 276480, 279936, 281250, 288000, 291600, 294912, 
	295245, 300000, 303750, 307200, 311040, 312500, 314928, 320000, 324000, 327680, 328050, 
	331776, 337500, 345600, 349920, 354294, 360000, 364500, 368640, 373248, 375000, 384000, 
	388800, 390625, 393216, 393660, 400000, 405000, 409600, 414720, 419904, 421875, 432000, 
	437400, 442368, 450000, 455625, 460800, 466560, 468750, 472392, 480000, 486000, 491520, 
	492075, 497664, 500000, 506250, 512000, 518400, 524288, 524880, 531441, 540000, 546750, 
	552960, 559872, 562500, 576000, 583200, 589824, 590490, 600000, 607500, 614400, 622080, 
	625000, 629856, 640000, 648000, 655360, 656100, 663552, 675000, 691200, 699840, 703125, 
	708588, 720000, 729000, 737280, 746496, 750000, 759375, 768000, 777600, 781250, 786432, 
	787320, 800000, 810000, 819200, 820125, 829440, 839808, 843750, 864000, 874800, 884736, 
	885735, 900000, 911250, 921600, 933120, 937500, 944784, 960000, 972000, 983040, 984150, 
	995328, 1000000, 1012500, 1024000, 1036800, 1048576, 1049760, 1062882, 1080000, 1093500, 
	1105920, 1119744, 1125000, 1152000, 1166400, 1171875, 1179648, 1180980, 1200000, 
	1215000, 1228800, 1244160, 1250000, 1259712, 1265625, 1280000, 1296000, 1310720, 
	1312200, 1327104, 1350000, 1366875, 1382400, 1399680, 1406250, 1417176, 1440000, 
	1458000, 1474560, 1476225, 1492992, 1500000, 1518750, 1536000, 1555200, 1562500, 
	1572864, 1574640, 1594323, 1600000, 1620000, 1638400, 1640250, 1658880, 1679616, 
	1687500, 1728000, 1749600, 1769472, 1771470, 1800000, 1822500, 1843200, 1866240, 
	1875000, 1889568, 1920000, 1944000, 1953125, 1966080, 1968300, 1990656, 2000000, 
	2025000, 2048000, 2073600, 2097152, 2099520, 2109375, 2125764, 2160000, 2187000, 
	2211840, 2239488, 2250000, 2278125, 2304000, 2332800, 2343750, 2359296, 2361960, 
	2400000, 2430000, 2457600, 2460375, 2488320, 2500000, 2519424, 2531250, 2560000, 
	2592000, 2621440, 2624400, 2654208, 2657205, 2700000, 2733750, 2764800, 2799360, 
	2812500, 2834352, 2880000, 2916000, 2949120, 2952450, 2985984, 3000000, 3037500, 
	3072000, 3110400, 3125000, 3145728, 3149280, 3188646, 3200000, 3240000, 3276800, 
	3280500, 3317760, 3359232, 3375000, 3456000, 3499200, 3515625, 3538944, 3542940, 
	3600000, 3645000, 3686400, 3732480, 3750000, 3779136, 3796875, 3840000, 3888000, 
	3906250, 3932160, 3936600, 3981312, 4000000, 4050000, 4096000, 4100625, 4147200, 
	4194304, 4199040, 4218750, 4251528, 4320000, 4374000, 4423680, 4428675, 4478976, 
	4500000, 4556250, 4608000, 4665600, 4687500, 4718592, 4723920, 4782969, 4800000, 
	4860000, 4915200, 4920750, 4976640, 5000000, 5038848, 5062500, 5120000, 5184000, 
	5242880, 5248800, 5308416, 5314410, 5400000, 5467500, 5529600, 5598720, 5625000, 
	5668704, 5760000, 5832000, 5859375, 5898240, 5904900, 5971968, 6000000, 6075000, 
	6144000, 6220800, 6250000, 6291456, 6298560, 6328125, 6377292, 6400000, 6480000, 
	6553600, 6561000, 6635520, 6718464, 6750000, 6834375, 6912000, 6998400, 7031250, 
	7077888, 7085880, 7200000, 7290000, 7372800, 7381125, 7464960, 7500000, 7558272, 
	7593750, 7680000, 7776000, 7812500, 7864320, 7873200, 7962624, 7971615, 8000000, 
	8100000, 8192000, 8201250, 8294400, 8388608, 8398080, 8437500, 8503056, 8640000, 
	8748000, 8847360, 8857350, 8957952, 9000000, 9112500, 9216000, 9331200, 9375000, 
	9437184, 9447840, 9565938, 9600000, 9720000, 9765625, 9830400, 9841500, 9953280, 
	10000000, 10077696, 10125000, 10240000, 10368000, 10485760, 10497600, 10546875, 10616832, 
	10628820, 10800000, 10935000, 11059200, 11197440, 11250000, 11337408, 11390625, 11520000, 
	11664000, 11718750, 11796480, 11809800, 11943936, 12000000, 12150000, 12288000, 12301875, 
	12441600, 12500000, 12582912, 12597120, 12656250, 12754584, 12800000, 12960000, 13107200, 
	13122000, 13271040, 13286025, 13436928, 13500000, 13668750, 13824000, 13996800, 14062500, 
	14155776, 14171760, 14400000, 14580000, 14745600, 14762250, 14929920, 15000000, 15116544, 
	15187500, 15360000, 15552000, 15625000, 15728640, 15746400, 15925248, 15943230, 16000000, 
	16200000, 16384000, 16402500, 16588800, 16777216, 16796160, 16875000, 17006112, 17280000, 
	17496000, 17578125, 17694720, 17714700, 17915904, 18000000, 18225000, 18432000, 18662400, 
	18750000, 18874368, 18895680, 18984375, 19131876, 19200000, 19440000, 19531250, 19660800, 
	19683000, 19906560, 20000000, 20155392, 20250000, 20480000, 20503125, 20736000, 20971520, 
	20995200, 21093750, 21233664, 21257640, 21600000, 21870000, 22118400, 22143375, 22394880, 
	22500000, 22674816, 22781250, 23040000, 23328000, 23437500, 23592960, 23619600, 23887872, 
	23914845, 24000000, 24300000, 24576000, 24603750, 24883200, 25000000, 25165824, 25194240, 
	25312500, 25509168, 25600000, 25920000, 26214400, 26244000, 26542080, 26572050, 26873856, 
	27000000, 27337500, 27648000, 27993600, 28125000, 28311552, 28343520, 28800000, 29160000, 
	29296875, 29491200, 29524500, 29859840, 30000000, 30233088, 30375000, 30720000, 31104000, 
	31250000, 31457280, 31492800, 31640625, 31850496, 31886460, 32000000, 32400000, 32768000, 
	32805000, 33177600, 33554432, 33592320, 33750000, 34012224, 34171875, 34560000, 34992000, 
	35156250, 35389440, 35429400, 35831808, 36000000, 36450000, 36864000, 36905625, 37324800, 
	37500000, 37748736, 37791360, 37968750, 38263752, 38400000, 38880000, 39062500, 39321600, 
	39366000, 39813120, 39858075, 40000000, 40310784, 40500000, 40960000, 41006250, 41472000, 
	41943040, 41990400, 42187500, 42467328, 42515280, 43200000, 43740000, 44236800, 44286750, 
	44789760, 45000000, 45349632, 45562500, 46080000, 46656000, 46875000, 47185920, 47239200, 
	47775744, 47829690, 48000000, 48600000, 48828125, 49152000, 49207500, 49766400, 50000000, 
	50331648, 50388480, 50625000, 51018336, 51200000, 51840000, 52428800, 52488000, 52734375, 
	53084160, 53144100, 53747712, 54000000, 54675000, 55296000, 55987200, 56250000, 56623104, 
	56687040, 56953125, 57600000, 58320000, 58593750, 58982400, 59049000, 59719680, 60000000, 
	60466176, 60750000, 61440000, 61509375, 62208000, 62500000, 62914560, 62985600, 63281250, 
	63700992, 63772920, 64000000, 64800000, 65536000, 65610000, 66355200, 66430125, 67108864, 
	67184640, 67500000, 68024448, 68343750, 69120000, 69984000, 70312500, 70778880, 70858800, 
	71663616, 72000000, 72900000, 73728000, 73811250, 74649600, 75000000, 75497472, 75582720, 
	75937500, 76527504, 76800000, 77760000, 78125000, 78643200, 78732000, 79626240, 79716150, 
	80000000, 80621568, 81000000, 81920000, 82012500, 82944000, 83886080, 83980800, 84375000, 
	84934656, 85030560, 86400000, 87480000, 87890625, 88473600, 88573500, 89579520, 90000000, 
	90699264, 91125000, 92160000, 93312000, 93750000, 94371840, 94478400, 94921875, 95551488, 
	95659380, 96000000, 97200000, 97656250, 98304000, 98415000, 99532800, 100000000, 
	100663296, 100776960, 101250000, 102036672, 102400000, 102515625, 103680000, 104857600, 
	104976000, 105468750, 106168320, 106288200, 107495424, 108000000, 109350000, 110592000, 
	110716875, 111974400, 112500000, 113246208, 113374080, 113906250, 115200000, 116640000, 
	117187500, 117964800, 118098000, 119439360, 119574225, 120000000, 120932352, 121500000, 
	122880000, 123018750, 124416000, 125000000, 125829120, 125971200, 126562500, 127401984, 
	127545840, 128000000, 129600000, 131072000, 131220000, 132710400, 132860250, 134217728, 
	134369280, 135000000, 136048896, 136687500, 138240000, 139968000, 140625000, 141557760, 
	141717600, 143327232, 144000000, 145800000, 146484375, 147456000, 147622500, 149299200, 
	150000000, 150994944, 151165440, 151875000, 153055008, 153600000, 155520000, 156250000, 
	157286400, 157464000, 158203125, 159252480, 159432300, 160000000, 161243136, 162000000, 
	163840000, 164025000, 165888000, 167772160, 167961600, 168750000, 169869312, 170061120, 
	170859375, 172800000, 174960000, 175781250, 176947200, 177147000, 179159040, 180000000, 
	181398528, 182250000, 184320000, 184528125, 186624000, 187500000, 188743680, 188956800, 
	189843750, 191102976, 191318760, 192000000, 194400000, 195312500, 196608000, 196830000, 
	199065600, 199290375, 200000000, 201326592, 201553920, 202500000, 204073344, 204800000, 
	205031250, 207360000, 209715200, 209952000, 210937500, 212336640, 212576400, 214990848, 
	216000000, 218700000, 221184000, 221433750, 223948800, 225000000, 226492416, 226748160, 
	227812500, 230400000, 233280000, 234375000, 235929600, 236196000, 238878720, 239148450, 
	240000000, 241864704, 243000000, 244140625, 245760000, 246037500, 248832000, 250000000, 
	251658240, 251942400, 253125000, 254803968, 255091680, 256000000, 259200000, 262144000, 
	262440000, 263671875, 265420800, 265720500, 268435456, 268738560, 270000000, 272097792, 
	273375000, 276480000, 279936000, 281250000, 283115520, 283435200, 284765625, 286654464, 
	288000000, 291600000, 292968750, 294912000, 295245000, 298598400, 300000000, 301989888, 
	302330880, 303750000, 306110016, 307200000, 307546875, 311040000, 312500000, 314572800, 
	314928000, 316406250, 318504960, 318864600, 320000000, 322486272, 324000000, 327680000, 
	328050000, 331776000, 332150625, 335544320, 335923200, 337500000, 339738624, 340122240, 
	341718750, 345600000, 349920000, 351562500, 353894400, 354294000, 358318080, 360000000, 
	362797056, 364500000, 368640000, 369056250, 373248000, 375000000, 377487360, 377913600, 
	379687500, 382205952, 382637520, 384000000, 388800000, 390625000, 393216000, 393660000, 
	398131200, 398580750, 400000000, 402653184, 403107840, 405000000, 408146688, 409600000, 
	410062500, 414720000, 419430400, 419904000, 421875000, 424673280, 425152800, 429981696, 
	432000000, 437400000, 439453125, 442368000, 442867500, 447897600, 450000000, 452984832, 
	453496320, 455625000, 460800000, 466560000, 468750000, 471859200, 472392000, 474609375, 
	477757440, 478296900, 480000000, 483729408, 486000000, 488281250, 491520000, 492075000, 
	497664000, 500000000, 503316480, 503884800, 506250000, 509607936, 510183360, 512000000, 
	512578125, 518400000, 524288000, 524880000, 527343750, 530841600, 531441000, 536870912, 
	537477120, 540000000, 544195584, 546750000, 552960000, 553584375, 559872000, 562500000, 
	566231040, 566870400, 569531250, 573308928, 576000000, 583200000, 585937500, 589824000, 
	590490000, 597196800, 597871125, 600000000, 603979776, 604661760, 607500000, 612220032, 
	614400000, 615093750, 622080000, 625000000, 629145600, 629856000, 632812500, 637009920, 
	637729200, 640000000, 644972544, 648000000, 655360000, 656100000, 663552000, 664301250, 
	671088640, 671846400, 675000000, 679477248, 680244480, 683437500, 691200000, 699840000, 
	703125000, 707788800, 708588000, 716636160, 720000000, 725594112, 729000000, 732421875, 
	737280000, 738112500, 746496000, 750000000, 754974720, 755827200, 759375000, 764411904, 
	765275040, 768000000, 777600000, 781250000, 786432000, 787320000, 791015625, 796262400, 
	797161500, 800000000, 805306368, 806215680, 810000000, 816293376, 819200000, 820125000, 
	829440000, 838860800, 839808000, 843750000, 849346560, 850305600, 854296875, 859963392, 
	864000000, 874800000, 878906250, 884736000, 885735000, 895795200, 900000000, 905969664, 
	906992640, 911250000, 921600000, 922640625, 933120000, 937500000, 943718400, 944784000, 
	949218750, 955514880, 956593800, 960000000, 967458816, 972000000, 976562500, 983040000, 
	984150000, 995328000, 996451875, 1000000000, 1006632960, 1007769600, 1012500000, 
	1019215872, 1020366720, 1024000000, 1025156250, 1036800000, 1048576000, 1049760000, 
	1054687500, 1061683200, 1062882000, 1073741824, 1074954240, 1080000000, 1088391168, 
	1093500000, 1105920000, 1107168750, 1119744000, 1125000000, 1132462080, 1133740800, 
	1139062500, 1146617856, 1152000000, 1166400000, 1171875000, 1179648000, 1180980000, 
	1194393600, 1195742250, 1200000000, 1207959552, 1209323520, 1215000000, 1220703125, 
	1224440064, 1228800000, 1230187500, 1244160000, 1250000000, 1258291200, 1259712000, 
	1265625000, 1274019840, 1275458400, 1280000000, 1289945088, 1296000000, 1310720000, 
	1312200000, 1318359375, 1327104000, 1328602500, 1342177280, 1343692800, 1350000000, 
	1358954496, 1360488960, 1366875000, 1382400000, 1399680000, 1406250000, 1415577600, 
	1417176000, 1423828125, 1433272320, 1440000000, 1451188224, 1458000000, 1464843750, 
	1474560000, 1476225000, 1492992000, 1500000000, 1509949440, 1511654400, 1518750000, 
	1528823808, 1530550080, 1536000000, 1537734375, 1555200000, 1562500000, 1572864000, 
	1574640000, 1582031250, 1592524800, 1594323000, 1600000000, 1610612736, 1612431360, 
	1620000000, 1632586752, 1638400000, 1640250000, 1658880000, 1660753125, 1677721600, 
	1679616000, 1687500000, 1698693120, 1700611200, 1708593750, 1719926784, 1728000000, 
	1749600000, 1757812500, 1769472000, 1771470000, 1791590400, 1800000000, 1811939328, 
	1813985280, 1822500000, 1843200000, 1845281250, 1866240000, 1875000000, 1887436800, 
	1889568000, 1898437500, 1911029760, 1913187600, 1920000000, 1934917632, 1944000000, 
	1953125000, 1966080000, 1968300000, 1990656000, 1992903750, 2000000000, 2013265920, 
	2015539200, 2025000000, 2038431744, 2040733440, 2048000000, 2050312500, 2073600000, 
	2097152000, 2099520000, 2109375000, 2123366400, 2125764000
};

static int _ccv_get_optimal_fft_size(int size)
{
	int a = 0, b = sizeof(_ccv_optimal_fft_size)/sizeof(_ccv_optimal_fft_size[0]) - 1;
    if((unsigned)size >= (unsigned)_ccv_optimal_fft_size[b])
		return -1;
	while(a < b)
	{
		int c = (a + b) >> 1;
		if(size <= _ccv_optimal_fft_size[c])
			b = c;
		else
			a = c + 1;
    }
    return _ccv_optimal_fft_size[b];
}

#ifdef HAVE_FFTW3
static void _ccv_filter_fftw(ccv_dense_matrix_t* a, ccv_dense_matrix_t* b, ccv_dense_matrix_t* d)
{
	int rows = ccv_min(a->rows, _ccv_get_optimal_fft_size(b->rows * 3));
	int cols = ccv_min(a->cols, _ccv_get_optimal_fft_size(b->cols * 3));
	int cols_2c = 2 * (cols / 2 + 1);
	double* fftw_a = (double*)fftw_malloc(rows * cols_2c * sizeof(double));
	double* fftw_b = (double*)fftw_malloc(rows * cols_2c * sizeof(double));
	memset(fftw_b, 0, rows * cols_2c * sizeof(double));
	double* fftw_d = (double*)fftw_malloc(rows * cols * sizeof(double));
	fftw_complex* fftw_ac = (fftw_complex*)fftw_a;
	fftw_complex* fftw_bc = (fftw_complex*)fftw_b;
	fftw_complex* fftw_dc = (fftw_complex*)fftw_malloc(rows * (cols / 2 + 1) * sizeof(fftw_complex));
	fftw_plan p, pinv;
	double scale = 1.0 / (rows * cols);
	p = fftw_plan_dft_r2c_2d(rows, cols, 0, 0, FFTW_ESTIMATE);
	pinv = fftw_plan_dft_c2r_2d(rows, cols, fftw_dc, fftw_d, FFTW_ESTIMATE);
	double* fftw_ptr;
	unsigned char* m_ptr;

	/* discrete kernel is always meant to be (0,0) centered, but in most case, it is (0,0) toplefted.
	 * to compensate that, Fourier function will assume that it is a periodic function, which, will
	 * result the following interleaving:
	 * | 0 1 2 3 |    | A B 8 9 |
	 * | 4 5 6 7 | to | E F C D |
	 * | 8 9 A B |    | 2 3 0 1 |
	 * | C D E F |    | 4 7 4 5 |
	 * a more classic way to do this is to pad both a and b to be a->rows + b->rows - 1,
	 * a->cols + b->cols - 1, but it is too expensive. In the way we introduced here, we also assume
	 * a border padding pattern in periodical way: |cd{BORDER}|abcd|{BORDER}ab|. */
	int i, j;
	if (b->rows > rows || b->cols > cols)
	{ /* reverse lookup */
		fftw_ptr = fftw_b;
		for (i = 0; i < rows; i++)
		{
			int y = (i + rows / 2) % rows - rows / 2 + b->rows / 2;
			for (j = 0; j < cols; j++)
			{
				int x = (j + cols / 2) % cols - cols / 2 + b->cols / 2;
				fftw_ptr[j] = (y >= 0 && y < b->rows && x >= 0 && x < b->cols) ? ccv_get_dense_matrix_cell_value(b, y, x) : 0;
			}
			fftw_ptr += cols_2c;
		}
	} else { /* forward lookup */
		int rows_bc = rows - b->rows / 2;
		int cols_bc = cols - b->cols / 2;
		for (i = 0; i < b->rows; i++)
		{
			int y = (i + rows_bc) % rows;
			for (j = 0; j < b->cols; j++)
				fftw_b[y * cols_2c + (j + cols_bc) % cols] = ccv_get_dense_matrix_cell_value(b, i, j);
		}
	}
	fftw_execute_dft_r2c(p, fftw_b, fftw_bc);

	int tile_x = ccv_max(1, (a->cols + cols - b->cols - 1) / (cols - b->cols));
	int tile_y = ccv_max(1, (a->rows + rows - b->rows - 1) / (rows - b->rows));
	/* do FFT for each tile */
#define for_block(_for_set, _for_get) \
	for (i = 0; i < tile_y; i++) \
		for (j = 0; j < tile_x; j++) \
		{ \
			int x, y; \
			memset(fftw_a, 0, rows * cols * sizeof(double)); \
			int iy = ccv_min(i * (rows - b->rows), a->rows - rows); \
			int ix = ccv_min(j * (cols - b->cols), a->cols - cols); \
			fftw_ptr = fftw_a; \
			m_ptr = (unsigned char*)ccv_get_dense_matrix_cell(a, iy, ix); \
			for (y = 0; y < rows; y++) \
			{ \
				for (x = 0; x < cols; x++) \
					fftw_ptr[x] = _for_get(m_ptr, x, 0); \
				fftw_ptr += cols_2c; \
				m_ptr += a->step; \
			} \
			fftw_execute_dft_r2c(p, fftw_a, fftw_ac); \
			for (x = 0; x < rows * (cols / 2 + 1); x++) \
				fftw_dc[x] = (fftw_ac[x] * fftw_bc[x]) * scale; \
			fftw_execute_dft_c2r(pinv, fftw_dc, fftw_d); \
			fftw_ptr = fftw_d + (i > 0) * b->rows / 2 * cols + (j > 0) * b->cols / 2; \
			int end_y = ccv_min(d->rows - iy, (rows - b->rows) + (i == 0) * b->rows / 2 + (i + 1 == tile_y) * (b->rows + 1) / 2); \
			int end_x = ccv_min(d->cols - ix, (cols - b->cols) + (j == 0) * b->cols / 2 + (j + 1 == tile_x) * (b->cols + 1) / 2); \
			m_ptr = (unsigned char*)ccv_get_dense_matrix_cell(d, iy + (i > 0) * b->rows / 2, ix + (j > 0) * b->cols / 2); \
			for (y = 0; y < end_y; y++) \
			{ \
				for (x = 0; x < end_x; x++) \
					_for_set(m_ptr, x, fftw_ptr[x], 0); \
				m_ptr += d->step; \
				fftw_ptr += cols; \
			} \
		}
	ccv_matrix_setter(d->type, ccv_matrix_getter, a->type, for_block);
#undef for_block
	fftw_destroy_plan(p);
	fftw_destroy_plan(pinv);
	fftw_free(fftw_a);
	fftw_free(fftw_b);
	fftw_free(fftw_d);
	fftw_free(fftw_dc);
}
#endif

void _ccv_filter_direct_8u(ccv_dense_matrix_t* a, ccv_dense_matrix_t* b, ccv_dense_matrix_t* d)
{
	int i, j, y, x, k;
	int nz = b->rows * b->cols;
	int* coeff = (int*)alloca(nz * sizeof(int));
	int* cx = (int*)alloca(nz * sizeof(int));
	int* cy = (int*)alloca(nz * sizeof(int));
	int scale = 1 << 14;
	nz = 0;
	for (i = 0; i < b->rows; i++)
		for (j = 0; j < b->cols; j++)
		{
			coeff[nz] = (int)(ccv_get_dense_matrix_cell_value(b, i, j) * scale + 0.5);
			if (coeff[nz] == 0)
				continue;
			cy[nz] = i;
			cx[nz] = j;
			nz++;
		}
	ccv_dense_matrix_t* pa = ccv_dense_matrix_new(a->rows + b->rows / 2 * 2, a->cols + b->cols / 2 * 2, CCV_8U | CCV_C1, 0, 0);
	/* the padding pattern is different from FFT: |aa{BORDER}|abcd|{BORDER}dd| */
	for (i = 0; i < pa->rows; i++)
		for (j = 0; j < pa->cols; j++)
			pa->data.ptr[i * pa->step + j] = a->data.ptr[ccv_clamp(i - b->rows / 2, 0, a->rows - 1) * a->step + ccv_clamp(j - b->cols / 2, 0, a->cols - 1)];
	unsigned char* m_ptr = d->data.ptr;
	unsigned char* a_ptr = pa->data.ptr;
	/* 0.5 denote the overhead for indexing x and y */
	if (nz < b->rows * b->cols * 0.75)
	{
		for (i = 0; i < d->rows; i++)
		{
			for (j = 0; j < d->cols; j++)
			{
				int z = 0;
				for (k = 0; k < nz; k++)
					z += a_ptr[cy[k] * pa->step + j + cx[k]] * coeff[k];
				m_ptr[j] = ccv_clamp(z >> 14, 0, 255);
			}
			m_ptr += d->step;
			a_ptr += pa->step;
		}
	} else {
		k = 0;
		for (i = 0; i < b->rows; i++)
			for (j = 0; j < b->cols; j++)
			{
				coeff[k] = (int)(ccv_get_dense_matrix_cell_value(b, i, j) * scale + 0.5);
				k++;
			}
		for (i = 0; i < d->rows; i++)
		{
			for (j = 0; j < d->cols; j++)
			{
				int* c_ptr = coeff;
				int z = 0;
				for (y = 0; y < b->rows; y++)
				{
					int iyx = y * pa->step;
					for (x = 0; x < b->cols; x++)
					{
						z += a_ptr[iyx + j + x] * c_ptr[0];
						c_ptr++;
					}
				}
				m_ptr[j] = ccv_clamp(z >> 14, 0, 255);
			}
			m_ptr += d->step;
			a_ptr += pa->step;
		}
	}
	ccv_matrix_free(pa);
}

void ccv_filter(ccv_matrix_t* a, ccv_matrix_t* b, ccv_matrix_t** d, int type)
{
	ccv_dense_matrix_t* da = ccv_get_dense_matrix(a);
	ccv_dense_matrix_t* db = ccv_get_dense_matrix(b);
	uint64_t sig = (da->sig == 0 || db->sig == 0) ? 0 : ccv_matrix_generate_signature("ccv_filter", 10, da->sig, db->sig, 0);
	type = (type == 0) ? CCV_GET_DATA_TYPE(da->type) | CCV_GET_CHANNEL(da->type) : CCV_GET_DATA_TYPE(type) | CCV_GET_CHANNEL(da->type);
	ccv_dense_matrix_t* dd = *d = ccv_dense_matrix_renew(*d, da->rows, da->cols, CCV_ALL_DATA_TYPE | CCV_GET_CHANNEL(da->type), type, sig);
	ccv_cache_return(dd, );

	/* 15 is the constant to indicate the high cost of FFT (even with O(nlog(m)) for
	 * integer image.
	 * NOTE: FFT has time complexity of O(nlog(n)), however, for convolution, it
	 * is not the case. Convolving one image (a) to a kernel (b), can be done by
	 * dividing image a to several blocks proportional to (b). Thus, one don't need
	 * to do FFT for the whole image. The image can be divided to n/m part, and
	 * the FFT itself is O(mlog(m)), so, the convolution process has time complexity
	 * of O(nlog(m)) */
	if ((db->rows * db->cols < (log((double)(db->rows * db->cols)) + 1) * 15) && (da->type & CCV_8U))
	{
		_ccv_filter_direct_8u(da, db, dd);
	} else {
#ifdef HAVE_FFTW3
		_ccv_filter_fftw(da, db, dd);
#endif
	}
}

void ccv_filter_kernel(ccv_dense_matrix_t* x, ccv_filter_kernel_f func, void* data)
{
	int i, j;
	unsigned char* m_ptr = x->data.ptr;
	double rows_2 = (x->rows - 1) * 0.5;
	double cols_2 = (x->cols - 1) * 0.5;
#define for_block(_, _for_set) \
	for (i = 0; i < x->rows; i++) \
	{ \
		for (j = 0; j < x->cols; j++) \
			_for_set(m_ptr, j, func(j - cols_2, i - rows_2, data), 0); \
		m_ptr += x->step; \
	}
	ccv_matrix_setter(x->type, for_block);
#undef for_block
	ccv_matrix_generate_signature((char*) x->data.ptr, x->rows * x->step, x->sig, 0);
}

/* End of ccv/lib/ccv_numeric.c */
#line 1 "ccv/lib/ccv_sift.c"
/* The code is adopted from VLFeat with heavily rewrite.
 * The original code is licenced under GPLv2, should be compatible
 * with New BSD Licence used by ccv. The original Copyright:
 *
 * AUTORIGHTS
 * Copyright (C) 2007-10 Andrea Vedaldi and Brian Fulkerson
 *
 * This file is part of VLFeat, available under the terms of the
 * GNU GPLv2, or (at your option) any later version.
 */
/* ccv.h already included */


inline static double _ccv_keypoint_interpolate(float N9[3][9], int ix, int iy, int is, ccv_keypoint_t* kp)
{
	double Dxx = N9[1][3] - 2 * N9[1][4] + N9[1][5]; 
	double Dyy = N9[1][1] - 2 * N9[1][4] + N9[1][7];
	double Dxy = (N9[1][8] - N9[1][6] - N9[1][2] + N9[1][0]) * 0.25;
	double score = (Dxx + Dyy) * (Dxx + Dyy) / (Dxx * Dyy - Dxy * Dxy);
	double Dx = (N9[1][5] - N9[1][3]) * 0.5;
	double Dy = (N9[1][7] - N9[1][1]) * 0.5;
	double Ds = (N9[2][4] - N9[0][4]) * 0.5;
	double Dxs = (N9[2][5] + N9[0][3] - N9[2][3] - N9[0][5]) * 0.25;
	double Dys = (N9[2][7] + N9[0][1] - N9[2][1] - N9[0][7]) * 0.25;
	double Dss = N9[0][4] - 2 * N9[1][4] + N9[2][4];
	double A[3][3] = { { Dxx, Dxy, Dxs },
					   { Dxy, Dyy, Dys },
					   { Dxs, Dys, Dss } };
	double b[3] = { -Dx, -Dy, -Ds };
	/* Gauss elimination */
	int i, j, ii, jj;
	for(j = 0; j < 3; j++)
	{
		double maxa = 0;
		double maxabsa = 0;
		int maxi = -1;
		double tmp;

		/* look for the maximally stable pivot */
		for (i = j; i < 3; i++)
		{
			double a = A[i][j];
			double absa = fabs(a);
			if (absa > maxabsa)
			{
				maxa = a;
				maxabsa = absa;
				maxi = i;
			}
		}

		/* if singular give up */
		if (maxabsa < 1e-10f)
		{
			b[0] = b[1] = b[2] = 0;
			break;
		}

		i = maxi;

		/* swap j-th row with i-th row and normalize j-th row */
		for(jj = j; jj < 3; jj++)
		{
			tmp = A[i][jj];
			A[i][jj] = A[j][jj];
			A[j][jj] = tmp;
			A[j][jj] /= maxa;
		}
		tmp = b[j];
		b[j] = b[i];
		b[i] = tmp;
		b[j] /= maxa;

		/* elimination */
		for (ii = j + 1; ii < 3; ii++)
		{
			double x = A[ii][j];
			for (jj = j; jj < 3; jj++)
				A[ii][jj] -= x * A[j][jj];
			b[ii] -= x * b[j];
		}
	}

	/* backward substitution */
	for (i = 2; i > 0; i--)
	{
		double x = b[i];
		for (ii = i - 1; ii >= 0; ii--)
		  b[ii] -= x * A[ii][i];
	}
	kp->x = ix + ccv_min(ccv_max(b[0], -1), 1);
	kp->y = iy + ccv_min(ccv_max(b[1], -1), 1);
	kp->regular.scale = is + b[2];
	return score;
}

static float _ccv_mod_2pi(float x)
{
	while (x > 2 * CCV_PI)
		x -= 2 * CCV_PI;
	while (x < 0)
		x += 2 * CCV_PI;
	return x;
}

static int _ccv_floor(float x)
{
	int xi = (int)x;
	if (x >= 0 || (float)xi == x)
		return xi;
	return xi - 1;
}

#define EXPN_SZ  256 /* fast_expn table size */
#define EXPN_MAX 25.0 /* fast_expn table max */
static double _ccv_expn_tab[EXPN_SZ + 1]; /* fast_expn table */
static int _ccv_expn_init = 0;

static inline double _ccv_expn(double x)
{
	double a, b, r;
	int i;
	assert(0 <= x && x <= EXPN_MAX);
	if (x > EXPN_MAX)
		return 0.0;
	x *= EXPN_SZ / EXPN_MAX;
	i = (int)x;
	r = x - i;
	a = _ccv_expn_tab[i];
	b = _ccv_expn_tab[i + 1];
	return a + r * (b - a);
}

static void _ccv_precomputed_expn()
{
	int i;
	for(i = 0; i < EXPN_SZ + 1; i++)
		_ccv_expn_tab[i] = exp(-(double)i * (EXPN_MAX / EXPN_SZ));
	_ccv_expn_init = 1;
}

void ccv_sift(ccv_dense_matrix_t* a, ccv_array_t** _keypoints, ccv_dense_matrix_t** _desc, int type, ccv_sift_param_t params)
{
	assert(CCV_GET_CHANNEL(a->type) == CCV_C1);
	ccv_dense_matrix_t** g = (ccv_dense_matrix_t**)alloca(sizeof(ccv_dense_matrix_t*) * (params.nlevels + 1) * (params.up2x ? params.noctaves + 1 : params.noctaves));
	memset(g, 0, sizeof(ccv_dense_matrix_t*) * (params.nlevels + 1) * (params.up2x ? params.noctaves + 1 : params.noctaves));
	ccv_dense_matrix_t** dog = (ccv_dense_matrix_t**)alloca(sizeof(ccv_dense_matrix_t*) * (params.nlevels - 1) * (params.up2x ? params.noctaves + 1 : params.noctaves));
	memset(dog, 0, sizeof(ccv_dense_matrix_t*) * (params.nlevels - 1) * (params.up2x ? params.noctaves + 1 : params.noctaves));
	ccv_dense_matrix_t** th = (ccv_dense_matrix_t**)alloca(sizeof(ccv_dense_matrix_t*) * (params.nlevels - 3) * (params.up2x ? params.noctaves + 1 : params.noctaves));
	memset(th, 0, sizeof(ccv_dense_matrix_t*) * (params.nlevels - 3) * (params.up2x ? params.noctaves + 1 : params.noctaves));
	ccv_dense_matrix_t** md = (ccv_dense_matrix_t**)alloca(sizeof(ccv_dense_matrix_t*) * (params.nlevels - 3) * (params.up2x ? params.noctaves + 1 : params.noctaves));
	memset(md, 0, sizeof(ccv_dense_matrix_t*) * (params.nlevels - 3) * (params.up2x ? params.noctaves + 1 : params.noctaves));
	if (params.up2x)
	{
		g += params.nlevels + 1;
		dog += params.nlevels - 1;
		th += params.nlevels - 3;
		md += params.nlevels - 3;
	}
	ccv_array_t* keypoints = *_keypoints;
	int custom_keypoints = 0;
	if (keypoints == 0)
		keypoints = *_keypoints = ccv_array_new(10, sizeof(ccv_keypoint_t));
	else
		custom_keypoints = 1;
	int i, j, k, x, y;
	double sigma0 = 1.6;
	double sigmak = pow(2.0, 1.0 / (params.nlevels - 3));
	double dsigma0 = sigma0 * sigmak * sqrt(1.0 - 1.0 / (sigmak * sigmak));
	if (params.up2x)
	{
		ccv_sample_up(a, &g[-(params.nlevels + 1)], 0, 0, 0);
		/* since there is a gaussian filter in sample_up function already,
		 * the default sigma for upsampled image is sqrt(2) */
		double sd = sqrt(sigma0 * sigma0 - 2.0);
		ccv_blur(g[-(params.nlevels + 1)], &g[-(params.nlevels + 1) + 1], CCV_32F | CCV_C1, sd);
		ccv_matrix_free(g[-(params.nlevels + 1)]);
		for (j = 1; j < params.nlevels; j++)
		{
			sd = dsigma0 * pow(sigmak, j - 1);
			ccv_blur(g[-(params.nlevels + 1) + j], &g[-(params.nlevels + 1) + j + 1], 0, sd);
			ccv_substract(g[-(params.nlevels + 1) + j + 1], g[-(params.nlevels + 1) + j], (ccv_matrix_t**)&dog[-(params.nlevels - 1) + j - 1], 0);
			if (j > 1 && j < params.nlevels - 1)
				ccv_gradient(g[-(params.nlevels + 1) + j], &th[-(params.nlevels - 3) + j - 2], 0, &md[-(params.nlevels - 3) + j - 2], 0, 1, 1);
			ccv_matrix_free(g[-(params.nlevels + 1) + j]);
		}
		ccv_matrix_free(g[-1]);
	}
	double sd = sqrt(sigma0 * sigma0 - 0.25);
	g[0] = a;
	/* generate gaussian pyramid (g, dog) & gradient pyramid (th, md) */
	ccv_blur(g[0], &g[1], CCV_32F | CCV_C1, sd);
	for (j = 1; j < params.nlevels; j++)
	{
		sd = dsigma0 * pow(sigmak, j - 1);
		ccv_blur(g[j], &g[j + 1], 0, sd);
		ccv_substract(g[j + 1], g[j], (ccv_matrix_t**)&dog[j - 1], 0);
		if (j > 1 && j < params.nlevels - 1)
			ccv_gradient(g[j], &th[j - 2], 0, &md[j - 2], 0, 1, 1);
		ccv_matrix_free(g[j]);
	}
	ccv_matrix_free(g[params.nlevels]);
	for (i = 1; i < params.noctaves; i++)
	{
		ccv_sample_down(g[(i - 1) * (params.nlevels + 1)], &g[i * (params.nlevels + 1)], 0, 0, 0);
		if (i - 1 > 0)
			ccv_matrix_free(g[(i - 1) * (params.nlevels + 1)]);
		sd = sqrt(sigma0 * sigma0 - 0.25);
		ccv_blur(g[i * (params.nlevels + 1)], &g[i * (params.nlevels + 1) + 1], CCV_32F | CCV_C1, sd);
		for (j = 1; j < params.nlevels; j++)
		{
			sd = dsigma0 * pow(sigmak, j - 1);
			ccv_blur(g[i * (params.nlevels + 1) + j], &g[i * (params.nlevels + 1) + j + 1], 0, sd);
			ccv_substract(g[i * (params.nlevels + 1) + j + 1], g[i * (params.nlevels + 1) + j], (ccv_matrix_t**)&dog[i * (params.nlevels - 1) + j - 1], 0);
			if (j > 1 && j < params.nlevels - 1)
				ccv_gradient(g[i * (params.nlevels + 1) + j], &th[i * (params.nlevels - 3) + j - 2], 0, &md[i * (params.nlevels - 3) + j - 2], 0, 1, 1);
			ccv_matrix_free(g[i * (params.nlevels + 1) + j]);
		}
		ccv_matrix_free(g[i * (params.nlevels + 1) + params.nlevels]);
	}
	ccv_matrix_free(g[(params.noctaves - 1) * (params.nlevels + 1)]);
	if (!custom_keypoints)
	{
		/* detect keypoint */
		for (i = (params.up2x ? -1 : 0); i < params.noctaves; i++)
		{
			double s = pow(2.0, i);
			int rows = dog[i * (params.nlevels - 1)]->rows;
			int cols = dog[i * (params.nlevels - 1)]->cols;
			for (j = 1; j < params.nlevels - 2; j++)
			{
				float* bf = dog[i * (params.nlevels - 1) + j - 1]->data.fl + cols;
				float* cf = dog[i * (params.nlevels - 1) + j]->data.fl + cols;
				float* uf = dog[i * (params.nlevels - 1) + j + 1]->data.fl + cols;

				for (y = 1; y < rows - 1; y++)
				{
					x = 1;
					for (x = 1; x < cols - 1; x++)
					{
						float v = cf[x];
#define locality_if(CMP, SGN) \
	(v CMP ## = SGN params.peak_threshold && v CMP cf[x - 1] && v CMP cf[x + 1] && \
	 v CMP cf[x - cols - 1] && v CMP cf[x - cols] && v CMP cf[x - cols + 1] && \
	 v CMP cf[x + cols - 1] && v CMP cf[x + cols] && v CMP cf[x + cols + 1] && \
	 v CMP bf[x - 1] && v CMP bf[x] && v CMP bf[x + 1] && \
	 v CMP bf[x - cols - 1] && v CMP bf[x - cols] && v CMP bf[x - cols + 1] && \
	 v CMP bf[x + cols - 1] && v CMP bf[x + cols] && v CMP bf[x + cols + 1] && \
	 v CMP uf[x - 1] && v CMP uf[x] && v CMP uf[x + 1] && \
	 v CMP uf[x - cols - 1] && v CMP uf[x - cols] && v CMP uf[x - cols + 1] && \
	 v CMP uf[x + cols - 1] && v CMP uf[x + cols] && v CMP uf[x + cols + 1])
						if (locality_if(<, -) || locality_if(>, +))
						{
							ccv_keypoint_t kp;
							int ix = x, iy = y;
							double score = -1;
							int cvg = 0;
							int offset = ix + (iy - y) * cols;
							/* iteratively converge to meet subpixel accuracy */
							for (k = 0; k < 5; k++)
							{
								offset = ix + (iy - y) * cols;
								
								float N9[3][9] = { { bf[offset - cols - 1], bf[offset - cols], bf[offset - cols + 1],
													 bf[offset - 1], bf[offset], bf[offset + 1],
													 bf[offset + cols - 1], bf[offset + cols], bf[offset + cols + 1] },
												   { cf[offset - cols - 1], cf[offset - cols], cf[offset - cols + 1],
													 cf[offset - 1], cf[offset], cf[offset + 1],
													 cf[offset + cols - 1], cf[offset + cols], cf[offset + cols + 1] },
												   { uf[offset - cols - 1], uf[offset - cols], uf[offset - cols + 1],
													 uf[offset - 1], uf[offset], uf[offset + 1],
													 uf[offset + cols - 1], uf[offset + cols], uf[offset + cols + 1] } };
								score = _ccv_keypoint_interpolate(N9, ix, iy, j, &kp);
								if (kp.x >= 1 && kp.x <= cols - 2 && kp.y >= 1 && kp.y <= rows - 2)
								{
									int nx = (int)(kp.x + 0.5);
									int ny = (int)(kp.y + 0.5);
									if (ix == nx && iy == ny)
										break;
									ix = nx;
									iy = ny;
								} else {
									cvg = -1;
									break;
								}
							}
							if (cvg == 0 && fabs(cf[offset]) > params.peak_threshold && score >= 0 && score < (params.edge_threshold + 1) * (params.edge_threshold + 1) / params.edge_threshold && kp.regular.scale > 0 && kp.regular.scale < params.nlevels - 1)
							{
								kp.x *= s;
								kp.y *= s;
								kp.octave = i;
								kp.level = j;
								kp.regular.scale = sigma0 * sigmak * pow(2.0, kp.regular.scale / (double)(params.nlevels - 3));
								ccv_array_push(keypoints, &kp);
							}
						}
#undef locality_if
					}
					bf += cols;
					cf += cols;
					uf += cols;
				}
			}
		}
	}
	/* repeatable orientation/angle (p.s. it will push more keypoints (with different angles) to array) */
	float const winf = 1.5;
	double bins[36];
	int kpnum = keypoints->rnum;
	if (!_ccv_expn_init)
		_ccv_precomputed_expn();
	for (i = 0; i < kpnum; i++)
	{
		ccv_keypoint_t* kp = (ccv_keypoint_t*)ccv_array_get(keypoints, i);
		float ds = pow(2.0, kp->octave);
		float dx = kp->x / ds;
		float dy = kp->y / ds;
		int ix = (int)(dx + 0.5);
		int iy = (int)(dy + 0.5);
		float const sigmaw = winf * kp->regular.scale;
		int wz = ccv_max((int)(3.0 * sigmaw + 0.5), 1);
		ccv_dense_matrix_t* tho = th[kp->octave * (params.nlevels - 3) + kp->level - 1];
		ccv_dense_matrix_t* mdo = md[kp->octave * (params.nlevels - 3) + kp->level - 1];
		assert(tho->rows == mdo->rows && tho->cols == mdo->cols);
		if (ix >= 0 && ix < tho->cols && iy >=0 && iy < tho->rows)
		{
			float* theta = tho->data.fl + ccv_max(iy - wz, 0) * tho->cols;
			float* magnitude = mdo->data.fl + ccv_max(iy - wz, 0) * mdo->cols;
			memset(bins, 0, 36 * sizeof(double));
			/* oriented histogram with bilinear interpolation */
			for (y = ccv_max(iy - wz, 0); y <= ccv_min(iy + wz, tho->rows - 1); y++)
			{
				for (x = ccv_max(ix - wz, 0); x <= ccv_min(ix + wz, tho->cols - 1); x++)
				{
					float r2 = (x - dx) * (x - dx) + (y - dy) * (y - dy);
					if (r2 > wz * wz + 0.6)
						continue;
					float weight = _ccv_expn(r2 / (2.0 * sigmaw * sigmaw));
					float fbin = theta[x] * 0.1;
					int ibin = _ccv_floor(fbin - 0.5);
					float rbin = fbin - ibin - 0.5;
					/* bilinear interpolation */
					bins[(ibin + 36) % 36] += (1 - rbin) * magnitude[x] * weight;
					bins[(ibin + 1) % 36] += rbin * magnitude[x] * weight;
				}
				theta += tho->cols;
				magnitude += mdo->cols;
			}
			/* smoothing histogram */
			for (j = 0; j < 6; j++)
			{
				double first = bins[0];
				double prev = bins[35];
				for (k = 0; k < 35; k++)
				{
					double nb = (prev + bins[k] + bins[k + 1]) / 3.0;
					prev = bins[k];
					bins[k] = nb;
				}
				bins[35] = (prev + bins[35] + first) / 3.0;
			}
			int maxib = 0;
			for (j = 1; j < 36; j++)
				if (bins[j] > bins[maxib])
					maxib = j;
			double maxb = bins[maxib];
			double bm = bins[(maxib + 35) % 36];
			double bp = bins[(maxib + 1) % 36];
			double di = -0.5 * (bp - bm) / (bp + bm - 2 * maxb);
			kp->regular.angle = 2 * CCV_PI * (maxib + di + 0.5) / 36.0;
			maxb *= 0.8;
			for (j = 0; j < 36; j++)
				if (j != maxib)
				{
					bm = bins[(j + 35) % 36];
					bp = bins[(j + 1) % 36];
					if (bins[j] > maxb && bins[j] > bm && bins[j] > bp)
					{
						di = -0.5 * (bp - bm) / (bp + bm - 2 * bins[j]);
						ccv_keypoint_t nkp = *kp;
						nkp.regular.angle = 2 * CCV_PI * (j + di + 0.5) / 36.0;
						ccv_array_push(keypoints, &nkp);
					}
				}
		}
	}
	/* calculate descriptor */
	if (_desc != 0)
	{
		ccv_dense_matrix_t* desc = *_desc = ccv_dense_matrix_new(keypoints->rnum, 128, CCV_32F | CCV_C1, 0, 0);
		float* fdesc = desc->data.fl;
		memset(fdesc, 0, sizeof(float) * keypoints->rnum * 128);
		for (i = 0; i < keypoints->rnum; i++)
		{
			ccv_keypoint_t* kp = (ccv_keypoint_t*)ccv_array_get(keypoints, i);
			float ds = pow(2.0, kp->octave);
			float dx = kp->x / ds;
			float dy = kp->y / ds;
			int ix = (int)(dx + 0.5);
			int iy = (int)(dy + 0.5);
			double SBP = 3.0 * kp->regular.scale;
			int wz = ccv_max((int)(SBP * sqrt(2.0) * 2.5 + 0.5), 1);
			ccv_dense_matrix_t* tho = th[kp->octave * (params.nlevels - 3) + kp->level - 1];
			ccv_dense_matrix_t* mdo = md[kp->octave * (params.nlevels - 3) + kp->level - 1];
			assert(tho->rows == mdo->rows && tho->cols == mdo->cols);
			assert(ix >= 0 && ix < tho->cols && iy >=0 && iy < tho->rows);
			float* theta = tho->data.fl + ccv_max(iy - wz, 0) * tho->cols;
			float* magnitude = mdo->data.fl + ccv_max(iy - wz, 0) * mdo->cols;
			float ca = cos(kp->regular.angle);
			float sa = sin(kp->regular.angle);
			float sigmaw = 2.0;
			/* sidenote: NBP = 4, NBO = 8 */
			for (y = ccv_max(iy - wz, 0); y <= ccv_min(iy + wz, tho->rows - 1); y++)
			{
				for (x = ccv_max(ix - wz, 0); x <= ccv_min(ix + wz, tho->cols - 1); x++)
				{
					float nx = (ca * (x - dx) + sa * (y - dy)) / SBP;
					float ny = (-sa * (x - dx) + ca * (y - dy)) / SBP;
					float nt = 8.0 * _ccv_mod_2pi(theta[x] * CCV_PI / 180.0 - kp->regular.angle) / (2.0 * CCV_PI);
					float weight = _ccv_expn((nx * nx + ny * ny) / (2.0 * sigmaw * sigmaw));
					int binx = _ccv_floor(nx - 0.5);
					int biny = _ccv_floor(ny - 0.5);
					int bint = _ccv_floor(nt);
					float rbinx = nx - (binx + 0.5);
					float rbiny = ny - (biny + 0.5);
					float rbint = nt - bint;
					int dbinx, dbiny, dbint;
					/* Distribute the current sample into the 8 adjacent bins*/
					for(dbinx = 0; dbinx < 2; dbinx++)
						for(dbiny = 0; dbiny < 2; dbiny++)
							for(dbint = 0; dbint < 2; dbint++)
								if (binx + dbinx >= -2 && binx + dbinx < 2 && biny + dbiny >= -2 && biny + dbiny < 2)
									fdesc[(2 + biny + dbiny) * 32 + (2 + binx + dbinx) * 8 + (bint + dbint) % 8] += weight * magnitude[x] * fabs(1 - dbinx - rbinx) * fabs(1 - dbiny - rbiny) * fabs(1 - dbint - rbint);
				}
				theta += tho->cols;
				magnitude += mdo->cols;
			}
			ccv_dense_matrix_t tm = ccv_dense_matrix(1, 128, CCV_32F | CCV_C1, fdesc, 0);
			ccv_dense_matrix_t* tmp = &tm;
 			double norm = ccv_normalize(&tm, (ccv_matrix_t**)&tmp, 0, CCV_L2_NORM);
			int num = (ccv_min(iy + wz, tho->rows - 1) - ccv_max(iy - wz, 0) + 1) * (ccv_min(ix + wz, tho->cols - 1) - ccv_max(ix - wz, 0) + 1);
			if (params.norm_threshold && norm < params.norm_threshold * num)
			{
				for (j = 0; j < 128; j++)
					fdesc[j] = 0;
			} else {
				for (j = 0; j < 128; j++)
					if (fdesc[j] > 0.2)
						fdesc[j] = 0.2;
				ccv_normalize(&tm, (ccv_matrix_t**)&tmp, 0, CCV_L2_NORM);
			}
			fdesc += 128;
		}
	}
	for (i = (params.up2x ? -(params.nlevels - 1) : 0); i < (params.nlevels - 1) * params.noctaves; i++)
		ccv_matrix_free(dog[i]);
	for (i = (params.up2x ? -(params.nlevels - 3) : 0); i < (params.nlevels - 3) * params.noctaves; i++)
	{
		ccv_matrix_free(th[i]);
		ccv_matrix_free(md[i]);
	}
}

/* End of ccv/lib/ccv_sift.c */
#line 1 "ccv/lib/ccv_sparse_coding.c"
/* ccv.h already included */


void ccv_sparse_coding(ccv_matrix_t* x, int k, ccv_matrix_t** A, int typeA, ccv_matrix_t** y, int typey)
{
}

/* End of ccv/lib/ccv_sparse_coding.c */
#line 1 "ccv/lib/ccv_swt.c"
/* ccv.h already included */


static inline int _ccv_median(int* buf, int low, int high)
{
	int middle, ll, hh, w;
	int median = (low + high) / 2;
	for (;;)
	{
		if (high <= low)
			return buf[median];
		if (high == low + 1)
		{
			if (buf[low] > buf[high])
				CCV_SWAP(buf[low], buf[high], w);
			return buf[median];
		}
		middle = (low + high) / 2;
		if (buf[middle] > buf[high])
			CCV_SWAP(buf[middle], buf[high], w);
		if (buf[low] > buf[high])
			CCV_SWAP(buf[low], buf[high], w);
		if (buf[middle] > buf[low])
			CCV_SWAP(buf[middle], buf[low], w);
		CCV_SWAP(buf[middle], buf[low + 1], w);
		ll = low + 1;
		hh = high;
		for (;;)
		{
			do ll++; while (buf[low] > buf[ll]);
			do hh--; while (buf[hh] > buf[low]);
			if (hh < ll)
				break;
			CCV_SWAP(buf[ll], buf[hh], w);
		}
		CCV_SWAP(buf[low], buf[hh], w);
		if (hh <= median)
			low = ll;
		else if (hh >= median)
			high = hh - 1;
	}
}

/* ccv_swt is only the method to generate stroke width map */
void ccv_swt(ccv_dense_matrix_t* a, ccv_dense_matrix_t** b, int type, ccv_swt_param_t params)
{
	assert(a->type & CCV_C1);
	char identifier[64];
	memset(identifier, 0, 64);
	snprintf(identifier, 64, "ccv_swt(%d,%d,%lf,%lf)", params.direction, params.size, params.low_thresh, params.high_thresh);
	uint64_t sig = (a->sig == 0) ? 0 : ccv_matrix_generate_signature(identifier, 64, a->sig, 0);
	type = (type == 0) ? CCV_32S | CCV_C1 : CCV_GET_DATA_TYPE(type) | CCV_C1;
	ccv_dense_matrix_t* db = *b = ccv_dense_matrix_renew(*b, a->rows, a->cols, CCV_C1 | CCV_ALL_DATA_TYPE, type, sig);
	ccv_cache_return(db, );
	ccv_dense_matrix_t* c = 0;
	ccv_canny(a, &c, 0, params.size, params.low_thresh, params.high_thresh);
	ccv_dense_matrix_t* dx = 0;
	ccv_sobel(a, &dx, 0, params.size, 0);
	ccv_dense_matrix_t* dy = 0;
	ccv_sobel(a, &dy, 0, 0, params.size);
	int i, j, k, w;
	int* buf = (int*)alloca(sizeof(int) * ccv_max(a->cols, a->rows));
	unsigned char* b_ptr = db->data.ptr;
	unsigned char* c_ptr = c->data.ptr;
	unsigned char* dx_ptr = dx->data.ptr;
	unsigned char* dy_ptr = dy->data.ptr;
	ccv_zero(db);
	int dx5[] = {-1, 0, 1, 0, 0};
	int dy5[] = {0, 0, 0, -1, 1};
	int dx9[] = {-1, 0, 1, -1, 0, 1, -1, 0, 1};
	int dy9[] = {0, 0, 0, -1, -1, -1, 1, 1, 1};
	int adx, ady, sx, sy, err, e2, x0, x1, y0, y1, kx, ky;
#define ray_reset() \
	err = adx - ady; e2 = 0; \
	x0 = j; y0 = i;
#define ray_increment() \
	e2 = 2 * err; \
	if (e2 > -ady) \
	{ \
		err -= ady; \
		x0 += sx; \
	} \
	if (e2 < adx) \
	{ \
		err += adx; \
		y0 += sy; \
	}
	int rdx, rdy, flag;
#define ray_emit(xx, xy, yx, yy, _for_get_d, _for_set_b, _for_get_b) \
	rdx = _for_get_d(dx_ptr, j, 0) * (xx) + _for_get_d(dy_ptr, j, 0) * (xy); \
	rdy = _for_get_d(dx_ptr, j, 0) * (yx) + _for_get_d(dy_ptr, j, 0) * (yy); \
	adx = abs(rdx); \
	ady = abs(rdy); \
	sx = rdx > 0 ? params.direction : -params.direction; \
	sy = rdy > 0 ? params.direction : -params.direction; \
	/* Bresenham's line algorithm */ \
	ray_reset(); \
	flag = 0; \
	kx = x0; \
	ky = y0; \
	for (w = 0; w < 70; w++) \
	{ \
		ray_increment(); \
		if (x0 >= a->cols - 1 || x0 < 1 || y0 >= a->rows - 1 || y0 < 1) \
			break; \
		if (abs(i - y0) < 2 && abs(j - x0) < 2) \
		{ \
			if (c_ptr[x0 + (y0 - i) * c->step]) \
			{ \
				kx = x0; \
				ky = y0; \
				flag = 1; \
				break; \
			} \
		} else { /* ideally, I can encounter another edge directly, but in practice, we should search in a small region around it */ \
			flag = 0; \
			for (k = 0; k < 5; k++) \
			{ \
				kx = x0 + dx5[k]; \
				ky = y0 + dy5[k]; \
				if (c_ptr[kx + (ky - i) * c->step]) \
				{ \
					flag = 1; \
					break; \
				} \
			} \
			if (flag) \
				break; \
		} \
	} \
	if (flag && kx < a->cols - 1 && kx > 0 && ky < a->rows - 1 && ky > 0) \
	{ \
		/* the opposite angle should be in d_p -/+ PI / 6 (otherwise discard),
		 * a faster computation should be:
		 * Tan(d_q - d_p) = (Tan(d_q) - Tan(d_p)) / (1 + Tan(d_q) * Tan(d_p))
		 * and -1 / sqrt(3) < Tan(d_q - d_p) < 1 / sqrt(3)
		 * also, we needs to check the whole 3x3 neighborhood in a hope that we don't miss one or two of them */ \
		flag = 0; \
		for (k = 0; k < 9; k++) \
		{ \
			int tn = _for_get_d(dy_ptr, j, 0) * _for_get_d(dx_ptr + (ky - i + dy9[k]) * dx->step, kx + dx9[k], 0) - \
					 _for_get_d(dx_ptr, j, 0) * _for_get_d(dy_ptr + (ky - i + dy9[k]) * dy->step, kx + dx9[k], 0); \
			int td = _for_get_d(dx_ptr, j, 0) * _for_get_d(dx_ptr + (ky - i + dy9[k]) * dx->step, kx + dx9[k], 0) + \
					 _for_get_d(dy_ptr, j, 0) * _for_get_d(dy_ptr + (ky - i + dy9[k]) * dy->step, kx + dx9[k], 0); \
			if (tn * 7 < -td * 4 && tn * 7 > td * 4) \
			{ \
				flag = 1; \
				break; \
			} \
		} \
		if (flag) \
		{ \
			x1 = x0; y1 = y0; \
			int n = 0; \
			ray_reset(); \
			w = (int)(sqrt((x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0)) + 0.5); \
			/* extend the line to be width of 1 */ \
			for (;;) \
			{ \
				if (_for_get_b(b_ptr + (y0 - i) * db->step, x0, 0) == 0 || _for_get_b(b_ptr + (y0 - i) * db->step, x0, 0) > w) \
				{ \
					_for_set_b(b_ptr + (y0 - i) * db->step, x0, w, 0); \
					buf[n++] = w; \
				} else if (_for_get_b(b_ptr + (y0 - i) * db->step, x0, 0) != 0) \
					buf[n++] = _for_get_b(b_ptr + (y0 - i) * db->step, x0, 0); \
				if (x0 == x1 && y0 == y1) \
					break; \
				ray_increment(); \
			} \
			int nw = _ccv_median(buf, 0, n - 1); \
			if (nw != w) \
			{ \
				ray_reset(); \
				for (;;) \
				{ \
					if (_for_get_b(b_ptr + (y0 - i) * db->step, x0, 0) > nw) \
						_for_set_b(b_ptr + (y0 - i) * db->step, x0, nw, 0); \
					if (x0 == x1 && y0 == y1) \
						break; \
					ray_increment(); \
				} \
			} \
		} \
	}
#define for_block(_for_get_d, _for_set_b, _for_get_b) \
	for (i = 0; i < a->rows; i++) \
	{ \
		for (j = 0; j < a->cols; j++) \
			if (c_ptr[j]) \
			{ \
				ray_emit(1, 0, 0, 1, _for_get_d, _for_set_b, _for_get_b); \
				ray_emit(1, -1, 1, 1, _for_get_d, _for_set_b, _for_get_b); \
				ray_emit(1, 1, -1, 1, _for_get_d, _for_set_b, _for_get_b); \
			} \
		b_ptr += db->step; \
		c_ptr += c->step; \
		dx_ptr += dx->step; \
		dy_ptr += dy->step; \
	}
	ccv_matrix_getter(dx->type, ccv_matrix_setter_getter, db->type, for_block);
#undef for_block
#undef ray_emit
#undef ray_reset
#undef ray_increment
	ccv_matrix_free(c);
	ccv_matrix_free(dx);
	ccv_matrix_free(dy);
}

ccv_array_t* _ccv_swt_connected_component(ccv_dense_matrix_t* a, int ratio)
{
	int i, j, k;
	int* a_ptr = a->data.i;
	int dx8[] = {-1, 1, -1, 0, 1, -1, 0, 1};
	int dy8[] = {0, 0, -1, -1, -1, 1, 1, 1};
	int* marker = (int*)ccmalloc(sizeof(int) * a->rows * a->cols);
	memset(marker, 0, sizeof(int) * a->rows * a->cols);
	int* m_ptr = marker;
	ccv_point_t* buffer = (ccv_point_t*)ccmalloc(sizeof(ccv_point_t) * a->rows * a->cols);
	ccv_array_t* contours = ccv_array_new(5, sizeof(ccv_contour_t*));
	for (i = 0; i < a->rows; i++)
	{
		for (j = 0; j < a->cols; j++)
			if (a_ptr[j] != 0 && !m_ptr[j])
			{
				m_ptr[j] = 1;
				ccv_contour_t* contour = ccv_contour_new(1);
				ccv_point_t* closed = buffer;
				closed->x = j;
				closed->y = i;
				ccv_point_t* open = buffer + 1;
				for (; closed < open; closed++)
				{
					ccv_contour_push(contour, *closed);
					int w = a_ptr[closed->x + (closed->y - i) * a->cols];
					for (k = 0; k < 8; k++)
					{
						int nx = closed->x + dx8[k];
						int ny = closed->y + dy8[k];
						if (nx >= 0 && nx < a->cols && ny >= 0 && ny < a->rows &&
							a_ptr[nx + (ny - i) * a->cols] != 0 &&
							!m_ptr[nx + (ny - i) * a->cols] &&
							(a_ptr[nx + (ny - i) * a->cols] <= ratio * w && a_ptr[nx + (ny - i) * a->cols] * ratio >= w))
						{
							m_ptr[nx + (ny - i) * a->cols] = 1;
							open->x = nx;
							open->y = ny;
							open++;
						}
					}
				}
				ccv_array_push(contours, &contour);
			}
		a_ptr += a->cols;
		m_ptr += a->cols;
	}
	ccfree(marker);
	ccfree(buffer);
	return contours;
}

typedef struct {
	ccv_rect_t rect;
	ccv_point_t center;
	int thickness;
	int intensity;
	double variance;
	double mean;
	ccv_contour_t* contour;
} ccv_letter_t;

static ccv_array_t* _ccv_swt_connected_letters(ccv_dense_matrix_t* a, ccv_dense_matrix_t* swt, ccv_swt_param_t params)
{
	ccv_array_t* contours = _ccv_swt_connected_component(swt, 3);
	ccv_array_t* letters = ccv_array_new(5, sizeof(ccv_letter_t));
	int i, j, x, y;
	int* buffer = (int*)ccmalloc(sizeof(int) * swt->rows * swt->cols);
	double aspect_ratio_inv = 1.0 / params.aspect_ratio;
	for (i = 0; i < contours->rnum; i++)
	{
		ccv_contour_t* contour = *(ccv_contour_t**)ccv_array_get(contours, i);
		if (contour->rect.height > params.max_height || contour->rect.height < params.min_height)
		{
			ccv_contour_free(contour);
			continue;
		}
		double ratio = (double)contour->rect.width / (double)contour->rect.height;
		if (ratio < aspect_ratio_inv || ratio > params.aspect_ratio)
		{
			ccv_contour_free(contour);
			continue;
		}
		double xc = (double)contour->m10 / contour->size;
		double yc = (double)contour->m01 / contour->size;
		double af = (double)contour->m20 / contour->size - xc * xc;
		double bf = 2 * ((double)contour->m11 / contour->size - xc * yc);
		double cf = (double)contour->m02 / contour->size - yc * yc;
		double delta = sqrt(bf * bf + (af - cf) * (af - cf));
		ratio = sqrt((af + cf + delta) / (af + cf - delta));
		if (ratio < aspect_ratio_inv || ratio > params.aspect_ratio)
		{
			ccv_contour_free(contour);
			continue;
		}
		double mean = 0;
		for (j = 0; j < contour->size; j++)
		{
			ccv_point_t* point = (ccv_point_t*)ccv_array_get(contour->set, j);
			mean += buffer[j] = swt->data.i[point->x + point->y * swt->cols];
		}
		mean = mean / contour->size;
		double variance = 0;
		for (j = 0; j < contour->size; j++)
			variance += (mean - buffer[j]) * (mean - buffer[j]);
		variance = variance / contour->size;
		ccv_letter_t letter;
		letter.variance = variance;
		letter.mean = mean;
		letter.thickness = _ccv_median(buffer, 0, contour->size - 1);
		letter.rect = contour->rect;
		letter.center.x = letter.rect.x + letter.rect.width / 2;
		letter.center.y = letter.rect.y + letter.rect.height / 2;
		letter.intensity = 0;
		letter.contour = contour;
		ccv_array_push(letters, &letter);
	}
	ccv_array_free(contours);
	memset(buffer, 0, sizeof(int) * swt->rows * swt->cols);
	ccv_array_t* new_letters = ccv_array_new(5, sizeof(ccv_letter_t));
	for (i = 0; i < letters->rnum; i++)
	{
		ccv_letter_t* letter = (ccv_letter_t*)ccv_array_get(letters, i);
		for (j = 0; j < letter->contour->size; j++)
		{
			ccv_point_t* point = (ccv_point_t*)ccv_array_get(letter->contour->set, j);
			buffer[point->x + point->y * swt->cols] = i + 1;
		}
	}
	for (i = 0; i < letters->rnum; i++)
	{
		ccv_letter_t* letter = (ccv_letter_t*)ccv_array_get(letters, i);
		if (sqrt(letter->variance) > letter->mean * params.variance_ratio)
		{
			ccv_contour_free(letter->contour);
			continue;
		}
		int another[] = {0, 0};
		int more = 0;
		for (x = letter->rect.x; x < letter->rect.x + letter->rect.width; x++)
			for (y = letter->rect.y; y < letter->rect.y + letter->rect.height; y++)
				if (buffer[x + swt->cols * y] && buffer[x + swt->cols * y] != i + 1)
				{
					if (another[0])
					{
						if (buffer[x + swt->cols * y] != another[0])
						{
							if (another[1])
							{
								if (buffer[x + swt->cols * y] != another[1])
								{
									more = 1;
									break;
								}
							} else {
								another[1] = buffer[x + swt->cols * y];
							}
						}
					} else {
						another[0] = buffer[x + swt->cols * y];
					}
				}
		if (more)
		{
			ccv_contour_free(letter->contour);
			continue;
		}
		for (j = 0; j < letter->contour->set->rnum; j++)
		{
			ccv_point_t* point = (ccv_point_t*)ccv_array_get(letter->contour->set, j);
			letter->intensity += a->data.ptr[point->x + point->y * a->step];
		}
		letter->intensity /= letter->contour->size;
		ccv_contour_free(letter->contour);
		letter->contour = 0;
		ccv_array_push(new_letters, letter);
	}
	ccv_array_free(letters);
	ccfree(buffer);
	return new_letters;
}

typedef struct {
	ccv_letter_t* left;
	ccv_letter_t* right;
	int dx;
	int dy;
} ccv_letter_pair_t;

typedef struct {
	ccv_rect_t rect;
	int neighbors;
	ccv_letter_t** letters;
} ccv_textline_t;

static int _ccv_in_textline(const void* a, const void* b, void* data)
{
	ccv_letter_pair_t* pair1 = (ccv_letter_pair_t*)a;
	ccv_letter_pair_t* pair2 = (ccv_letter_pair_t*)b;
	if (pair1->left == pair2->left || pair1->right == pair2->right)
	{
		int tn = pair1->dy * pair2->dx - pair1->dx * pair2->dy;
		int td = pair1->dx * pair2->dx + pair1->dy * pair2->dy;
		// share the same end, opposite direction
		if (tn * 7 < -td * 4 && tn * 7 > td * 4)
			return 1;
	} else if (pair1->left == pair2->right || pair1->right == pair2->left) {
		int tn = pair1->dy * pair2->dx - pair1->dx * pair2->dy;
		int td = pair1->dx * pair2->dx + pair1->dy * pair2->dy;
		// share the other end, same direction
		if (tn * 7 < td * 4 && tn * 7 > -td * 4)
			return 1;
	}
	return 0;
}

static void _ccv_swt_add_letter(ccv_textline_t* textline, ccv_letter_t* letter)
{
	if (textline->neighbors == 0)
	{
		textline->rect = letter->rect;
		textline->neighbors = 1;
		textline->letters = (ccv_letter_t**)ccmalloc(sizeof(ccv_letter_t*) * textline->neighbors);
		textline->letters[0] = letter;
	} else {
		int i, flag = 0;
		for (i = 0; i < textline->neighbors; i++)
			if (textline->letters[i] == letter)
			{
				flag = 1;
				break;
			}
		if (flag)
			return;
		if (letter->rect.x < textline->rect.x)
		{
			textline->rect.width += textline->rect.x - letter->rect.x;
			textline->rect.x = letter->rect.x;
		}
		if (letter->rect.x + letter->rect.width > textline->rect.x + textline->rect.width)
			textline->rect.width = letter->rect.x + letter->rect.width - textline->rect.x;
		if (letter->rect.y < textline->rect.y)
		{
			textline->rect.height += textline->rect.y - letter->rect.y;
			textline->rect.y = letter->rect.y;
		}
		if (letter->rect.y + letter->rect.height > textline->rect.y + textline->rect.height)
			textline->rect.height = letter->rect.y + letter->rect.height - textline->rect.y;
		textline->neighbors++;
		textline->letters = (ccv_letter_t**)ccrealloc(textline->letters, sizeof(ccv_letter_t*) * textline->neighbors);
		textline->letters[textline->neighbors - 1] = letter;
	}
}

static ccv_array_t* _ccv_swt_merge_textline(ccv_array_t* letters, ccv_swt_param_t params)
{
	int i, j;
	ccv_array_t* pairs = ccv_array_new(letters->rnum * letters->rnum, sizeof(ccv_letter_pair_t));
	double thickness_ratio_inv = 1.0 / params.thickness_ratio;
	double height_ratio_inv = 1.0 / params.height_ratio;
	for (i = 0; i < letters->rnum - 1; i++)
	{
		ccv_letter_t* li = (ccv_letter_t*)ccv_array_get(letters, i);
		for (j = i + 1; j < letters->rnum; j++)
		{
			ccv_letter_t* lj = (ccv_letter_t*)ccv_array_get(letters, j);
			double ratio = (double)li->thickness / lj->thickness;
			if (ratio > params.thickness_ratio || ratio < thickness_ratio_inv)
				continue;
			ratio = (double)li->rect.height / lj->rect.height;
			if (ratio > params.height_ratio || ratio < height_ratio_inv)
				continue;
			if (abs(li->intensity - lj->intensity) > params.intensity_thresh)
				continue;
			int dx = li->rect.x - lj->rect.x + (li->rect.width - lj->rect.width) / 2;
			int dy = li->rect.y - lj->rect.y + (li->rect.height - lj->rect.height) / 2;
			if (abs(dx) > params.distance_ratio * ccv_max(li->rect.width, lj->rect.width))
				continue;
			int oy = ccv_min(li->rect.y + li->rect.height, lj->rect.y + lj->rect.height) - ccv_max(li->rect.y, lj->rect.y);
			if (oy * params.intersect_ratio < ccv_min(li->rect.height, lj->rect.height))
				continue;
			ccv_letter_pair_t pair = { .left = li, .right = lj, .dx = dx, .dy = dy };
			ccv_array_push(pairs, &pair);
		}
	}
	ccv_array_t* idx = 0;
	int nchains = ccv_array_group(pairs, &idx, _ccv_in_textline, 0);
	ccv_textline_t* chain = (ccv_textline_t*)ccmalloc(nchains * sizeof(ccv_textline_t));
	for (i = 0; i < nchains; i++)
		chain[i].neighbors = 0;
	for (i = 0; i < pairs->rnum; i++)
	{
		j = *(int*)ccv_array_get(idx, i);
		_ccv_swt_add_letter(chain + j,((ccv_letter_pair_t*)ccv_array_get(pairs, i))->left);
		_ccv_swt_add_letter(chain + j, ((ccv_letter_pair_t*)ccv_array_get(pairs, i))->right);
	}
	ccv_array_free(pairs);
	ccv_array_t* regions = ccv_array_new(5, sizeof(ccv_textline_t));
	for (i = 0; i < nchains; i++)
		if (chain[i].neighbors >= params.letter_thresh && chain[i].rect.width > chain[i].rect.height * params.elongate_ratio)
			ccv_array_push(regions, chain + i);
		else if (chain[i].neighbors > 0)
			ccfree(chain[i].letters);
	ccfree(chain);
	return regions;
}

#define less_than(a, b, aux) ((a)->center.x < (b)->center.x)
CCV_IMPLEMENT_QSORT(_ccv_sort_letters, ccv_letter_t*, less_than)
#undef less_than

static ccv_array_t* _ccv_swt_break_words(ccv_array_t* textline, ccv_swt_param_t params)
{
	int i, j, n = 0;
	for (i = 0; i < textline->rnum; i++)
	{
		ccv_textline_t* t = (ccv_textline_t*)ccv_array_get(textline, i);
		if (t->neighbors > n + 1)
			n = t->neighbors - 1;
	}
	int* buffer = (int*)alloca(n * sizeof(int));
	ccv_array_t* words = ccv_array_new(textline->rnum, sizeof(ccv_rect_t));
	for (i = 0; i < textline->rnum; i++)
	{
		ccv_textline_t* t = (ccv_textline_t*)ccv_array_get(textline, i);
		_ccv_sort_letters(t->letters, t->neighbors, 0);
		int range = 0;
		double mean = 0;
		for (j = 0; j < t->neighbors - 1; j++)
		{
			buffer[j] = t->letters[j + 1]->center.x - t->letters[j]->center.x;
			if (buffer[j] >= range)
				range = buffer[j] + 1;
			mean += buffer[j];
		}
		ccv_dense_matrix_t otsu = ccv_dense_matrix(1, t->neighbors - 1, CCV_32S | CCV_C1, buffer, 0);
		double var;
		int threshold = ccv_otsu(&otsu, &var, range);
		mean = mean / (t->neighbors - 1);
		if (var > mean * params.breakdown_ratio)
		{
			ccv_textline_t nt = { .neighbors = 0 };
			_ccv_swt_add_letter(&nt, t->letters[0]);
			for (j = 0; j < t->neighbors - 1; j++)
			{
				if (buffer[j] > threshold)
				{
					if (nt.neighbors >= params.letter_thresh && nt.rect.width > nt.rect.height * params.elongate_ratio)
						ccv_array_push(words, &nt.rect);
					nt.neighbors = 0;
				}
				_ccv_swt_add_letter(&nt, t->letters[j + 1]);
			}
			if (nt.neighbors >= params.letter_thresh && nt.rect.width > nt.rect.height * params.elongate_ratio)
				ccv_array_push(words, &nt.rect);
		} else {
			ccv_array_push(words, &(t->rect));
		}
	}
	return words;
}

static int _ccv_is_same_textline(const void* a, const void* b, void* data)
{
	ccv_textline_t* t1 = (ccv_textline_t*)a;
	ccv_textline_t* t2 = (ccv_textline_t*)b;
	int width = ccv_min(t1->rect.x + t1->rect.width, t2->rect.x + t2->rect.width) - ccv_max(t1->rect.x, t2->rect.x);
	int height = ccv_min(t1->rect.y + t1->rect.height, t2->rect.y + t2->rect.height) - ccv_max(t1->rect.y, t2->rect.y);
	/* overlapped 80% */
	return (width > 0 && height > 0 && width * height * 10 > ccv_min(t1->rect.width * t1->rect.height, t2->rect.width * t2->rect.height) * 8);
}

ccv_array_t* ccv_swt_detect_words(ccv_dense_matrix_t* a, ccv_swt_param_t params)
{
	ccv_dense_matrix_t* swt = 0;
	params.direction = -1;
	ccv_swt(a, &swt, 0, params);
	/* perform connected component analysis */
	ccv_array_t* lettersB = _ccv_swt_connected_letters(a, swt, params);
	ccv_matrix_free(swt);
	ccv_array_t* textline = _ccv_swt_merge_textline(lettersB, params);
	swt = 0;
	params.direction = 1;
	ccv_swt(a, &swt, 0, params);
	ccv_array_t* lettersF = _ccv_swt_connected_letters(a, swt, params);
	ccv_matrix_free(swt);
	ccv_array_t* textline2 = _ccv_swt_merge_textline(lettersF, params);
	int i;
	for (i = 0; i < textline2->rnum; i++)
		ccv_array_push(textline, ccv_array_get(textline2, i));
	ccv_array_free(textline2);
	ccv_array_t* idx = 0;
	int ntl = ccv_array_group(textline, &idx, _ccv_is_same_textline, 0);
	ccv_array_t* words;
	if (params.breakdown)
	{
		textline2 = ccv_array_new(ntl, sizeof(ccv_textline_t));
		ccv_array_zero(textline2);
		textline2->rnum = ntl;
		for (i = 0; i < textline->rnum; i++)
		{
			ccv_textline_t* r = (ccv_textline_t*)ccv_array_get(textline, i);
			int k = *(int*)ccv_array_get(idx, i);
			ccv_textline_t* r2 = (ccv_textline_t*)ccv_array_get(textline2, k);
			if (r2->rect.width < r->rect.width)
			{
				if (r2->letters != 0)
					ccfree(r2->letters);
				*r2 = *r;
			}
		}
		ccv_array_free(idx);
		ccv_array_free(textline);
		words = _ccv_swt_break_words(textline2, params);
		for (i = 0; i < textline2->rnum; i++)
			ccfree(((ccv_textline_t*)ccv_array_get(textline2, i))->letters);
		ccv_array_free(textline2);
		ccv_array_free(lettersB);
		ccv_array_free(lettersF);
	} else {
		ccv_array_free(lettersB);
		ccv_array_free(lettersF);
		words = ccv_array_new(ntl, sizeof(ccv_rect_t));
		ccv_array_zero(words);
		words->rnum = ntl;
		for (i = 0; i < textline->rnum; i++)
		{
			ccv_textline_t* r = (ccv_textline_t*)ccv_array_get(textline, i);
			ccfree(r->letters);
			int k = *(int*)ccv_array_get(idx, i);
			ccv_rect_t* r2 = (ccv_rect_t*)ccv_array_get(words, k);
			if (r2->width * r2->height < r->rect.width * r->rect.height)
				*r2 = r->rect;
		}
		ccv_array_free(idx);
		ccv_array_free(textline);
	}
	return words;
}

/* End of ccv/lib/ccv_swt.c */
#line 1 "ccv/lib/ccv_util.c"
/* ccv.h already included */


int _ccv_get_sparse_prime[] = { 53, 97, 193, 389, 769, 1543, 3079, 6151, 12289, 24593, 49157, 98317, 196613, 393241, 786433, 1572869 };

ccv_dense_matrix_t* ccv_get_dense_matrix(ccv_matrix_t* mat)
{
	int type = *(int*)mat;
	if (type & CCV_MATRIX_DENSE)
		return (ccv_dense_matrix_t*)mat;
	return 0;
}

ccv_sparse_matrix_t* ccv_get_sparse_matrix(ccv_matrix_t* mat)
{
	int type = *(int*)mat;
	if (type & CCV_MATRIX_SPARSE)
		return (ccv_sparse_matrix_t*)mat;
	return 0;
}

void ccv_shift(ccv_matrix_t* a, ccv_matrix_t** b, int type, int lr, int rr)
{
	ccv_dense_matrix_t* da = ccv_get_dense_matrix(a);
	char identifier[64];
	memset(identifier, 0, 64);
	snprintf(identifier, 64, "ccv_shift(%d,%d)", lr, rr);
	uint64_t sig = ccv_matrix_generate_signature(identifier, 64, da->sig, 0);
	type = (type == 0) ? CCV_GET_DATA_TYPE(da->type) | CCV_GET_CHANNEL(da->type) : CCV_GET_DATA_TYPE(type) | CCV_GET_CHANNEL(da->type);
	ccv_dense_matrix_t* db = *b = ccv_dense_matrix_renew(*b, da->rows, da->cols, CCV_ALL_DATA_TYPE | CCV_GET_CHANNEL(da->type), type, sig); 
	ccv_cache_return(db, );
	int i, j, ch = CCV_GET_CHANNEL(da->type);
	unsigned char* aptr = da->data.ptr;
	unsigned char* bptr = db->data.ptr;
#define for_block(_for_get, _for_set) \
	for (i = 0; i < da->rows; i++) \
	{ \
		for (j = 0; j < da->cols * ch; j++) \
		{ \
			_for_set(bptr, j, _for_get(aptr, j, lr), rr); \
		} \
		aptr += da->step; \
		bptr += db->step; \
	}
	ccv_matrix_getter(da->type, ccv_matrix_setter, db->type, for_block);
#undef for_block
}

ccv_dense_vector_t* ccv_get_sparse_matrix_vector(ccv_sparse_matrix_t* mat, int index)
{
	if (mat->vector[(index * 33) % CCV_GET_SPARSE_PRIME(mat->prime)].index != -1)
	{
		ccv_dense_vector_t* vector = &mat->vector[(index * 33) % CCV_GET_SPARSE_PRIME(mat->prime)];
		while (vector != 0 && vector->index != index)
			vector = vector->next;
		return vector;
	}
	return 0;
}

ccv_matrix_cell_t ccv_get_sparse_matrix_cell(ccv_sparse_matrix_t* mat, int row, int col)
{
	ccv_dense_vector_t* vector = ccv_get_sparse_matrix_vector(mat, (mat->major == CCV_SPARSE_COL_MAJOR) ? col : row);
	ccv_matrix_cell_t cell;
	cell.ptr = 0;
	if (vector != 0 && vector->length > 0)
	{
		int cell_width = CCV_GET_DATA_TYPE_SIZE(mat->type) * CCV_GET_CHANNEL(mat->type);
		int vidx = (mat->major == CCV_SPARSE_COL_MAJOR) ? row : col;
		if (mat->type & CCV_DENSE_VECTOR)
		{
			cell.ptr = vector->data.ptr + cell_width * vidx;
		} else {
			int h = (vidx * 33) % vector->length, i = 0;
			while (vector->indice[(h + i * i) % vector->length] != vidx && vector->indice[(h + i * i) % vector->length] != -1)
				i++;
			i = (h + i * i) % vector->length;
			if (vector->indice[i] != -1)
				cell.ptr = vector->data.ptr + i * cell_width;
		}
	}
	return cell;
}

static void _ccv_dense_vector_expand(ccv_sparse_matrix_t* mat, ccv_dense_vector_t* vector)
{
	if (vector->prime == -1)
		return;
	vector->prime++;
	int new_length = CCV_GET_SPARSE_PRIME(vector->prime);
	int cell_width = CCV_GET_DATA_TYPE_SIZE(mat->type) * CCV_GET_CHANNEL(mat->type);
	int new_step = (new_length * cell_width + 3) & -4;
	ccv_matrix_cell_t new_data;
	new_data.ptr = (unsigned char*)ccmalloc(new_step + sizeof(int) * new_length);
	int* new_indice = (int*)(new_data.ptr + new_step);
	int i;
	for (i = 0; i < new_length; i++)
		new_indice[i] = -1;
	for (i = 0; i < vector->length; i++)
		if (vector->indice[i] != -1)
		{
			int index = vector->indice[i];
			int h = (index * 33) % new_length, j = 0;
			while (new_indice[(h + j * j) % new_length] != index && new_indice[(h + j * j) % new_length] != -1)
				j++;
			j = (h + j * j) % new_length;
			new_indice[j] = index;
			memcpy(new_data.ptr + j * cell_width, vector->data.ptr + i * cell_width, cell_width);
		}
	vector->length = new_length;
	ccfree(vector->data.ptr);
	vector->data = new_data;
	vector->indice = new_indice;
}

static void _ccv_sparse_matrix_expand(ccv_sparse_matrix_t* mat)
{
	int length = CCV_GET_SPARSE_PRIME(mat->prime);
	mat->prime++;
	int new_length = CCV_GET_SPARSE_PRIME(mat->prime);
	ccv_dense_vector_t* new_vector = (ccv_dense_vector_t*)ccmalloc(new_length * sizeof(ccv_dense_vector_t));
	int i;
	for (i = 0; i < new_length; i++)
	{
		new_vector[i].index = -1;
		new_vector[i].length = 0;
		new_vector[i].next = 0;
	}
	for (i = 0; i < length; i++)
		if (mat->vector[i].index != -1)
		{
			int h = (mat->vector[i].index * 33) % new_length;
			if (new_vector[h].length == 0)
			{
				memcpy(new_vector + h, mat->vector + i, sizeof(ccv_dense_vector_t));
				new_vector[h].next = 0;
			} else {
				ccv_dense_vector_t* t = (ccv_dense_vector_t*)ccmalloc(sizeof(ccv_dense_vector_t));
				memcpy(t, mat->vector + i, sizeof(ccv_dense_vector_t));
				t->next = new_vector[h].next;
				new_vector[h].next = t;
			}
			ccv_dense_vector_t* iter = mat->vector[i].next;
			while (iter != 0)
			{
				ccv_dense_vector_t* iter_next = iter->next;
				h = (iter->index * 33) % new_length;
				if (new_vector[h].length == 0)
				{
					memcpy(new_vector + h, iter, sizeof(ccv_dense_vector_t));
					new_vector[h].next = 0;
					ccfree(iter);
				} else {
					iter->next = new_vector[h].next;
					new_vector[h].next = iter;
				}
				iter = iter_next;
			}
		}
	ccfree(mat->vector);
	mat->vector = new_vector;
}

void ccv_set_sparse_matrix_cell(ccv_sparse_matrix_t* mat, int row, int col, void* data)
{
	int i;
	int index = (mat->major == CCV_SPARSE_COL_MAJOR) ? col : row;
	int vidx = (mat->major == CCV_SPARSE_COL_MAJOR) ? row : col;
	int length = CCV_GET_SPARSE_PRIME(mat->prime);
	ccv_dense_vector_t* vector = ccv_get_sparse_matrix_vector(mat, index);
	if (vector == 0)
	{
		mat->load_factor++;
		if (mat->load_factor * 4 > CCV_GET_SPARSE_PRIME(mat->prime) * 3)
		{
			_ccv_sparse_matrix_expand(mat);
			length = CCV_GET_SPARSE_PRIME(mat->prime);
		}
		vector = &mat->vector[(index * 33) % length];
		if (vector->index != -1)
		{
			vector = (ccv_dense_vector_t*)ccmalloc(sizeof(ccv_dense_vector_t));
			vector->index = -1;
			vector->length = 0;
			vector->next = mat->vector[(index * 33) % length].next;
			mat->vector[(index * 33) % length].next = vector;
		}
	}
	int cell_width = CCV_GET_DATA_TYPE_SIZE(mat->type) * CCV_GET_CHANNEL(mat->type);
	if (mat->type & CCV_DENSE_VECTOR)
	{
		if (vector->index == -1)
		{
			vector->prime = -1;
			vector->length = (mat->major == CCV_SPARSE_COL_MAJOR) ? mat->rows : mat->cols;
			vector->index = index;
			vector->step = (vector->length * cell_width + 3) & -4;
			vector->data.ptr = (unsigned char*)calloc(vector->step, 1);
		}
		if (data != 0)
			memcpy(vector->data.ptr + vidx * cell_width, data, cell_width);
	} else {
		if (vector->index == -1)
		{
			vector->prime = 0;
			vector->load_factor = 0;
			vector->length = CCV_GET_SPARSE_PRIME(vector->prime);
			vector->index = index;
			vector->step  = (vector->length * cell_width + 3) & -4;
			vector->data.ptr = (unsigned char*)ccmalloc(vector->step + sizeof(int) * vector->length);
			vector->indice = (int*)(vector->data.ptr + vector->step);
			for (i = 0; i < vector->length; i++)
				vector->indice[i] = -1;
		}
		vector->load_factor++;
		if (vector->load_factor * 2 > vector->length)
		{
			_ccv_dense_vector_expand(mat, vector);
		}
		i = 0;
		int h = (vidx * 33) % vector->length;
		while (vector->indice[(h + i * i) % vector->length] != vidx && vector->indice[(h + i * i) % vector->length] != -1)
			i++;
		i = (h + i * i) % vector->length;
		vector->indice[i] = vidx;
		if (data != 0)
			memcpy(vector->data.ptr + i * cell_width, data, cell_width);
	}
}

#define _ccv_indice_less_than(i1, i2, aux) ((i1) < (i2))
#define _ccv_swap_indice_and_uchar_data(i1, i2, array, aux, t) {  \
	unsigned char td = (aux)[(int)(&(i1) - (array))];			  \
	(aux)[(int)(&(i1) - (array))] = (aux)[(int)(&(i2) - (array))]; \
	(aux)[(int)(&(i2) - (array))] = td;							\
	CCV_SWAP(i1, i2, t); }
#define _ccv_swap_indice_and_int_data(i1, i2, array, aux, t) {	\
	int td = (aux)[(int)(&(i1) - (array))];						\
	(aux)[(int)(&(i1) - (array))] = (aux)[(int)(&(i2) - (array))]; \
	(aux)[(int)(&(i2) - (array))] = td;							\
	CCV_SWAP(i1, i2, t); }
#define _ccv_swap_indice_and_float_data(i1, i2, array, aux, t) {  \
	float td = (aux)[(int)(&(i1) - (array))];					  \
	(aux)[(int)(&(i1) - (array))] = (aux)[(int)(&(i2) - (array))]; \
	(aux)[(int)(&(i2) - (array))] = td;							\
	CCV_SWAP(i1, i2, t); }
#define _ccv_swap_indice_and_double_data(i1, i2, array, aux, t) { \
	double td = (aux)[(int)(&(i1) - (array))];					 \
	(aux)[(int)(&(i1) - (array))] = (aux)[(int)(&(i2) - (array))]; \
	(aux)[(int)(&(i2) - (array))] = td;							\
	CCV_SWAP(i1, i2, t); }

CCV_IMPLEMENT_QSORT_EX(_ccv_indice_uchar_sort, int, _ccv_indice_less_than, _ccv_swap_indice_and_uchar_data, unsigned char*);
CCV_IMPLEMENT_QSORT_EX(_ccv_indice_int_sort, int, _ccv_indice_less_than, _ccv_swap_indice_and_int_data, int*);
CCV_IMPLEMENT_QSORT_EX(_ccv_indice_float_sort, int, _ccv_indice_less_than, _ccv_swap_indice_and_float_data, float*);
CCV_IMPLEMENT_QSORT_EX(_ccv_indice_double_sort, int, _ccv_indice_less_than, _ccv_swap_indice_and_double_data, double*);

void ccv_compress_sparse_matrix(ccv_sparse_matrix_t* mat, ccv_compressed_sparse_matrix_t** csm)
{
	int i, j;
	int nnz = 0;
	int length = CCV_GET_SPARSE_PRIME(mat->prime);
	for (i = 0; i < length; i++)
	{
		ccv_dense_vector_t* vector = &mat->vector[i];
#define while_block(_, _while_get) \
		while (vector != 0) \
		{ \
			if (vector->index != -1) \
			{ \
				if (mat->type & CCV_DENSE_VECTOR) \
				{ \
					for (j = 0; j < vector->length; j++) \
						if (_while_get(vector->data.ptr, j, 0) != 0) \
							nnz++; \
				} else { \
					nnz += vector->load_factor; \
				} \
			} \
			vector = vector->next; \
		}
		ccv_matrix_getter(mat->type, while_block);
#undef while_block
	}
	ccv_compressed_sparse_matrix_t* cm = *csm = (ccv_compressed_sparse_matrix_t*)ccmalloc(sizeof(ccv_compressed_sparse_matrix_t) + nnz * sizeof(int) + nnz * CCV_GET_DATA_TYPE_SIZE(mat->type) + (((mat->major == CCV_SPARSE_COL_MAJOR) ? mat->cols : mat->rows) + 1) * sizeof(int));
	cm->type = (mat->type & ~CCV_MATRIX_SPARSE & ~CCV_SPARSE_VECTOR & ~CCV_DENSE_VECTOR) | ((mat->major == CCV_SPARSE_COL_MAJOR) ? CCV_MATRIX_CSC : CCV_MATRIX_CSR);
	cm->nnz = nnz;
	cm->rows = mat->rows;
	cm->cols = mat->cols;
	cm->index = (int*)(cm + 1);
	cm->offset = cm->index + nnz;
	cm->data.i = cm->offset + ((mat->major == CCV_SPARSE_COL_MAJOR) ? mat->cols : mat->rows) + 1;
	unsigned char* m_ptr = cm->data.ptr;
	int* idx = cm->index;
	cm->offset[0] = 0;
	for (i = 0; i < ((mat->major == CCV_SPARSE_COL_MAJOR) ? mat->cols : mat->rows); i++)
	{
		ccv_dense_vector_t* vector = ccv_get_sparse_matrix_vector(mat, i);
		if (vector == 0)
			cm->offset[i + 1] = cm->offset[i];
		else {
			if (mat->type & CCV_DENSE_VECTOR)
			{
				int k = 0;
#define for_block(_for_set, _for_get) \
				for (j = 0; j < vector->length; j++) \
					if (_for_get(vector->data.ptr, j, 0) != 0) \
					{ \
						_for_set(m_ptr, k, _for_get(vector->data.ptr, j, 0), 0); \
						idx[k] = j; \
						k++; \
					}
				ccv_matrix_setter(mat->type, ccv_matrix_getter, mat->type, for_block);
#undef for_block
				cm->offset[i + 1] = cm->offset[i] + k;
				idx += k;
				m_ptr += k * CCV_GET_DATA_TYPE_SIZE(mat->type);
			} else {
				int k = 0;
#define for_block(_for_set, _for_get) \
				for (j = 0; j < vector->length; j++) \
					if (vector->indice[j] != -1) \
					{ \
						_for_set(m_ptr, k, _for_get(vector->data.ptr, j, 0), 0); \
						idx[k] = vector->indice[j]; \
						k++; \
					}
				ccv_matrix_setter(mat->type, ccv_matrix_getter, mat->type, for_block);
#undef for_block
				switch (CCV_GET_DATA_TYPE(mat->type))
				{
					case CCV_8U:
						_ccv_indice_uchar_sort(idx, vector->load_factor, (unsigned char*)m_ptr);
						break;
					case CCV_32S:
						_ccv_indice_int_sort(idx, vector->load_factor, (int*)m_ptr);
						break;
					case CCV_32F:
						_ccv_indice_float_sort(idx, vector->load_factor, (float*)m_ptr);
						break;
					case CCV_64F:
						_ccv_indice_double_sort(idx, vector->load_factor, (double*)m_ptr);
						break;
				}
				cm->offset[i + 1] = cm->offset[i] + vector->load_factor;
				idx += vector->load_factor;
				m_ptr += vector->load_factor * CCV_GET_DATA_TYPE_SIZE(mat->type);
			}
		}
	}
}

void ccv_decompress_sparse_matrix(ccv_compressed_sparse_matrix_t* csm, ccv_sparse_matrix_t** smt)
{
	ccv_sparse_matrix_t* mat = *smt = ccv_sparse_matrix_new(csm->rows, csm->cols, csm->type & ~CCV_MATRIX_CSR & ~CCV_MATRIX_CSC, (csm->type & CCV_MATRIX_CSR) ? CCV_SPARSE_ROW_MAJOR : CCV_SPARSE_COL_MAJOR, 0);
	int i, j;
	for (i = 0; i < ((mat->major == CCV_SPARSE_COL_MAJOR) ? mat->cols : mat->rows); i++)
		for (j = csm->offset[i]; j < csm->offset[i + 1]; j++)
			if (mat->major == CCV_SPARSE_COL_MAJOR)
				ccv_set_sparse_matrix_cell(mat, csm->index[j], i, csm->data.ptr + CCV_GET_DATA_TYPE_SIZE(csm->type) * j);
			else
				ccv_set_sparse_matrix_cell(mat, i, csm->index[j], csm->data.ptr + CCV_GET_DATA_TYPE_SIZE(csm->type) * j);
}

int ccv_matrix_eq(ccv_matrix_t* a, ccv_matrix_t* b)
{
	int a_type = *(int*)a;
	int b_type = *(int*)b;
	if ((a_type & CCV_MATRIX_DENSE) && (b_type & CCV_MATRIX_DENSE))
	{
		ccv_dense_matrix_t* da = (ccv_dense_matrix_t*)a;
		ccv_dense_matrix_t* db = (ccv_dense_matrix_t*)b;
		if (CCV_GET_DATA_TYPE(da->type) != CCV_GET_DATA_TYPE(db->type))
			return -1;
		if (CCV_GET_CHANNEL(da->type) != CCV_GET_CHANNEL(db->type))
			return -1;
		if (da->rows != db->rows)
			return -1;
		if (da->cols != db->cols)
			return -1;
		int i, j, ch = CCV_GET_CHANNEL(da->type);
		unsigned char* a_ptr = da->data.ptr;
		unsigned char* b_ptr = db->data.ptr;
#define for_block(_, _for_get) \
		for (i = 0; i < da->rows; i++) \
		{ \
			for (j = 0; j < da->cols * ch; j++) \
			{ \
				if (fabs(_for_get(b_ptr, j, 0) - _for_get(a_ptr, j, 0)) > 1e-6) \
					return -1; \
			} \
			a_ptr += da->step; \
			b_ptr += db->step; \
		}
		ccv_matrix_getter(da->type, for_block);
#undef for_block
	}
	return 0;
}

void ccv_slice(ccv_matrix_t* a, ccv_matrix_t** b, int btype, int y, int x, int rows, int cols)
{
	int type = *(int*)a;
	if (type & CCV_MATRIX_DENSE)
	{
		ccv_dense_matrix_t* da = ccv_get_dense_matrix(a);
		assert(y >= 0 && y + rows <= da->rows && x >= 0 && x + cols <= da->cols);
		char identifier[128];
		memset(identifier, 0, 128);
		snprintf(identifier, 128, "ccv_slice(%d,%d,%d,%d)", y, x, rows, cols);
		uint64_t sig = (da->sig == 0) ? 0 : ccv_matrix_generate_signature(identifier, 128, da->sig, 0);
		btype = (btype == 0) ? CCV_GET_DATA_TYPE(da->type) | CCV_GET_CHANNEL(da->type) : CCV_GET_DATA_TYPE(btype) | CCV_GET_CHANNEL(da->type);
		ccv_dense_matrix_t* db = *b = ccv_dense_matrix_renew(*b, rows, cols, CCV_ALL_DATA_TYPE | CCV_GET_CHANNEL(da->type), btype, sig);
		ccv_cache_return(db, );
		int i, j, ch = CCV_GET_CHANNEL(da->type);
		unsigned char* a_ptr = da->data.ptr + x * ch * CCV_GET_DATA_TYPE_SIZE(da->type) + y * da->step;
		unsigned char* b_ptr = db->data.ptr;
#define for_block(_, _for_set, _for_get) \
		for (i = 0; i < rows; i++) \
		{ \
			for (j = 0; j < cols * ch; j++) \
			{ \
				_for_set(b_ptr, j, _for_get(a_ptr, j, 0), 0); \
			} \
			a_ptr += da->step; \
			b_ptr += db->step; \
		}
		ccv_matrix_setter_getter(da->type, for_block);
#undef for_block
	} else if (type & CCV_MATRIX_SPARSE) {
	}
}

void ccv_move(ccv_matrix_t* a, ccv_matrix_t** b, int btype, int y, int x)
{
	int type = *(int*)a;
	if (type & CCV_MATRIX_DENSE)
	{
		ccv_dense_matrix_t* da = ccv_get_dense_matrix(a);
		char identifier[128];
		memset(identifier, 0, 128);
		snprintf(identifier, 128, "ccv_move(%d,%d)", y, x);
		uint64_t sig = (da->sig == 0) ? 0 : ccv_matrix_generate_signature(identifier, 128, da->sig, 0);
		btype = (btype == 0) ? CCV_GET_DATA_TYPE(da->type) | CCV_GET_CHANNEL(da->type) : CCV_GET_DATA_TYPE(btype) | CCV_GET_CHANNEL(da->type);
		ccv_dense_matrix_t* db = *b = ccv_dense_matrix_renew(*b, da->rows, da->cols, CCV_ALL_DATA_TYPE | CCV_GET_CHANNEL(da->type), btype, sig);
		ccv_cache_return(db, );
		int i, j, ch = CCV_GET_CHANNEL(da->type);
		unsigned char* a_ptr = da->data.ptr + ccv_max(x, 0) * ch * CCV_GET_DATA_TYPE_SIZE(da->type) + ccv_max(y, 0) * da->step;
		unsigned char* b_ptr = db->data.ptr + ccv_max(-x, 0) * ch * CCV_GET_DATA_TYPE_SIZE(db->type) + ccv_max(-y, 0) * db->step;
#define for_block(_, _for_set, _for_get) \
		for (i = abs(y); i < db->rows; i++) \
		{ \
			for (j = abs(x) * ch; j < db->cols * ch; j++) \
			{ \
				_for_set(b_ptr, j, _for_get(a_ptr, j, 0), 0); \
			} \
			a_ptr += da->step; \
			b_ptr += db->step; \
		}
		ccv_matrix_setter_getter(da->type, for_block);
#undef for_block
	} else if (type & CCV_MATRIX_SPARSE) {
	}
}

ccv_array_t* ccv_array_new(int rnum, int rsize)
{
	ccv_array_t* array = (ccv_array_t*)ccmalloc(sizeof(ccv_array_t));
	array->rnum = 0;
	array->rsize = rsize;
	array->size = rnum;
	array->data = ccmalloc(rnum * rsize);
	return array;
}

void ccv_array_push(ccv_array_t* array, void* r)
{
	array->rnum++;
	if (array->rnum > array->size)
	{
		array->size = array->size * 2;
		array->data = ccrealloc(array->data, array->size * array->rsize);
	}
	memcpy(ccv_array_get(array, array->rnum - 1), r, array->rsize);
}

void ccv_array_zero(ccv_array_t* array)
{
	memset(array->data, 0, array->size * array->rsize);
}

void ccv_array_clear(ccv_array_t* array)
{
	array->rnum = 0;
}

void ccv_array_free(ccv_array_t* array)
{
	ccfree(array->data);
	ccfree(array);
}

typedef struct ccv_ptree_node_t
{
	struct ccv_ptree_node_t* parent;
	void* element;
	int rank;
} ccv_ptree_node_t;

/* the code for grouping array is adopted from OpenCV's cvSeqPartition func, it is essentially a find-union algorithm */
int ccv_array_group(ccv_array_t* array, ccv_array_t** index, ccv_array_group_f gfunc, void* data)
{
	int i, j;
	ccv_ptree_node_t* node = (ccv_ptree_node_t*)ccmalloc(array->rnum * sizeof(ccv_ptree_node_t));
	for (i = 0; i < array->rnum; i++)
	{
		node[i].parent = 0;
		node[i].element = ccv_array_get(array, i);
		node[i].rank = 0;
	}
	for (i = 0; i < array->rnum; i++)
	{
		if (!node[i].element)
			continue;
		ccv_ptree_node_t* root = node + i;
		while (root->parent)
			root = root->parent;
		for (j = 0; j < array->rnum; j++)
		{
			if( i != j && node[j].element && gfunc(node[i].element, node[j].element, data))
			{
				ccv_ptree_node_t* root2 = node + j;

				while(root2->parent)
					root2 = root2->parent;

				if(root2 != root)
				{
					if(root->rank > root2->rank)
						root2->parent = root;
					else
					{
						root->parent = root2;
						root2->rank += root->rank == root2->rank;
						root = root2;
					}

					/* compress path from node2 to the root: */
					ccv_ptree_node_t* node2 = node + j;
					while(node2->parent)
					{
						ccv_ptree_node_t* temp = node2;
						node2 = node2->parent;
						temp->parent = root;
					}

					/* compress path from node to the root: */
					node2 = node + i;
					while(node2->parent)
					{
						ccv_ptree_node_t* temp = node2;
						node2 = node2->parent;
						temp->parent = root;
					}
				}
			}
		}
	}
	if (*index == 0)
		*index = ccv_array_new(array->rnum, sizeof(int));
	else
		ccv_array_clear(*index);
	ccv_array_t* idx = *index;

	int class_idx = 0;
	for(i = 0; i < array->rnum; i++)
	{
		j = -1;
		ccv_ptree_node_t* node1 = node + i;
		if(node1->element)
		{
			while(node1->parent)
				node1 = node1->parent;
			if(node1->rank >= 0)
				node1->rank = ~class_idx++;
			j = ~node1->rank;
		}
		ccv_array_push(idx, &j);
	}
	ccfree(node);
	return class_idx;
}

ccv_contour_t* ccv_contour_new(int set)
{
	ccv_contour_t* contour = (ccv_contour_t*)ccmalloc(sizeof(ccv_contour_t));
	contour->rect.x = contour->rect.y =
	contour->rect.width = contour->rect.height = 0;
	contour->size = 0;
	if (set)
		contour->set = ccv_array_new(5, sizeof(ccv_point_t));
	else
		contour->set = 0;
	return contour;
}

void ccv_contour_push(ccv_contour_t* contour, ccv_point_t point)
{
	if (contour->size == 0)
	{
		contour->rect.x = point.x;
		contour->rect.y = point.y;
		contour->rect.width = contour->rect.height = 1;
		contour->m10 = point.x;
		contour->m01 = point.y;
		contour->m11 = point.x * point.y;
		contour->m20 = point.x * point.x;
		contour->m02 = point.y * point.y;
		contour->size = 1;
	} else {
		if (point.x < contour->rect.x)
		{
			contour->rect.width += contour->rect.x - point.x;
			contour->rect.x = point.x;
		} else if (point.x > contour->rect.x + contour->rect.width - 1) {
			contour->rect.width = point.x - contour->rect.x + 1;
		}
		if (point.y < contour->rect.y)
		{
			contour->rect.height += contour->rect.y - point.y;
			contour->rect.y = point.y;
		} else if (point.y > contour->rect.y + contour->rect.height - 1) {
			contour->rect.height = point.y - contour->rect.y + 1;
		}
		contour->m10 += point.x;
		contour->m01 += point.y;
		contour->m11 += point.x * point.y;
		contour->m20 += point.x * point.x;
		contour->m02 += point.y * point.y;
		contour->size++;
	}
	if (contour->set)
		ccv_array_push(contour->set, &point);
}

void ccv_contour_free(ccv_contour_t* contour)
{
	if (contour->set)
		ccv_array_free(contour->set);
	ccfree(contour);
}

/* End of ccv/lib/ccv_util.c */
