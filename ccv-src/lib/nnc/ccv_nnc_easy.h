/**********************************************************
 * C-based/Cached/Core Computer Vision Library
 * Liu Liu, 2010-02-01
 **********************************************************/

/**********************************************************
 * CCV - Neural Network Collection
 **********************************************************/

#ifndef GUARD_ccv_nnc_easy_h
#define GUARD_ccv_nnc_easy_h

#include <ccv.h>
#include <nnc/ccv_nnc.h>

/**
 * Convenience API
 *
 * This header provides convenience APIs for nnc usage. Being convenience API,
 * it is optimized for shorthand coding, and may collide the naming space with
 * others.
 */
// c99 only, make sure your compiler supports that.
#define LIST_COUNT_N(_1,_2,_3,_4,_5,_6,_7,_8,_9,_10,_11,_12,_13,_14,_15,_16,_17,_18,_19,_20,_21,_22,_23,_24,_25,_26,_27,_28,_29,_30,_31,_32,_33,_34,_35,_36,_37,_38,_39,_40,_41,_42,_43,_44,_45,_46,_47,_48,_49,_50,_51,_52,_53,_54,_55,_56,_57,_58,_59,_60,_61,_62,_63,...) (_63)
#define LIST_COUNT(...) LIST_COUNT_N(__VA_ARGS__,62,61,60,59,58,57,56,55,54,53,52,51,50,49,48,47,46,45,44,43,42,41,40,39,38,37,36,35,34,33,32,31,30,29,28,27,26,25,24,23,22,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1)

#define TENSOR_LIST_X(...) (ccv_nnc_tensor_t* []){__VA_ARGS__}
#define TENSOR_LIST(...) TENSOR_LIST_X(__VA_ARGS__), LIST_COUNT(__VA_ARGS__)

#define TENSOR_PARAM_LIST_X(...) (ccv_nnc_tensor_param_t []){__VA_ARGS__}
#define TENSOR_PARAM_LIST(...) TENSOR_PARAM_LIST_X(__VA_ARGS__), LIST_COUNT(__VA_ARGS__)

#define NO_TENSOR_SYMBOL (ccv_nnc_tensor_symbol_t){.d = -1}

#define TENSOR_SYMBOL_LIST_X(...) (ccv_nnc_tensor_symbol_t []){__VA_ARGS__}
#define TENSOR_SYMBOL_LIST(...) TENSOR_SYMBOL_LIST_X(__VA_ARGS__), LIST_COUNT(__VA_ARGS__)

#define KV_X(_x, _y, ...) {(_x), (_y)}
#define KV(...) KV_X(__VA_ARGS__, 0)
#define TENSOR_BIND_MAP_X(...) (ccv_nnc_tensor_bind_t []){__VA_ARGS__}
#define TENSOR_BIND_MAP(...) TENSOR_BIND_MAP_X(__VA_ARGS__), (sizeof(TENSOR_BIND_MAP_X(__VA_ARGS__)) / sizeof(ccv_nnc_tensor_bind_t))

#define TENSOR_SYMBOL_MAP_X(...) (ccv_nnc_tensor_symbol_map_t []){__VA_ARGS__}
#define TENSOR_SYMBOL_MAP(...) TENSOR_SYMBOL_MAP_X(__VA_ARGS__), (sizeof(TENSOR_SYMBOL_MAP_X(__VA_ARGS__)) / sizeof(ccv_nnc_tensor_symbol_map_t))

#define GRAPH_EXEC_LIST_X(...) (ccv_nnc_graph_exec_t []){__VA_ARGS__}
#define GRAPH_EXEC_LIST(...) GRAPH_EXEC_LIST_X(__VA_ARGS__), LIST_COUNT(__VA_ARGS__)

#define GRAPH_EXEC_SYMBOL_LIST_X(...) (ccv_nnc_graph_exec_symbol_t []){__VA_ARGS__}
#define GRAPH_EXEC_SYMBOL_LIST(...) GRAPH_EXEC_SYMBOL_LIST_X(__VA_ARGS__), LIST_COUNT(__VA_ARGS__)

#define CPU_TENSOR_NHWC(...) ((ccv_nnc_tensor_param_t){.type=CCV_TENSOR_CPU_MEMORY,.format=CCV_TENSOR_FORMAT_NHWC,.datatype=CCV_32F,.dim={__VA_ARGS__}})
#define CPU_TENSOR_NCHW(...) ((ccv_nnc_tensor_param_t){.type=CCV_TENSOR_CPU_MEMORY,.format=CCV_TENSOR_FORMAT_NCHW,.datatype=CCV_32F,.dim={__VA_ARGS__}})
#define ONE_CPU_TENSOR CPU_TENSOR_NHWC // The default is NHWC
// This way, we can do error check on the device type :)
#define GPU_TENSOR_NHWC(device_id, ...) ((ccv_nnc_tensor_param_t){.type=(CCV_COMPUTE_DEVICE_##device_id) | CCV_TENSOR_GPU_MEMORY,.format=CCV_TENSOR_FORMAT_NHWC,.datatype=CCV_32F,.dim={__VA_ARGS__}})
#define GPU_TENSOR_NCHW(device_id, ...) ((ccv_nnc_tensor_param_t){.type=(CCV_COMPUTE_DEVICE_##device_id) | CCV_TENSOR_GPU_MEMORY,.format=CCV_TENSOR_FORMAT_NCHW,.datatype=CCV_32F,.dim={__VA_ARGS__}})
#define ONE_GPU_TENSOR GPU_TENSOR_NHWC // The default is NHWC

#define DIM_ALLOC(...) (int [CCV_NNC_MAX_DIM_ALLOC]){__VA_ARGS__}

#define ESCAPE_X(...) __VA_ARGS__
#define HINT_X_1(_stride_) ((ccv_nnc_hint_t){.stride={.dim={ESCAPE_X _stride_}}, .border={.begin={0},.end={0}}})
#define HINT_X_2(_stride_, _border_) ((ccv_nnc_hint_t){.stride={.dim={ESCAPE_X _stride_}}, .border={.begin={ESCAPE_X _border_},.end={ESCAPE_X _border_}}})
#define HINT_X_3(_stride_, _begin_, _end_) ((ccv_nnc_hint_t){.stride={.dim={ESCAPE_X _stride_}}, .border={.begin={ESCAPE_X _begin_},.end={ESCAPE_X _end_}}})
#define HINT_X_SEL(_1, _2, _3, _FX, ...) _FX
#define HINT(...) HINT_X_SEL(__VA_ARGS__, HINT_X_3, HINT_X_2, HINT_X_1)(__VA_ARGS__)

static inline size_t ccv_nnc_dimension_count(const int dim[CCV_NNC_MAX_DIM_ALLOC])
{
	if (dim[0] == 0)
		return 0;
	int i;
	size_t count = 1;
	for (i = 0; i < CCV_NNC_MAX_DIM_ALLOC && dim[i] > 0; i++)
		count *= dim[i];
	return count;
}

static inline size_t ccv_nnc_tensor_count(const ccv_nnc_tensor_param_t params)
{
	return ccv_nnc_dimension_count(params.dim);
}

static inline size_t ccv_nnc_tensor_data_size(const ccv_nnc_tensor_param_t params)
{
	return (CCV_GET_DATA_TYPE_SIZE(params.datatype) * (ssize_t)ccv_nnc_tensor_count(params) + 15) & -16;
}

static inline void ccv_nnc_tensor_view_get_dim(const ccv_nnc_tensor_view_t* const tv, int dim[CCV_NNC_MAX_DIM_ALLOC])
{
	int x;
	const int nd = ccv_nnc_tensor_nd(tv->info.dim);
	const int offset = CCV_NNC_MAX_DIM + 2 - nd;
	for (x = 0; x < offset; x++)
		dim[x] = 1;
	for (x = offset; x < CCV_NNC_MAX_DIM + 2; x++)
		dim[x] = tv->info.dim[x - offset];
}

static inline void ccv_nnc_tensor_view_check_dim(const ccv_nnc_tensor_view_t* const tv, int dim[CCV_NNC_MAX_DIM_ALLOC])
{
	int x;
	const int nd = ccv_nnc_tensor_nd(tv->info.dim);
	const int offset = CCV_NNC_MAX_DIM + 2 - nd;
	for (x = 0; x < offset; x++)
	{
		assert(dim[x] == 1);
	}
	for (x = offset; x < CCV_NNC_MAX_DIM + 2; x++)
	{
		assert(dim[x] == tv->info.dim[x - offset]);
	}
}

static inline void ccv_nnc_tensor_view_get_inc(const ccv_nnc_tensor_view_t* const tv, int inc[CCV_NNC_MAX_DIM_ALLOC])
{
	int x;
	const int nd = ccv_nnc_tensor_nd(tv->info.dim);
	const int offset = CCV_NNC_MAX_DIM + 2 - nd;
	for (x = 0; x < offset; x++)
		inc[x] = 1;
	for (x = offset; x < CCV_NNC_MAX_DIM + 2; x++)
		inc[x] = CCV_IS_TENSOR_VIEW(tv) ? tv->inc[x - offset] : tv->info.dim[x - offset];
}

static inline int ccv_nnc_tensor_get_n(const ccv_nnc_tensor_param_t params)
{
	switch (params.format)
	{
		case CCV_TENSOR_FORMAT_NHWC:
		case CCV_TENSOR_FORMAT_NCHW:
			if (ccv_nnc_tensor_nd(params.dim) == CCV_NNC_MAX_DIM + 1)
				return 1;
			else
				return params.dim[0];
		case CCV_TENSOR_FORMAT_CHWN:
			return params.dim[CCV_NNC_MAX_DIM + 1];
	}
	return 0;
}

static inline int ccv_nnc_tensor_get_c(const ccv_nnc_tensor_param_t params)
{
	switch (params.format)
	{
		case CCV_TENSOR_FORMAT_NHWC:
			if (ccv_nnc_tensor_nd(params.dim) == CCV_NNC_MAX_DIM + 1)
				return params.dim[CCV_NNC_MAX_DIM];
			else
				return params.dim[CCV_NNC_MAX_DIM + 1];
		case CCV_TENSOR_FORMAT_NCHW:
			if (ccv_nnc_tensor_nd(params.dim) == CCV_NNC_MAX_DIM + 1)
				return params.dim[0];
			else
				return params.dim[1];
		case CCV_TENSOR_FORMAT_CHWN:
			return params.dim[0];
	}
	return 0;
}

static inline void ccv_nnc_tensor_set_n(ccv_nnc_tensor_param_t* const params, const int n)
{
	switch (params->format)
	{
		case CCV_TENSOR_FORMAT_NHWC:
		case CCV_TENSOR_FORMAT_NCHW:
			params->dim[0] = n;
			break;
		case CCV_TENSOR_FORMAT_CHWN:
			params->dim[CCV_NNC_MAX_DIM + 1] = n;
			break;
	}
}

static inline void ccv_nnc_tensor_set_c(ccv_nnc_tensor_param_t* const params, const int nd, const int c)
{
	switch (params->format)
	{
		case CCV_TENSOR_FORMAT_NHWC:
			if (nd == CCV_NNC_MAX_DIM + 1)
				params->dim[CCV_NNC_MAX_DIM] = c;
			else
				params->dim[CCV_NNC_MAX_DIM + 1] = c;
			break;
		case CCV_TENSOR_FORMAT_NCHW:
			if (nd == CCV_NNC_MAX_DIM + 1)
				params->dim[0] = c;
			else
				params->dim[1] = c;
			break;
		case CCV_TENSOR_FORMAT_CHWN:
			params->dim[0] = c;
			break;
	}
}


#define CMD_BLAS(...) ((ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.blas={.a={__VA_ARGS__}}})
#define CMD_GEMM(_count) ((ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.blas={.a={1,1},.count=_count}}) // We default to alpha = 1 and beta = 1
#define CMD_GENERIC_X_0() ((ccv_nnc_cmd_param_t){.size={.dim={1,1,1}}})
#define CMD_GENERIC_X_F(...) ("This should not be used, you should have either 0 parameter or 3 parameters for CMD_GENERIC")
#define CMD_GENERIC_X_3(...) ((ccv_nnc_cmd_param_t){.size={.dim={__VA_ARGS__}}})
#define CMD_GENERIC_X_SEL(_0, _1, _2, _3, _FX, ...) _FX
// Using ## so that if it is empty, we omit one comma.
#define CMD_GENERIC(...) CMD_GENERIC_X_SEL(CMD_GENERIC_X_F, ##__VA_ARGS__, CMD_GENERIC_X_3, CMD_GENERIC_X_F, CMD_GENERIC_X_F, CMD_GENERIC_X_0)(__VA_ARGS__)

extern const int ccv_nnc_no_ofs[CCV_NNC_MAX_DIM_ALLOC];

extern const ccv_nnc_hint_t ccv_nnc_no_hint;
int ccv_nnc_is_no_hint(const ccv_nnc_hint_t hint);

extern const ccv_nnc_cmd_param_t ccv_nnc_cmd_auto;
int ccv_nnc_is_cmd_auto(const ccv_nnc_cmd_param_t params);

extern const ccv_nnc_tensor_param_t ccv_nnc_tensor_auto;
int ccv_nnc_is_tensor_auto(const ccv_nnc_tensor_param_t params);

// Generated command flags for easy creation of ccv_nnc_cmd_t objects.
#include "cmd/ccv_nnc_cmd_easy.h"

#endif
