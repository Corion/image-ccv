/**********************************************************
 * C-based/Cached/Core Computer Vision Library
 * Liu Liu, 2010-02-01
 **********************************************************/

/**********************************************************
 * CCV - Neural Network Collection
 **********************************************************/

#ifndef GUARD_ccv_nnc_internal_h
#define GUARD_ccv_nnc_internal_h

#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>

// Define some internal constraints

#define CCV_NNC_STACK_BITMASK_ALLOC (2)

typedef void (*ccv_nnc_cmd_tensor_auto_f)(const ccv_nnc_cmd_param_t cmd, const ccv_nnc_tensor_param_t* const inputs, const int input_size, const ccv_nnc_hint_t hint, ccv_nnc_tensor_param_t* const outputs, const int output_size);
typedef int (*ccv_nnc_cmd_bitmask_f)(const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size);

typedef struct {
	int flags;
	ccv_nnc_cmd_bitmask_f bitmask;
	ccv_nnc_cmd_tensor_auto_f tensor_auto;
} ccv_nnc_cmd_registry_t;

typedef struct {
	int tensor_formats; /**< [formats] The supported formats for this API implementation. */
	int tensor_datatypes; /**< [datatypes] The supported data types for this API implementation. */
	int tensor_memory; /**< [memory] The supported tensor memory type for this API implementation. */
	int algorithms; /**< [algorithms] Number of algorithms variation. */
	ccv_nnc_cmd_exec_f exec;
	ccv_nnc_cmd_autotune_f autotune;
} ccv_nnc_cmd_backend_registry_t;

static inline void ccv_nnc_hint_tensor_forward(const ccv_nnc_cmd_param_t cmd, const ccv_nnc_tensor_param_t a, const ccv_nnc_hint_t hint, ccv_nnc_tensor_param_t* b)
{
	int i;
	assert(a.format == b->format);
	const int nd = ccv_nnc_tensor_nd(a.dim);
	assert(nd == CCV_NNC_MAX_DIM + 1 || nd == CCV_NNC_MAX_DIM + 2);
	int hw = -1;
	if ((a.format == CCV_TENSOR_FORMAT_CHWN) ||
		(a.format == CCV_TENSOR_FORMAT_NHWC && nd == CCV_NNC_MAX_DIM + 1))
		hw = 0;
	else if ((a.format == CCV_TENSOR_FORMAT_NHWC && nd == CCV_NNC_MAX_DIM + 2) ||
			 (a.format == CCV_TENSOR_FORMAT_NCHW && nd == CCV_NNC_MAX_DIM + 1))
		hw = 1;
	else if (a.format == CCV_TENSOR_FORMAT_NCHW && nd == CCV_NNC_MAX_DIM + 2)
		hw = 2;
	assert(hw >= 0);
	for (i = 0; i < CCV_NNC_MAX_DIM; i++)
	{
		int stride = ccv_max(1, hint.stride.dim[i]);
		b->dim[i + hw] = (a.dim[i + hw] + hint.border.begin[i] + hint.border.end[i] - cmd.size.dim[i]) / stride + 1;
	}
}

static inline void ccv_nnc_hint_tensor_backward(const ccv_nnc_cmd_param_t cmd, const ccv_nnc_tensor_param_t a, const ccv_nnc_hint_t hint, ccv_nnc_tensor_param_t* b)
{
	int i;
	assert(a.format == b->format);
	const int nd = ccv_nnc_tensor_nd(a.dim);
	assert(nd == CCV_NNC_MAX_DIM + 1 || nd == CCV_NNC_MAX_DIM + 2);
	int hw = -1;
	if ((a.format == CCV_TENSOR_FORMAT_CHWN) ||
		(a.format == CCV_TENSOR_FORMAT_NHWC && nd == CCV_NNC_MAX_DIM + 1))
		hw = 0;
	else if ((a.format == CCV_TENSOR_FORMAT_NHWC && nd == CCV_NNC_MAX_DIM + 2) ||
			 (a.format == CCV_TENSOR_FORMAT_NCHW && nd == CCV_NNC_MAX_DIM + 1))
		hw = 1;
	else if (a.format == CCV_TENSOR_FORMAT_NCHW && nd == CCV_NNC_MAX_DIM + 2)
		hw = 2;
	assert(hw >= 0);
	for (i = 0; i < CCV_NNC_MAX_DIM; i++)
	{
		int stride = ccv_max(1, hint.stride.dim[i]);
		b->dim[i + hw] = (a.dim[i + hw] - 1) * stride - hint.border.begin[i] - hint.border.end[i] + cmd.size.dim[i];
	}
}

void ccv_nnc_hint_tensor_auto_forward_from_inputs(const ccv_nnc_cmd_param_t cmd, const ccv_nnc_tensor_param_t* const inputs, const int input_size, const ccv_nnc_hint_t hint, ccv_nnc_tensor_param_t* const outputs, const int output_size);
void ccv_nnc_hint_tensor_auto_backward_from_gradient(const ccv_nnc_cmd_param_t cmd, const ccv_nnc_tensor_param_t* const inputs, const int input_size, const ccv_nnc_hint_t hint, ccv_nnc_tensor_param_t* const outputs, const int output_size);
void ccv_nnc_hint_tensor_auto_backward_from_inputs(const ccv_nnc_cmd_param_t cmd, const ccv_nnc_tensor_param_t* const inputs, const int input_size, const ccv_nnc_hint_t hint, ccv_nnc_tensor_param_t* const outputs, const int output_size);

static inline off_t ccv_nnc_tensor_view_offset(const ccv_nnc_tensor_view_t* const tv, const int ofs[CCV_NNC_MAX_DIM_ALLOC])
{
	int i;
	off_t offset = 0;
	size_t inc = CCV_GET_DATA_TYPE_SIZE(tv->info.datatype);
	const int nd = ccv_nnc_tensor_nd(tv->inc);
	for (i = nd - 1; i >= 0; i--)
	{
		offset += ofs[i] * inc;
		inc *= tv->inc[i];
	}
	return offset;
}

#ifdef __cplusplus
#define REGISTER_COMMAND_BACKEND(x, y) extern "C" void _register_command_ ## x ## _backend_ ## y
#define REGISTER_COMMAND(x) extern "C" void _register_command_ ## x
#else
#define REGISTER_COMMAND_BACKEND(x, y) void _register_command_ ## x ## _backend_ ## y
#define REGISTER_COMMAND(x) void _register_command_ ## x
#endif
#define FIND_BACKEND(...)
#define FIND_FILE(...)

// x is the dimension.
// n[x] is the start point for the filter on y axis, so that we can avoid computing the padding.
// m[x] shows how long we should loop for filter on y axis, avoid computing the padding too.
#define SET_BORDER_OFFSET_SIZE_FOR(x, i, hint, wd, ad, n, m) \
	do { \
		n[x] = ccv_max(i[x] * hint.stride.dim[x] - hint.border.begin[x], 0) - (i[x] * hint.stride.dim[x] - hint.border.begin[x]); \
		m[x] = (wd)[x] - n[x] - (i[x] * hint.stride.dim[x] - hint.border.begin[x] + (wd)[x] - ccv_min(ad[x], i[x] * hint.stride.dim[x] - hint.border.begin[x] + (wd)[x])); \
	} while (0)

// Defines common graph visit macros

// The visitor function / macro takes parameter visitor(node_type* node, int index, int level, int term);
#define CCV_NNC_GRAPH_VISIT(_graph, nodes, node_size, sources, source_size, destinations, destination_size, visitor) \
	do { \
		/* Use the same data structure to do topological ordering. */ \
		typedef struct { \
			int8_t d; /* tag if this is the destination node. */ \
			int8_t r; /* tag if this is reached as destination node. */ \
			int32_t c; /* number of incoming edges. */ \
		} ccv_nnc_incoming_t; \
		/* Statistics of how many incoming edges for all nodes of a graph. */ \
		int _heap_mem_ = (node_size > 1024); \
		int _i_, _j_; \
		ccv_nnc_incoming_t* _incomings_; \
		if (_heap_mem_) \
			_incomings_ = (ccv_nnc_incoming_t*)ccmalloc(sizeof(ccv_nnc_incoming_t) * (node_size) + sizeof(int32_t) * (node_size) * 2); \
		else \
			_incomings_ = (ccv_nnc_incoming_t*)alloca(sizeof(ccv_nnc_incoming_t) * (node_size) + sizeof(int32_t) * (node_size) * 2); \
		memset(_incomings_, 0, sizeof(ccv_nnc_incoming_t) * (node_size)); \
		for (_i_ = 0; _i_ < (node_size); _i_++) \
		{ \
			if ((nodes)[_i_].outgoings) \
				for (_j_ = 0; _j_ < (nodes)[_i_].outgoings->rnum; _j_++) \
					++_incomings_[*(int*)ccv_array_get((nodes)[_i_].outgoings, _j_)].c; \
		} \
		/* After we have that statistics, we can do topsort and run the command. */ \
		int32_t* _exists_[2] = { \
			(int32_t*)(_incomings_ + (node_size)), \
			(int32_t*)(_incomings_ + (node_size)) + (node_size), \
		}; \
		for (_i_ = 0; _i_ < (destination_size); _i_++) \
		{ \
			assert((destinations)[_i_].graph == _graph); \
			/* tagging destination nodes. */ \
			_incomings_[(destinations)[_i_].d].d = 1; \
		} \
		for (_i_ = 0; _i_ < (source_size); _i_++) \
		{ \
			assert((sources)[_i_].graph == _graph); \
			_exists_[0][_i_] = (sources)[_i_].d; \
		} \
		int _exist_size_[2] = { \
			(source_size), \
			0, \
		}; \
		int _p_ = 0, _q_ = 1, _k_ = 0, _d_ = 0; /* ping, pong swap. */ \
		while (_exist_size_[_p_] > 0) \
		{ \
			_exist_size_[_q_] = 0; \
			for (_i_ = 0; _i_ < _exist_size_[_p_]; _i_++) \
			{ \
				visitor(((nodes) + _exists_[_p_][_i_]), (_exists_[_p_][_i_]), _k_, (_incomings_[_exists_[_p_][_i_]].d)); \
				/* mark as reached */ \
				if (_incomings_[_exists_[_p_][_i_]].d) \
				{ \
					++_d_; \
					_incomings_[_exists_[_p_][_i_]].r = 1; \
				} \
				if ((nodes)[_exists_[_p_][_i_]].outgoings) \
					for (_j_ = 0; _j_ < (nodes)[_exists_[_p_][_i_]].outgoings->rnum; _j_++) \
					{ \
						int d = *(int*)ccv_array_get((nodes)[_exists_[_p_][_i_]].outgoings, _j_); \
						--_incomings_[d].c; \
						/* If all incoming edges are consumed, and not all destination node are computed, push it into next round */ \
						if (_incomings_[d].c == 0 && _d_ < (destination_size)) \
						{ \
							_exists_[_q_][_exist_size_[_q_]] = d; \
							++_exist_size_[_q_]; \
						} \
					} \
			} \
			/* swap p and q. */ \
			CCV_SWAP(_p_, _q_, _i_ /* using i as temp holder */); \
			++_k_; \
		} \
		for (_i_ = 0; _i_ < (destination_size); _i_++) \
		{ \
			assert((destinations)[_i_].graph == _graph); \
			/* skip if this is already reached. */ \
			if (_incomings_[(destinations)[_i_].d].r) \
				continue; \
			/* this destination node should have every incoming nodes consumed. */ \
			assert(_incomings_[(destinations)[_i_].d].c == 0); \
			/* fetch the info for destination node and exec current node. */ \
			visitor(((nodes) + (destinations)[_i_].d), ((destinations)[_i_].d), _k_, (_incomings_[(destinations)[_i_].d].d)); \
		} \
		if (_heap_mem_) \
			ccfree(_incomings_); \
	} while (0);

typedef struct {
	int size;
	struct {
		int index;
		int level;
		int term;
	} node[1];
} ccv_nnc_graph_visit_t;

static inline void ccv_nnc_graph_visit_free(ccv_nnc_graph_visit_t* graph_visit)
{
	ccfree(graph_visit);
}

#define CCV_NNC_GRAPH_VISIT_FOR1(graph_visit, nodes, _node_, _index_, _level_, _term_, ...) { \
	int _i_; \
	for (_i_ = 0; _i_ < (graph_visit)->size; _i_++) { \
		const int _index_ __attribute__((unused)) = (graph_visit)->node[_i_].index; \
		const int _level_ __attribute__((unused)) = (graph_visit)->node[_i_].level; \
		const int _term_ __attribute__((unused)) = (graph_visit)->node[_i_].term; \
		typeof ((nodes)) const _node_ __attribute__((unused)) = (nodes) + _index_; \

#define ccv_nnc_graph_visit_for(graph_visit, nodes, ...) \
	CCV_NNC_GRAPH_VISIT_FOR1(graph_visit, nodes, ##__VA_ARGS__, _node_unused_, _index_unused_, _level_unused_, _term_unused_)

#define ccv_nnc_graph_visit_endfor } }

#define CCV_NNC_GRAPH_VISIT_NEW_VISITOR1(_, _index_, _level_, _term_) \
	_visit_->node[_visit_->size].index = (_index_); \
	_visit_->node[_visit_->size].level = (_level_); \
	_visit_->node[_visit_->size].term = (_term_); \
	++_visit_->size;

#define ccv_nnc_graph_visit_new(_graph, nodes, node_size, sources, source_size, destinations, destination_size) ({\
	ccv_nnc_graph_visit_t* _visit_ = (ccv_nnc_graph_visit_t*)ccmalloc(sizeof(ccv_nnc_graph_visit_t) + sizeof(_visit_->node[0]) * ((node_size) - 1)); \
	_visit_->size = 0; \
	CCV_NNC_GRAPH_VISIT(_graph, nodes, node_size, sources, source_size, destinations, destination_size, CCV_NNC_GRAPH_VISIT_NEW_VISITOR1); \
	assert(_visit_->size <= (node_size)); \
	_visit_; \
})

#endif
