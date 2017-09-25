#include "ccv_nnc.h"
#include "ccv_nnc_easy.h"
#include "ccv_nnc_internal.h"
#include "ccv_internal.h"
#ifdef HAVE_CUDA
#include "gpu/ccv_nnc_compat.h"
#endif
#include "_ccv_nnc_graph.h"

void ccv_nnc_tensor_multiview(ccv_nnc_tensor_t* const tv, ccv_numeric_data_t data[], const uint8_t kind, const uint16_t repeat, const ccv_nnc_graph_t* const graph, ccv_nnc_tensor_multiview_t* const tensor_multiview)
{
	assert(kind == CCV_NNC_MULTIVIEW_K0N || kind == CCV_NNC_MULTIVIEW_K1N);
	assert(repeat > 0);
	tensor_multiview->type = CCV_TENSOR_MULTIVIEW;
	tensor_multiview->kind = kind;
	tensor_multiview->repeat = repeat;
	tensor_multiview->anchor = (intptr_t)graph;
	tensor_multiview->tv = tv;
	tensor_multiview->p = 0;
	tensor_multiview->offset = 0;
	tensor_multiview->rtvs = 0;
	int i;
	// Currently, only CCV_NNC_MULTIVIEW_K12 uses 3 tensors.
	for (i = 0; i < repeat + kind; i++)
	{
		tensor_multiview->data[i] = data[i];
		if (!tv)
		{
			ccv_nnc_tensor_multiview_t* const mv = (ccv_nnc_tensor_multiview_t*)data[i].ptr;
			mv->p = tensor_multiview;
		}
	}
}

void ccv_nnc_tensor_multiview_free(const ccv_nnc_tensor_multiview_t tensor_multiview)
{
	if (tensor_multiview.rtvs)
		ccv_array_free(tensor_multiview.rtvs);
}

void ccv_nnc_tensor_reference_to_multiview(ccv_nnc_tensor_multiview_t* const tensor_multiview, const off_t offset, ccv_nnc_tensor_t* const tensor)
{
	ccv_nnc_tensor_reference_t tensor_reference = {
		.offset = offset,
		.tensor = tensor,
	};
	if (!tensor_multiview->rtvs)
		tensor_multiview->rtvs = ccv_array_new(sizeof(ccv_nnc_tensor_reference_t), 0, 0);
	ccv_array_push(tensor_multiview->rtvs, &tensor_reference);
}

void ccv_nnc_tensor_multiview_broadcast(const ccv_nnc_tensor_multiview_t* const tensor_multiview)
{
	assert(tensor_multiview->tv);
	// Update the pointer on tv only if it is not a single tensor pointer.
	if (!CCV_NNC_MULTIVIEW_K01(tensor_multiview))
		tensor_multiview->tv->data = tensor_multiview->it;
	unsigned char* const data = tensor_multiview->tv->data.u8 - tensor_multiview->offset;
	const ccv_nnc_tensor_multiview_t* c = tensor_multiview;
	int i;
	do {
		if (c->rtvs)
			for (i = 0; i < c->rtvs->rnum; i++)
			{
				ccv_nnc_tensor_reference_t* reference = (ccv_nnc_tensor_reference_t*)ccv_array_get(c->rtvs, i);
				reference->tensor->data.u8 = data + reference->offset;
			}
		c = c->p;
	} while (c);
}

ccv_nnc_graph_exec_t ccv_nnc_graph_while(ccv_nnc_graph_t* const graph, uint32_t cmd, ccv_nnc_graph_t* const while_graph)
{
	assert(cmd == CCV_NNC_GRAPH_FORWARD || cmd == CCV_NNC_GRAPH_BACKWARD);
	ccv_nnc_graph_exec_t while_exec = ccv_nnc_graph_exec_new(graph, ccv_nnc_cmd(cmd, 0, CMD_GENERIC(), 0), ccv_nnc_no_hint, 0, 0, 0, 0);
	ccv_nnc_graph_exec_info_t* while_exec_info = (ccv_nnc_graph_exec_info_t*)ccv_array_get(graph->exec_info, while_exec.d);
	if (!graph->sub_graphs)
		graph->sub_graphs = ccv_array_new(sizeof(ccv_nnc_graph_t*), 1, 0);
	int i;
	if (while_graph->wraps)
	{
		// Copy wraps from sub graph to parent graph.
		if (!graph->wraps)
			graph->wraps = ccv_array_new(sizeof(ccv_nnc_graph_exec_t), while_graph->wraps->rnum, 0);
		for (i = 0; i < while_graph->wraps->rnum; i++)
			ccv_array_push(graph->wraps, ccv_array_get(while_graph->wraps, i));
	}
	ccv_array_push(graph->sub_graphs, &while_graph);
	while_graph->p = graph;
	while_graph->exec_idx = while_exec.d + 1;
	while_exec_info->graph_ref = graph->sub_graphs->rnum;
	return while_exec;
}

void ccv_nnc_graph_set_while_expr(ccv_nnc_graph_t* const while_graph, const ccv_nnc_graph_while_f while_expr, const void* const while_data, const ccv_nnc_graph_exec_t* const breakpoints, const int breakpoint_size)
{
	while_graph->while_data = while_data;
	while_graph->while_expr = while_expr;
	assert(breakpoint_size > 0);
	while_graph->breakpoint_size = breakpoint_size;
	while_graph->breakpoints = (ccv_nnc_graph_exec_t*)((while_graph->breakpoints) ? ccrealloc(while_graph->breakpoints, sizeof(ccv_nnc_graph_exec_t) * breakpoint_size) : ccmalloc(sizeof(ccv_nnc_graph_exec_t) * breakpoint_size));
	memcpy(while_graph->breakpoints, breakpoints, sizeof(ccv_nnc_graph_exec_t) * breakpoint_size);
}

#define TAG_TENSOR_REQUIRE_BROADCAST(x) (ccv_nnc_tensor_t*)((intptr_t)(x) | 1)
#define UNTAG_TENSOR_REQUIRE_BROADCAST(x) (ccv_nnc_tensor_t*)((intptr_t)(x) & ~(intptr_t)1)
#define IS_TAGGED_TENSOR_REQUIRE_BROADCAST(x) ((intptr_t)(x) & 1)

static void _ccv_nnc_graph_unwrap(const ccv_nnc_graph_t* const graph, const int count)
{
	if (!graph->wraps)
		return;
	int i, j;
	for (i = 0; i < graph->wraps->rnum; i++)
	{
		const ccv_nnc_graph_exec_t* const exec = (const ccv_nnc_graph_exec_t*)ccv_array_get(graph->wraps, i);
		const ccv_nnc_graph_t* const sub_graph = exec->graph;
		ccv_nnc_graph_exec_info_t* const exec_info = (ccv_nnc_graph_exec_info_t*)ccv_array_get(sub_graph->exec_info, exec->d);
		// Unwrap io first.
		if (exec_info->io_wraps)
		{
			ccv_nnc_tensor_t** const tensors = exec_info->inputs + (exec_info->input_size + exec_info->output_size) * exec_info->io_wrap_ptr;
			const int tensor_size = exec_info->input_size + exec_info->output_size;
			int rewrap = 0;
			for (j = 0; j < tensor_size && !rewrap; j++)
				// If I have a multi-view tensor and this multi-view tensor need to be unwrapped at this level (wrap_anchor)
				if (CCV_IS_TENSOR_MULTIVIEW(tensors[j]) && ((ccv_nnc_tensor_multiview_t*)tensors[j])->anchor == (intptr_t)graph)
					rewrap = 1;
			if (rewrap)
			{
				// Unwrap tensors at this level.
				++exec_info->io_wrap_ptr;
				ccv_nnc_tensor_t** const unwrap_tensors = exec_info->inputs + (exec_info->input_size + exec_info->output_size) * exec_info->io_wrap_ptr;
				for (j = 0; j < tensor_size; j++)
				{
					assert(!IS_TAGGED_TENSOR_REQUIRE_BROADCAST(tensors[j])); // I cannot encounter a tagged pointer.
					ccv_nnc_tensor_t* tensor = tensors[j];
					// Just copy it over if it is not a multiview tensor.
					while (CCV_IS_TENSOR_MULTIVIEW(tensor) && ((ccv_nnc_tensor_multiview_t*)tensor)->anchor == (intptr_t)graph)
					{
						// This can be unwrapped, do that.
						ccv_nnc_tensor_multiview_t* mv = (ccv_nnc_tensor_multiview_t*)tensor;
						const int off = mv->kind;
						const int mod = mv->repeat;
						// If reached the root.
						if (mv->tv)
						{
							// If it is a single tensor view pointer wrapped into multi-view tensor, no need to update pointer at all.
							if (!CCV_NNC_MULTIVIEW_K01(mv))
								// Update the pointer
								mv->it = mv->data[count >= off ? ((count - off) % mod) + off : count]; // See the comment of the CCV_NNC_MULTIVIEW_KXX enum for why the computation carried out this way.
							tensor = TAG_TENSOR_REQUIRE_BROADCAST(tensor); // Keep it dirty yet, will unwrap the first time encountered it in actual execution, using tagged pointer to keep track.
							break;
							// In this way, I can broadcast the pointer change only when executing it, to avoid early abortion causing no pointer
							// update is needed.
						} else
							tensor = (ccv_nnc_tensor_t*)mv->data[count >= off ? ((count - off) % mod) + off : count].ptr; // Unwrap.
					}
					unwrap_tensors[j] = tensor;
				}
			}
		}
		// Then unwrap cast.
		if (exec_info->cast_wraps)
		{
			ccv_nnc_tensor_t** const tensors = exec_info->casts + exec_info->cast_size * exec_info->cast_wrap_ptr;
			const int tensor_size = exec_info->cast_size;
			int rewrap = 0;
			for (j = 0; j < tensor_size && !rewrap; j++)
				// If I have a multi-view tensor and this multi-view tensor need to be unwrapped at this level (wrap_anchor)
				if (CCV_IS_TENSOR_MULTIVIEW(tensors[j]) && ((ccv_nnc_tensor_multiview_t*)tensors[j])->anchor == (intptr_t)graph)
					rewrap = 1;
			if (rewrap)
			{
				// Unwrap tensors at this level.
				++exec_info->cast_wrap_ptr;
				ccv_nnc_tensor_t** const unwrap_tensors = exec_info->casts + exec_info->cast_size * exec_info->cast_wrap_ptr;
				for (j = 0; j < tensor_size; j++)
				{
					assert(!IS_TAGGED_TENSOR_REQUIRE_BROADCAST(tensors[j])); // I cannot encounter a tagged pointer.
					ccv_nnc_tensor_t* tensor = tensors[j];
					// Just copy it over if it is not a multiview tensor.
					while (CCV_IS_TENSOR_MULTIVIEW(tensor) && ((ccv_nnc_tensor_multiview_t*)tensor)->anchor == (intptr_t)graph)
					{
						// This can be unwrapped, do that.
						ccv_nnc_tensor_multiview_t* mv = (ccv_nnc_tensor_multiview_t*)tensor;
						const int off = mv->kind;
						const int mod = mv->repeat;
						// If reached the root.
						if (mv->tv)
						{
							// If it is a single tensor view pointer wrapped into multi-view tensor, no need to update pointer at all.
							if (!CCV_NNC_MULTIVIEW_K01(mv))
								// Update the pointer
								mv->it = mv->data[count >= off ? ((count - off) % mod) + off : count]; // See the comment of the CCV_NNC_MULTIVIEW_KXX enum for why the computation carried out this way.
							tensor = TAG_TENSOR_REQUIRE_BROADCAST(tensor); // Keep it dirty yet, will unwrap the first time encountered it in actual execution, using tagged pointer to keep track.
							break;
							// In this way, I can broadcast the pointer change only when executing it, to avoid early abortion causing no pointer
							// update is needed.
						} else
							tensor = (ccv_nnc_tensor_t*)mv->data[count >= off ? ((count - off) % mod) + off : count].ptr; // Unwrap.
					}
					unwrap_tensors[j] = tensor;
				}
			}
		}
	}
}

static void _ccv_nnc_graph_rewrap(const ccv_nnc_graph_t* const graph) // Call this method at the end to roll the wrap_ptr back
{
	if (!graph->wraps)
		return;
	int i, j;
	for (i = 0; i < graph->wraps->rnum; i++)
	{
		const ccv_nnc_graph_exec_t* const exec = (const ccv_nnc_graph_exec_t*)ccv_array_get(graph->wraps, i);
		const ccv_nnc_graph_t* const sub_graph = exec->graph;
		ccv_nnc_graph_exec_info_t* const exec_info = (ccv_nnc_graph_exec_info_t*)ccv_array_get(sub_graph->exec_info, exec->d);
		// Rewrap io first.
		if (exec_info->io_wraps)
		{
			if (exec_info->io_wrap_ptr > 0)
			{
				ccv_nnc_tensor_t** const tensors = exec_info->inputs + (exec_info->input_size + exec_info->output_size) * (exec_info->io_wrap_ptr - 1);
				const int tensor_size = exec_info->input_size + exec_info->output_size;
				int rewrap = 0;
				for (j = 0; j < tensor_size && !rewrap; j++)
					// If I have a multi-view tensor and this multi-view tensor need to be unwrapped at this level (wrap_anchor)
					if (CCV_IS_TENSOR_MULTIVIEW(tensors[j]) && ((ccv_nnc_tensor_multiview_t*)tensors[j])->anchor == (intptr_t)graph)
						rewrap = 1;
				// If I did rewrap before, pop the pointer.
				if (rewrap)
					--exec_info->io_wrap_ptr;
			}
			assert(exec_info->io_wrap_ptr >= 0);
		}
		// Then rewrap cast.
		if (exec_info->cast_wraps)
		{
			if (exec_info->cast_wrap_ptr > 0)
			{
				ccv_nnc_tensor_t** const tensors = exec_info->casts + exec_info->cast_size * (exec_info->cast_wrap_ptr - 1);
				const int tensor_size = exec_info->cast_size;
				int rewrap = 0;
				for (j = 0; j < tensor_size && !rewrap; j++)
					// If I have a multi-view tensor and this multi-view tensor need to be unwrapped at this level (wrap_anchor)
					if (CCV_IS_TENSOR_MULTIVIEW(tensors[j]) && ((ccv_nnc_tensor_multiview_t*)tensors[j])->anchor == (intptr_t)graph)
						rewrap = 1;
				// If I did rewrap before, pop the pointer.
				if (rewrap)
					--exec_info->cast_wrap_ptr;
			}
			assert(exec_info->cast_wrap_ptr >= 0);
		}
	}
}

static int _ccv_nnc_graph_while_run(const ccv_nnc_graph_t* const graph, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_tensor_tape_t* const tensor_tape, const int flags, const ccv_nnc_graph_exec_t* const sources, const int source_size, const ccv_nnc_graph_exec_t* const destinations, const int destination_size)
{
	assert(tensor_tape == 0); // Cannot handle tensor tape yet.
	int i, j;
	for (i = 0; i < source_size; i++)
		if (sources[i].graph != graph)
			return CCV_NNC_EXEC_INVALID;
	for (i = 0; i < destination_size; i++)
		if (destinations[i].graph != graph)
			return CCV_NNC_EXEC_INVALID;
#define visitor(node, idx, d, ...) \
	do { \
		ccv_nnc_tensor_t** inputs = node->inputs + (node->input_size + node->output_size) * node->io_wrap_ptr; \
		ccv_nnc_tensor_t** outputs = inputs + node->input_size; \
		ccv_nnc_tensor_t** casts = node->casts + node->cast_size * node->cast_wrap_ptr; \
 		/* Broadcast the updates to all subscribed references for input / output, even though at this
		 * time output is not written yet, propagate pointer change is still valid. */ \
		for (i = 0; i < node->input_size; i++) \
			if (IS_TAGGED_TENSOR_REQUIRE_BROADCAST(inputs[i])) \
			{ \
				ccv_nnc_tensor_multiview_t* mv = (ccv_nnc_tensor_multiview_t*)UNTAG_TENSOR_REQUIRE_BROADCAST(inputs[i]); \
				assert(CCV_IS_TENSOR_MULTIVIEW(mv)); \
				if (mv->tv) /* This is marked dirty. Unwrap it and broadcast.*/ \
					ccv_nnc_tensor_multiview_broadcast(mv), inputs[i] = mv->tv; \
			} \
		for (i = 0; i < node->output_size; i++) \
			if (IS_TAGGED_TENSOR_REQUIRE_BROADCAST(outputs[i])) \
			{ \
				ccv_nnc_tensor_multiview_t* mv = (ccv_nnc_tensor_multiview_t*)UNTAG_TENSOR_REQUIRE_BROADCAST(outputs[i]); \
				assert(CCV_IS_TENSOR_MULTIVIEW(mv)); \
				if (mv->tv) /* This is marked dirty. Unwrap it and broadcast.*/ \
					ccv_nnc_tensor_multiview_broadcast(mv), outputs[i] = mv->tv; \
			} \
		for (i = 0; i < node->cast_size; i++) \
			if (IS_TAGGED_TENSOR_REQUIRE_BROADCAST(casts[i])) \
			{ \
				ccv_nnc_tensor_multiview_t* mv = (ccv_nnc_tensor_multiview_t*)UNTAG_TENSOR_REQUIRE_BROADCAST(casts[i]); \
				assert(CCV_IS_TENSOR_MULTIVIEW(mv)); \
				if (mv->tv) /* This is marked dirty. Unwrap it and broadcast.*/ \
					ccv_nnc_tensor_multiview_broadcast(mv), casts[i] = mv->tv; \
			} \
		if (node->cmd.cmd == CCV_NNC_GRAPH_FORWARD || node->cmd.cmd == CCV_NNC_GRAPH_BACKWARD) \
		{ \
			ccv_nnc_graph_t* sub_graph = *(ccv_nnc_graph_t**)ccv_array_get(graph->sub_graphs, node->graph_ref - 1); \
			_ccv_nnc_graph_while_run(sub_graph, inputs, node->input_size, outputs, node->output_size, tensor_tape, flags, (ccv_nnc_graph_exec_t*)ccv_array_get(sub_graph->sources, 0), sub_graph->sources->rnum, (ccv_nnc_graph_exec_t*)ccv_array_get(sub_graph->destinations, 0), sub_graph->destinations->rnum); \
		} else { \
			PRINT(CCV_CLI_VERBOSE, "%s [%d, %d]: [%d] -> [%d]\n", ccv_nnc_cmd_name(node->cmd.cmd), idx, d, node->input_size, node->output_size); \
			for (i = 0; i < node->input_size; i++) \
				PRINT(CCV_CLI_VERBOSE, "|-> %d. %p (%p)\n", i + 1, inputs[i], (inputs[i] ? inputs[i]->data.u8 : 0)); \
			for (i = 0; i < node->output_size; i++) \
				PRINT(CCV_CLI_VERBOSE, "|<- %d. %p (%p)\n", i + 1, outputs[i], (outputs[i] ? outputs[i]->data.u8 : 0)); \
			ccv_nnc_cmd_exec(node->cmd, node->hint, flags, inputs, node->input_size, outputs, node->output_size, 0); \
		} \
	} while (0)
	if (graph->while_expr)
	{
		// TODO: Need to do a broadcast when first entering this graph for all the inputs.
		// This is a while loop.
		ccv_array_t* follows = ccv_array_new(sizeof(ccv_nnc_graph_exec_t), graph->breakpoint_size, 0);
		for (i = 0; i < graph->breakpoint_size; i++)
		{
			const ccv_nnc_graph_exec_info_t* const exec_info = (const ccv_nnc_graph_exec_info_t*)ccv_array_get(graph->exec_info, graph->breakpoints->d);
			if (exec_info->outgoings)
				for (j = 0; j < exec_info->outgoings->rnum; j++)
				{
					const ccv_nnc_graph_exec_t exec = {
						.d = *(int*)ccv_array_get(exec_info->outgoings, j),
						.graph = graph,
					};
					ccv_array_push(follows, &exec);
				}
		}
		uint64_t count = 0;
		ccv_nnc_tensor_t count_tensor = ccv_nnc_tensor(&count, ONE_CPU_TENSOR(1, 1, 1), 0);
		ccv_nnc_tensor_t* special_tensors[] = { &count_tensor };
		for (;; ++count)
		{
			_ccv_nnc_graph_unwrap(graph, count);
			CCV_NNC_GRAPH_VISIT(graph, (ccv_nnc_graph_exec_info_t*)ccv_array_get(graph->exec_info, 0), graph->exec_info->rnum, sources, source_size, graph->breakpoints, graph->breakpoint_size, visitor);
			// Reached breakpoints, now check the breakpoint, if not met, break out.
			if (!graph->while_expr(special_tensors, 1, inputs, input_size, outputs, output_size, graph->while_data))
			{
				_ccv_nnc_graph_rewrap(graph);
				break;
			}
			if (follows->rnum > 0)
				CCV_NNC_GRAPH_VISIT(graph, (ccv_nnc_graph_exec_info_t*)ccv_array_get(graph->exec_info, 0), graph->exec_info->rnum, (ccv_nnc_graph_exec_t*)ccv_array_get(follows, 0), follows->rnum, destinations, destination_size, visitor);
			_ccv_nnc_graph_rewrap(graph);
		}
		ccv_array_free(follows);
	} else {
		CCV_NNC_GRAPH_VISIT(graph, (ccv_nnc_graph_exec_info_t*)ccv_array_get(graph->exec_info, 0), graph->exec_info->rnum, sources, source_size, destinations, destination_size, visitor);
	}
	return CCV_NNC_EXEC_SUCCESS;
}

int ccv_nnc_graph_while_run(const ccv_nnc_graph_t* const graph, ccv_nnc_tensor_tape_t* const tensor_tape, const int flags, const ccv_nnc_graph_exec_t* const sources, const int source_size, const ccv_nnc_graph_exec_t* const destinations, const int destination_size)
{
	return _ccv_nnc_graph_while_run(graph, 0, 0, 0, 0, tensor_tape, flags, sources, source_size, destinations, destination_size);
}
