#include <ccv.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_internal.h>

static int _ccv_nnc_ewsum_forw_bitmask(const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	if (output_bitmasks[0] == 1)
	{
		int i, j, flag = 0;
		for (i = 0; i < input_bitmask_size; i++)
		{
			for (j = 0; j < 64; j++)
				if (input_bitmasks[i] & (uint64_t)1 << j)
				{
					if (flag)
						return 0;
				} else
					break;
			// Trailing zero even if it is not the end of input_bitmask_size, mark flag,
			// if we encounter additional 1, return invalid.
			if (j < 64)
				flag = 1;
			// Always like 1111100000, no 1110010101
			for (; j < 64; j++)
				if (input_bitmasks[i] & (uint64_t)1 << j)
					return 0;
		}
		return 1;
	}
	return 0;
}

static int _ccv_nnc_ewsum_back_bitmask(const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	if ((input_bitmasks[0] & 1u) == 1u)
	{
		int i, j, flag = 0;
		for (i = 0; i < output_bitmask_size; i++)
		{
			for (j = 0; j < 64; j++)
				if (output_bitmasks[i] & (uint64_t)1 << j)
				{
					if (flag)
						return 0;
				} else
					break;
			// Trailing zero even if it is not the end of input_bitmask_size, mark flag,
			// if we encounter additional 1, return invalid.
			if (j < 64)
				flag = 1;
			// Always like 1111100000, no 1110010101
			for (; j < 64; j++)
				if (output_bitmasks[i] & (uint64_t)1 << j)
					return 0;
		}
		return 1;
	}
	return 0;
}

REGISTER_COMMAND(CCV_NNC_EWSUM_FORWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_ew_cpu_ref.c)
{
	registry->flags = CCV_NNC_CMD_ATTR_INPLACE;
	registry->bitmask = _ccv_nnc_ewsum_forw_bitmask;
	registry->tensor_auto = ccv_nnc_hint_tensor_auto_forward_from_inputs;
}

REGISTER_COMMAND(CCV_NNC_EWSUM_BACKWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_ew_cpu_ref.c)
{
	registry->flags = CCV_NNC_CMD_ATTR_INPLACE | CCV_NNC_CMD_ATTR_PASSTHROUGH;
	registry->bitmask = _ccv_nnc_ewsum_back_bitmask;
	registry->tensor_auto = ccv_nnc_hint_tensor_auto_backward_from_gradient;
}

static int _ccv_nnc_ewprod_forw_bitmask(const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	if (output_bitmasks[0] == 1)
	{
		int i, j, flag = 0;
		for (i = 0; i < input_bitmask_size; i++)
		{
			for (j = 0; j < 64; j++)
				if (input_bitmasks[i] & (uint64_t)1 << j)
				{
					if (flag)
						return 0;
				} else
					break;
			// Trailing zero even if it is not the end of input_bitmask_size, mark flag,
			// if we encounter additional 1, return invalid.
			if (j < 64)
				flag = 1;
			// Always like 1111100000, no 1110010101
			for (; j < 64; j++)
				if (input_bitmasks[i] & (uint64_t)1 << j)
					return 0;
		}
		return 1;
	}
	return 0;
}

static int _ccv_nnc_ewprod_back_bitmask(const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	int i, j;
	int input_flag = 0;
	int input_bitcount = 0;
	for (i = 0; i < input_bitmask_size; i++)
	{
		for (j = 0; j < 64; j++)
			// The first parameter can be absent.
			if (input_bitmasks[i] & (uint64_t)1 << j)
			{
				if (input_flag)
					return 0;
			} else
				break;
		input_bitcount += j;
		if (j < 64)
			input_flag = 1;
		// Always like 1111100000, no 1110010101
		for (; j < 64; j++)
			if (input_bitmasks[i] & (uint64_t)1 << j)
				return 0;
	}
	int output_flag = 0;
	int output_bitcount = 0;
	for (i = 0; i < output_bitmask_size; i++)
	{
		for (j = 0; j < 64; j++)
			if ((output_bitmasks[i] & (uint64_t)1 << j))
			{
				if (output_flag)
					return 0;
			} else
				break;
		output_bitcount += j;
		if (j < 64)
			output_flag = 1;
		for (; j < 64; j++)
			if (output_bitmasks[i] & (uint64_t)1 << j)
				return 0;
	}
	return output_bitcount + 2 /* Gradient + Original output */ == input_bitcount;
}

REGISTER_COMMAND(CCV_NNC_EWPROD_FORWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_ew_cpu_ref.c)
{
	registry->flags = CCV_NNC_CMD_ATTR_INPLACE;
	registry->bitmask = _ccv_nnc_ewprod_forw_bitmask;
	registry->tensor_auto = ccv_nnc_hint_tensor_auto_forward_from_inputs;
}

REGISTER_COMMAND(CCV_NNC_EWPROD_BACKWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_ew_cpu_ref.c)
{
	registry->flags = CCV_NNC_CMD_ATTR_NULL_IS_ONES;
	registry->bitmask = _ccv_nnc_ewprod_back_bitmask;
	registry->tensor_auto = ccv_nnc_hint_tensor_auto_backward_from_gradient;
}

static int _ccv_nnc_ewdiv_forw_bitmask(const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	if ((input_bitmasks[0] & 3u) == ((1u << 0) | (1u << 1)) && output_bitmasks[0] == 1u)
		return 1;
	// Nominator can be null (meaning 1).
	if ((input_bitmasks[0] & 3u) == ((0u << 0) | (1u << 1)) && output_bitmasks[0] == 1u)
		return 1;
	return 0;
}

REGISTER_COMMAND(CCV_NNC_EWDIV_FORWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_ew_cpu_ref.c)
{
	registry->flags = CCV_NNC_CMD_ATTR_INPLACE | CCV_NNC_CMD_ATTR_NULL_IS_ONES;
	registry->bitmask = _ccv_nnc_ewdiv_forw_bitmask;
	registry->tensor_auto = ccv_nnc_hint_tensor_auto_forward_from_inputs;
}

static int _ccv_nnc_ewdiv_back_bitmask(const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	// We don't need to know the original output.
	if ((input_bitmasks[0] & (15u & ~((uint64_t)1u << 1))) == ((1u << 0) | (0u << 1) | (1u << 2) | (1u << 3)) && output_bitmasks[0] == ((1u << 0) | (1u << 1)))
		return 1;
	if ((input_bitmasks[0] & (15u & ~((uint64_t)1u << 1))) == ((1u << 0) | (0u << 1) | (1u << 2) | (0u << 3)) && output_bitmasks[0] == ((1u << 0) | (0u << 1)))
		return 1;
	return 0;
}

REGISTER_COMMAND(CCV_NNC_EWDIV_BACKWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_ew_cpu_ref.c)
{
	registry->flags = CCV_NNC_CMD_ATTR_NULL_IS_ONES;
	registry->bitmask = _ccv_nnc_ewdiv_back_bitmask;
	registry->tensor_auto = ccv_nnc_hint_tensor_auto_backward_from_gradient;
}

static int _ccv_nnc_ewexp_forw_bitmask(const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	if ((input_bitmasks[0] & 1u) == 1u && output_bitmasks[0] == 1u)
		return 1;
	return 0;
}

static int _ccv_nnc_ewexp_back_bitmask(const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	if ((input_bitmasks[0] & (7u & ~((uint64_t)1u << 1))) == ((1u << 0) | (0u << 1) | (1u << 2)) && output_bitmasks[0] == 1u)
		return 1;
	return 0;
}

REGISTER_COMMAND(CCV_NNC_EWEXP_FORWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_ew_cpu_ref.c)
{
	registry->flags = CCV_NNC_CMD_ATTR_INPLACE;
	registry->bitmask = _ccv_nnc_ewexp_forw_bitmask;
	registry->tensor_auto = ccv_nnc_hint_tensor_auto_forward_from_inputs;
}

REGISTER_COMMAND(CCV_NNC_EWEXP_BACKWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_ew_cpu_ref.c)
{
	registry->flags = CCV_NNC_CMD_ATTR_INPLACE | CCV_NNC_CMD_ATTR_NULL_IS_ONES;
	registry->bitmask = _ccv_nnc_ewexp_back_bitmask;
	registry->tensor_auto = ccv_nnc_hint_tensor_auto_backward_from_gradient;
}

static int _ccv_nnc_ewlog_forw_bitmask(const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	if ((input_bitmasks[0] & 1u) == 1u && output_bitmasks[0] == 1u)
		return 1;
	return 0;
}

static int _ccv_nnc_ewlog_back_bitmask(const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	// We don't care about the original input.
	if ((input_bitmasks[0] & 3u) == 3u && output_bitmasks[0] == 1u)
		return 1;
	return 0;
}

REGISTER_COMMAND(CCV_NNC_EWLOG_FORWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_ew_cpu_ref.c)
{
	registry->flags = CCV_NNC_CMD_ATTR_INPLACE;
	registry->bitmask = _ccv_nnc_ewlog_forw_bitmask;
	registry->tensor_auto = ccv_nnc_hint_tensor_auto_forward_from_inputs;
}

REGISTER_COMMAND(CCV_NNC_EWLOG_BACKWARD)(ccv_nnc_cmd_registry_t* const registry)
	FIND_BACKEND(ccv_nnc_ew_cpu_ref.c)
{
	registry->flags = CCV_NNC_CMD_ATTR_INPLACE | CCV_NNC_CMD_ATTR_NULL_IS_ONES;
	registry->bitmask = _ccv_nnc_ewlog_back_bitmask;
	registry->tensor_auto = ccv_nnc_hint_tensor_auto_backward_from_gradient;
}
