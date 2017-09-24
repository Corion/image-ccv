// Matt Zeiler's Model, Smaller One
ccv_convnet_layer_param_t matt_a_params[13] = {
	// first layer (convolutional => max pool => rnorm)
	{
		.type = CCV_CONVNET_CONVOLUTIONAL,
		.bias = 0,
		.glorot = 1.41421356237, // sqrtf(2)
		.input = {
			.matrix = {
				.rows = 225,
				.cols = 225,
				.channels = 3,
				.partition = 1,
			},
		},
		.output = {
			.convolutional = {
				.count = 96,
				.strides = 2,
				.border = 1,
				.rows = 7,
				.cols = 7,
				.channels = 3,
				.partition = 2,
			},
		},
	},
	{
		.type = CCV_CONVNET_LOCAL_RESPONSE_NORM,
		.input = {
			.matrix = {
				.rows = 111,
				.cols = 111,
				.channels = 96,
				.partition = 2,
			},
		},
		.output = {
			.rnorm = {
				.size = 5,
				.kappa = 2,
				.alpha = 1e-4,
				.beta = 0.75,
			},
		},
	},
	{
		.type = CCV_CONVNET_MAX_POOL,
		.input = {
			.matrix = {
				.rows = 111,
				.cols = 111,
				.channels = 96,
				.partition = 2,
			},
		},
		.output = {
			.pool = {
				.strides = 2,
				.size = 3,
				.border = 0,
			},
		},
	},
	// second layer (convolutional => max pool => rnorm)
	{
		.type = CCV_CONVNET_CONVOLUTIONAL,
		.bias = 1,
		.glorot = 1.41421356237, // sqrtf(2)
		.input = {
			.matrix = {
				.rows = 55,
				.cols = 55,
				.channels = 96,
				.partition = 2,
			},
		},
		.output = {
			.convolutional = {
				.count = 256,
				.strides = 2,
				.border = 1,
				.rows = 5,
				.cols = 5,
				.channels = 96,
				.partition = 2,
			},
		},
	},
	{
		.type = CCV_CONVNET_LOCAL_RESPONSE_NORM,
		.input = {
			.matrix = {
				.rows = 27,
				.cols = 27,
				.channels = 256,
				.partition = 2,
			},
		},
		.output = {
			.rnorm = {
				.size = 5,
				.kappa = 2,
				.alpha = 1e-4,
				.beta = 0.75,
			},
		},
	},
	{
		.type = CCV_CONVNET_MAX_POOL,
		.input = {
			.matrix = {
				.rows = 27,
				.cols = 27,
				.channels = 256,
				.partition = 2,
			},
		},
		.output = {
			.pool = {
				.strides = 2,
				.size = 3,
				.border = 0,
			},
		},
	},
	// third layer (convolutional)
	{
		.type = CCV_CONVNET_CONVOLUTIONAL,
		.bias = 0,
		.glorot = 1.41421356237, // sqrtf(2)
		.input = {
			.matrix = {
				.rows = 13,
				.cols = 13,
				.channels = 256,
				.partition = 1,
			},
		},
		.output = {
			.convolutional = {
				.count = 384,
				.strides = 1,
				.border = 1,
				.rows = 3,
				.cols = 3,
				.channels = 256,
				.partition = 2,
			},
		},
	},
	// fourth layer (convolutional)
	{
		.type = CCV_CONVNET_CONVOLUTIONAL,
		.bias = 1,
		.glorot = 1.41421356237, // sqrtf(2)
		.input = {
			.matrix = {
				.rows = 13,
				.cols = 13,
				.channels = 384,
				.partition = 2,
			},
		},
		.output = {
			.convolutional = {
				.count = 384,
				.strides = 1,
				.border = 1,
				.rows = 3,
				.cols = 3,
				.channels = 384,
				.partition = 2,
			},
		},
	},
	// fifth layer (convolutional => max pool)
	{
		.type = CCV_CONVNET_CONVOLUTIONAL,
		.bias = 1,
		.glorot = 1.41421356237, // sqrtf(2)
		.input = {
			.matrix = {
				.rows = 13,
				.cols = 13,
				.channels = 384,
				.partition = 2,
			},
		},
		.output = {
			.convolutional = {
				.count = 256,
				.strides = 1,
				.border = 1,
				.rows = 3,
				.cols = 3,
				.channels = 384,
				.partition = 2,
			},
		},
	},
	{
		.type = CCV_CONVNET_MAX_POOL,
		.input = {
			.matrix = {
				.rows = 13,
				.cols = 13,
				.channels = 256,
				.partition = 2,
			},
		},
		.output = {
			.pool = {
				.strides = 2,
				.size = 3,
				.border = 0,
			},
		},
	},
	// sixth layer (full connect)
	{
		.type = CCV_CONVNET_FULL_CONNECT,
		.bias = 1,
		.glorot = 1.41421356237, // sqrtf(2)
		.input = {
			.matrix = {
				.rows = 6,
				.cols = 6,
				.channels = 256,
				.partition = 1,
			},
			.node = {
				.count = 6 * 6 * 256,
			},
		},
		.output = {
			.full_connect = {
				.relu = 1,
				.count = 4096,
			},
		},
	},
	// seventh layer (full connect)
	{
		.type = CCV_CONVNET_FULL_CONNECT,
		.bias = 1,
		.glorot = 1.41421356237, // sqrtf(2)
		.input = {
			.matrix = {
				.rows = 4096,
				.cols = 1,
				.channels = 1,
				.partition = 1,
			},
			.node = {
				.count = 4096,
			},
		},
		.output = {
			.full_connect = {
				.relu = 1,
				.count = 4096,
			},
		},
	},
	// eighth layer (full connect)
	{
		.type = CCV_CONVNET_FULL_CONNECT,
		.bias = 0,
		.glorot = 1.41421356237, // sqrtf(2)
		.input = {
			.matrix = {
				.rows = 4096,
				.cols = 1,
				.channels = 1,
				.partition = 1,
			},
			.node = {
				.count = 4096,
			},
		},
		.output = {
			.full_connect = {
				.relu = 0,
				.count = 1000,
			},
		},
	},
};
// Matt Zeiler's Model, Deeper and Larger One
ccv_convnet_layer_param_t matt_b_params[16] = {
	// first layer (convolutional => max pool => rnorm)
	{
		.type = CCV_CONVNET_CONVOLUTIONAL,
		.bias = 0,
		.glorot = 1.41421356237, // sqrtf(2)
		.input = {
			.matrix = {
				.rows = 225,
				.cols = 225,
				.channels = 3,
				.partition = 1,
			},
		},
		.output = {
			.convolutional = {
				.count = 128,
				.strides = 2,
				.border = 1,
				.rows = 7,
				.cols = 7,
				.channels = 3,
				.partition = 2,
			},
		},
	},
	{
		.type = CCV_CONVNET_LOCAL_RESPONSE_NORM,
		.input = {
			.matrix = {
				.rows = 111,
				.cols = 111,
				.channels = 128,
				.partition = 2,
			},
		},
		.output = {
			.rnorm = {
				.size = 5,
				.kappa = 2,
				.alpha = 1e-4,
				.beta = 0.75,
			},
		},
	},
	{
		.type = CCV_CONVNET_MAX_POOL,
		.input = {
			.matrix = {
				.rows = 111,
				.cols = 111,
				.channels = 128,
				.partition = 2,
			},
		},
		.output = {
			.pool = {
				.strides = 2,
				.size = 3,
				.border = 0,
			},
		},
	},
	// second layer (convolutional => max pool => rnorm)
	{
		.type = CCV_CONVNET_CONVOLUTIONAL,
		.bias = 1,
		.glorot = 1.41421356237, // sqrtf(2)
		.input = {
			.matrix = {
				.rows = 55,
				.cols = 55,
				.channels = 128,
				.partition = 2,
			},
		},
		.output = {
			.convolutional = {
				.count = 384,
				.strides = 2,
				.border = 1,
				.rows = 5,
				.cols = 5,
				.channels = 128,
				.partition = 2,
			},
		},
	},
	{
		.type = CCV_CONVNET_LOCAL_RESPONSE_NORM,
		.input = {
			.matrix = {
				.rows = 27,
				.cols = 27,
				.channels = 384,
				.partition = 2,
			},
		},
		.output = {
			.rnorm = {
				.size = 5,
				.kappa = 2,
				.alpha = 1e-4,
				.beta = 0.75,
			},
		},
	},
	{
		.type = CCV_CONVNET_MAX_POOL,
		.input = {
			.matrix = {
				.rows = 27,
				.cols = 27,
				.channels = 384,
				.partition = 2,
			},
		},
		.output = {
			.pool = {
				.strides = 2,
				.size = 3,
				.border = 0,
			},
		},
	},
	// third layer (convolutional)
	{
		.type = CCV_CONVNET_CONVOLUTIONAL,
		.bias = 0,
		.glorot = 1.41421356237, // sqrtf(2)
		.input = {
			.matrix = {
				.rows = 13,
				.cols = 13,
				.channels = 384,
				.partition = 1,
			},
		},
		.output = {
			.convolutional = {
				.count = 512,
				.strides = 1,
				.border = 1,
				.rows = 3,
				.cols = 3,
				.channels = 384,
				.partition = 2,
			},
		},
	},
	// fourth layer (convolutional)
	{
		.type = CCV_CONVNET_CONVOLUTIONAL,
		.bias = 1,
		.glorot = 1.41421356237, // sqrtf(2)
		.input = {
			.matrix = {
				.rows = 13,
				.cols = 13,
				.channels = 512,
				.partition = 2,
			},
		},
		.output = {
			.convolutional = {
				.count = 512,
				.strides = 1,
				.border = 1,
				.rows = 3,
				.cols = 3,
				.channels = 512,
				.partition = 2,
			},
		},
	},
	{
		.type = CCV_CONVNET_CONVOLUTIONAL,
		.bias = 1,
		.glorot = 1.41421356237, // sqrtf(2)
		.input = {
			.matrix = {
				.rows = 13,
				.cols = 13,
				.channels = 512,
				.partition = 2,
			},
		},
		.output = {
			.convolutional = {
				.count = 512,
				.strides = 1,
				.border = 1,
				.rows = 3,
				.cols = 3,
				.channels = 512,
				.partition = 2,
			},
		},
	},
	// fifth layer (convolutional => max pool)
	{
		.type = CCV_CONVNET_CONVOLUTIONAL,
		.bias = 1,
		.glorot = 1.41421356237, // sqrtf(2)
		.input = {
			.matrix = {
				.rows = 13,
				.cols = 13,
				.channels = 512,
				.partition = 1,
			},
		},
		.output = {
			.convolutional = {
				.count = 512,
				.strides = 1,
				.border = 1,
				.rows = 3,
				.cols = 3,
				.channels = 512,
				.partition = 2,
			},
		},
	},
	{
		.type = CCV_CONVNET_CONVOLUTIONAL,
		.bias = 1,
		.glorot = 1.41421356237, // sqrtf(2)
		.input = {
			.matrix = {
				.rows = 13,
				.cols = 13,
				.channels = 512,
				.partition = 2,
			},
		},
		.output = {
			.convolutional = {
				.count = 512,
				.strides = 1,
				.border = 1,
				.rows = 3,
				.cols = 3,
				.channels = 512,
				.partition = 2,
			},
		},
	},
	{
		.type = CCV_CONVNET_MAX_POOL,
		.input = {
			.matrix = {
				.rows = 13,
				.cols = 13,
				.channels = 512,
				.partition = 2,
			},
		},
		.output = {
			.pool = {
				.strides = 2,
				.size = 3,
				.border = 0,
			},
		},
	},
	{
		.type = CCV_CONVNET_AVERAGE_POOL,
		.input = {
			.matrix = {
				.rows = 6,
				.cols = 6,
				.channels = 512,
				.partition = 1,
			},
		},
		.output = {
			.pool = {
				.strides = 3,
				.size = 3,
				.border = 0,
			},
		},
	},
	// sixth layer (full connect)
	{
		.type = CCV_CONVNET_FULL_CONNECT,
		.bias = 1,
		.glorot = 1.41421356237, // sqrtf(2)
		.input = {
			.matrix = {
				.rows = 2,
				.cols = 2,
				.channels = 512,
				.partition = 1,
			},
			.node = {
				.count = 2 * 2 * 512,
			},
		},
		.output = {
			.full_connect = {
				.relu = 1,
				.count = 2048,
			},
		},
	},
	// seventh layer (full connect)
	{
		.type = CCV_CONVNET_FULL_CONNECT,
		.bias = 1,
		.glorot = 1.41421356237, // sqrtf(2)
		.input = {
			.matrix = {
				.rows = 2048,
				.cols = 1,
				.channels = 1,
				.partition = 1,
			},
			.node = {
				.count = 2048,
			},
		},
		.output = {
			.full_connect = {
				.relu = 1,
				.count = 2048,
			},
		},
	},
	// eighth layer (full connect)
	{
		.type = CCV_CONVNET_FULL_CONNECT,
		.bias = 0,
		.glorot = 1.41421356237, // sqrtf(2)
		.input = {
			.matrix = {
				.rows = 2048,
				.cols = 1,
				.channels = 1,
				.partition = 1,
			},
			.node = {
				.count = 2048,
			},
		},
		.output = {
			.full_connect = {
				.relu = 0,
				.count = 1000,
			},
		},
	},
};
// Matt Zeiler's Model, Deeper and Wider One
ccv_convnet_layer_param_t matt_c_params[17] = {
	// first layer (convolutional => max pool => rnorm)
	{
		.type = CCV_CONVNET_CONVOLUTIONAL,
		.bias = 0,
		.glorot = 1.41421356237, // sqrtf(2)
		.input = {
			.matrix = {
				.rows = 225,
				.cols = 225,
				.channels = 3,
				.partition = 1,
			},
		},
		.output = {
			.convolutional = {
				.count = 128,
				.strides = 2,
				.border = 1,
				.rows = 7,
				.cols = 7,
				.channels = 3,
				.partition = 2,
			},
		},
	},
	{
		.type = CCV_CONVNET_LOCAL_RESPONSE_NORM,
		.input = {
			.matrix = {
				.rows = 111,
				.cols = 111,
				.channels = 128,
				.partition = 2,
			},
		},
		.output = {
			.rnorm = {
				.size = 5,
				.kappa = 2,
				.alpha = 1e-4,
				.beta = 0.75,
			},
		},
	},
	{
		.type = CCV_CONVNET_MAX_POOL,
		.input = {
			.matrix = {
				.rows = 111,
				.cols = 111,
				.channels = 128,
				.partition = 2,
			},
		},
		.output = {
			.pool = {
				.strides = 2,
				.size = 3,
				.border = 0,
			},
		},
	},
	// second layer (convolutional => max pool => rnorm)
	{
		.type = CCV_CONVNET_CONVOLUTIONAL,
		.bias = 1,
		.glorot = 1.41421356237, // sqrtf(2)
		.input = {
			.matrix = {
				.rows = 55,
				.cols = 55,
				.channels = 128,
				.partition = 2,
			},
		},
		.output = {
			.convolutional = {
				.count = 384,
				.strides = 1,
				.border = 1,
				.rows = 3,
				.cols = 3,
				.channels = 128,
				.partition = 2,
			},
		},
	},
	{
		.type = CCV_CONVNET_LOCAL_RESPONSE_NORM,
		.input = {
			.matrix = {
				.rows = 55,
				.cols = 55,
				.channels = 384,
				.partition = 2,
			},
		},
		.output = {
			.rnorm = {
				.size = 5,
				.kappa = 2,
				.alpha = 1e-4,
				.beta = 0.75,
			},
		},
	},
	{
		.type = CCV_CONVNET_MAX_POOL,
		.input = {
			.matrix = {
				.rows = 55,
				.cols = 55,
				.channels = 384,
				.partition = 2,
			},
		},
		.output = {
			.pool = {
				.strides = 2,
				.size = 3,
				.border = 0,
			},
		},
	},
	// third layer (convolutional)
	{
		.type = CCV_CONVNET_CONVOLUTIONAL,
		.bias = 0,
		.glorot = 1.41421356237, // sqrtf(2)
		.input = {
			.matrix = {
				.rows = 27,
				.cols = 27,
				.channels = 384,
				.partition = 1,
			},
		},
		.output = {
			.convolutional = {
				.count = 512,
				.strides = 1,
				.border = 1,
				.rows = 3,
				.cols = 3,
				.channels = 384,
				.partition = 2,
			},
		},
	},
	// fourth layer (convolutional)
	{
		.type = CCV_CONVNET_CONVOLUTIONAL,
		.bias = 1,
		.glorot = 1.41421356237, // sqrtf(2)
		.input = {
			.matrix = {
				.rows = 27,
				.cols = 27,
				.channels = 512,
				.partition = 2,
			},
		},
		.output = {
			.convolutional = {
				.count = 512,
				.strides = 1,
				.border = 1,
				.rows = 3,
				.cols = 3,
				.channels = 512,
				.partition = 2,
			},
		},
	},
	{
		.type = CCV_CONVNET_CONVOLUTIONAL,
		.bias = 1,
		.glorot = 1.41421356237, // sqrtf(2)
		.input = {
			.matrix = {
				.rows = 27,
				.cols = 27,
				.channels = 512,
				.partition = 2,
			},
		},
		.output = {
			.convolutional = {
				.count = 512,
				.strides = 1,
				.border = 1,
				.rows = 3,
				.cols = 3,
				.channels = 512,
				.partition = 2,
			},
		},
	},
	{
		.type = CCV_CONVNET_MAX_POOL,
		.input = {
			.matrix = {
				.rows = 27,
				.cols = 27,
				.channels = 512,
				.partition = 2,
			},
		},
		.output = {
			.pool = {
				.strides = 2,
				.size = 3,
				.border = 0,
			},
		},
	},
	// fifth layer (convolutional => max pool)
	{
		.type = CCV_CONVNET_CONVOLUTIONAL,
		.bias = 1,
		.glorot = 1.41421356237, // sqrtf(2)
		.input = {
			.matrix = {
				.rows = 13,
				.cols = 13,
				.channels = 512,
				.partition = 1,
			},
		},
		.output = {
			.convolutional = {
				.count = 512,
				.strides = 1,
				.border = 1,
				.rows = 3,
				.cols = 3,
				.channels = 512,
				.partition = 2,
			},
		},
	},
	{
		.type = CCV_CONVNET_CONVOLUTIONAL,
		.bias = 1,
		.glorot = 1.41421356237, // sqrtf(2)
		.input = {
			.matrix = {
				.rows = 13,
				.cols = 13,
				.channels = 512,
				.partition = 2,
			},
		},
		.output = {
			.convolutional = {
				.count = 512,
				.strides = 1,
				.border = 1,
				.rows = 3,
				.cols = 3,
				.channels = 512,
				.partition = 2,
			},
		},
	},
	{
		.type = CCV_CONVNET_MAX_POOL,
		.input = {
			.matrix = {
				.rows = 13,
				.cols = 13,
				.channels = 512,
				.partition = 2,
			},
		},
		.output = {
			.pool = {
				.strides = 2,
				.size = 3,
				.border = 0,
			},
		},
	},
	{
		.type = CCV_CONVNET_AVERAGE_POOL,
		.input = {
			.matrix = {
				.rows = 6,
				.cols = 6,
				.channels = 512,
				.partition = 1,
			},
		},
		.output = {
			.pool = {
				.strides = 3,
				.size = 3,
				.border = 0,
			},
		},
	},
	// sixth layer (full connect)
	{
		.type = CCV_CONVNET_FULL_CONNECT,
		.bias = 1,
		.glorot = 1.41421356237, // sqrtf(2)
		.input = {
			.matrix = {
				.rows = 2,
				.cols = 2,
				.channels = 512,
				.partition = 1,
			},
			.node = {
				.count = 2 * 2 * 512,
			},
		},
		.output = {
			.full_connect = {
				.relu = 1,
				.count = 2048,
			},
		},
	},
	// seventh layer (full connect)
	{
		.type = CCV_CONVNET_FULL_CONNECT,
		.bias = 1,
		.glorot = 1.41421356237, // sqrtf(2)
		.input = {
			.matrix = {
				.rows = 2048,
				.cols = 1,
				.channels = 1,
				.partition = 1,
			},
			.node = {
				.count = 2048,
			},
		},
		.output = {
			.full_connect = {
				.relu = 1,
				.count = 2048,
			},
		},
	},
	// eighth layer (full connect)
	{
		.type = CCV_CONVNET_FULL_CONNECT,
		.bias = 0,
		.glorot = 1.41421356237, // sqrtf(2)
		.input = {
			.matrix = {
				.rows = 2048,
				.cols = 1,
				.channels = 1,
				.partition = 1,
			},
			.node = {
				.count = 2048,
			},
		},
		.output = {
			.full_connect = {
				.relu = 0,
				.count = 1000,
			},
		},
	},
};
