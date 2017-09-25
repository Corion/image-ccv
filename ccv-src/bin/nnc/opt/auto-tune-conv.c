#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <3rdparty/dsfmt/dSFMT.h>
#include <sys/time.h>
#include <ctype.h>

static unsigned int get_current_time(void)
{
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec * 1000 + tv.tv_usec / 1000;
}

#define INPUT_DIM (3)
#define OUTPUT_DIM (128)

#define INPUT_SIZE (224)
#define OUTPUT_SIZE (224)

#define KERNEL_SIZE (3)

int main(int argc, char** argv)
{
	ccv_nnc_init();
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(INPUT_SIZE, INPUT_SIZE, INPUT_DIM), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(OUTPUT_SIZE, OUTPUT_SIZE, OUTPUT_DIM), 0);
	ccv_nnc_cmd_t cmd = CMD_CONVOLUTION_FORWARD(OUTPUT_DIM, KERNEL_SIZE, KERNEL_SIZE, INPUT_DIM);
	ccv_nnc_hint_t hint = ccv_nnc_hint_auto(cmd.info, a->info, b->info);
	assert(ccv_nnc_hint_verify(hint, cmd.info, a->info, b->info) == 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(OUTPUT_DIM, KERNEL_SIZE, KERNEL_SIZE, INPUT_DIM), 0);
	ccv_nnc_tensor_t* bias = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(OUTPUT_DIM), 0);
	// configure the inlets.
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < INPUT_DIM * KERNEL_SIZE * KERNEL_SIZE * OUTPUT_DIM; i++)
		w->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / (INPUT_DIM * KERNEL_SIZE * KERNEL_SIZE);
	for (i = 0; i < INPUT_SIZE * INPUT_SIZE * INPUT_DIM; i++)
		a->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < OUTPUT_DIM; i++)
		bias->data.f32[i] = (float)i / OUTPUT_DIM;
	unsigned int elapsed_time = get_current_time();
	ccv_nnc_cmd_exec(cmd, hint, 0, TENSOR_LIST(a, w, bias), TENSOR_LIST(b), 0);
	elapsed_time = get_current_time() - elapsed_time;
	printf("%u ms for ref\n", elapsed_time);
	ccv_nnc_tensor_t* c = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(OUTPUT_SIZE, OUTPUT_SIZE, OUTPUT_DIM), 0);
	ccv_nnc_cmd_t tuned_cmd = ccv_nnc_cmd_autotune(cmd, 0, hint, 0, TENSOR_LIST(a, w, bias), TENSOR_LIST(c), 0);
	elapsed_time = get_current_time();
	ccv_nnc_cmd_exec(tuned_cmd, hint, 0, TENSOR_LIST(a, w, bias), TENSOR_LIST(c), 0);
	elapsed_time = get_current_time() - elapsed_time;
	printf("%u ms for auto-tuned, backend %d, algorithm %d\n", elapsed_time, tuned_cmd.backend, tuned_cmd.algorithm);
	for (i = 0; i < OUTPUT_DIM * OUTPUT_SIZE * OUTPUT_SIZE; i++)
		if (fabs(b->data.f32[i] - c->data.f32[i]) > 1e-5)
			printf("%d %f %f\n", i, b->data.f32[i], c->data.f32[i]);
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(bias);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(a);
}
