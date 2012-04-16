#include "ccv.h"
#include "ccv_internal.h"
#ifdef HAVE_GSL
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#endif
#ifndef _WIN32
#include <sys/time.h>
#endif
#ifdef USE_OPENMP
#include <omp.h>
#endif
#ifdef HAVE_LIBLINEAR
#include <linear.h>
#endif

#define CCV_DPM_WINDOW_SIZE (8)
/*
static unsigned int _ccv_dpm_time_measure()
{
	struct timeval tv;
	gettimeofday(&tv, 0);
	return tv.tv_sec * 1000000 + tv.tv_usec;
}
*/
#define less_than(fn1, fn2, aux) ((fn1).value >= (fn2).value)
static CCV_IMPLEMENT_QSORT(_ccv_dpm_aspect_qsort, struct feature_node, less_than)
#undef less_than

#define less_than(a1, a2, aux) ((a1) < (a2))
static CCV_IMPLEMENT_QSORT(_ccv_dpm_area_qsort, int, less_than)
#undef less_than

static void _ccv_dpm_write_checkpoint(ccv_dpm_mixture_model_t* model, const char* dir)
{
	FILE* w = fopen(dir, "w+");
	fprintf(w, ",\n");
	int i, j, x, y, ch, count = 0;
	for (i = 0; i < model->count; i++)
	{
		if (model->root[i].root.w == 0)
			break;
		count++;
	}
	fprintf(w, "%d %d\n", model->count, count);
	for (i = 0; i < count; i++)
	{
		ccv_dpm_root_classifier_t* root_classifier = model->root + i;
		fprintf(w, "%d %d\n", root_classifier->root.w->rows, root_classifier->root.w->cols);
		fprintf(w, "%a\n", root_classifier->beta);
		ch = CCV_GET_CHANNEL(root_classifier->root.w->type);
		for (y = 0; y < root_classifier->root.w->rows; y++)
		{
			for (x = 0; x < root_classifier->root.w->cols * ch; x++)
				fprintf(w, "%a ", root_classifier->root.w->data.f32[y * root_classifier->root.w->cols * ch + x]);
			fprintf(w, "\n");
		}
		fprintf(w, "%d\n", root_classifier->count);
		for (j = 0; j < root_classifier->count; j++)
		{
			ccv_dpm_part_classifier_t* part_classifier = root_classifier->part + j;
			fprintf(w, "%d %d %d\n", part_classifier->x, part_classifier->y, part_classifier->z);
			fprintf(w, "%la %la %la %la\n", part_classifier->dx, part_classifier->dy, part_classifier->dxx, part_classifier->dyy);
			fprintf(w, "%d %d\n", part_classifier->w->rows, part_classifier->w->cols);
			ch = CCV_GET_CHANNEL(part_classifier->w->type);
			for (y = 0; y < part_classifier->w->rows; y++)
			{
				for (x = 0; x < part_classifier->w->cols * ch; x++)
					fprintf(w, "%a ", part_classifier->w->data.f32[y * part_classifier->w->cols * ch + x]);
				fprintf(w, "\n");
			}
		}
	}
	fclose(w);
}

/*
static void _ccv_dpm_read_checkpoint(ccv_dpm_mixture_model_t* model, char* dir)
{
	int count;
	char flag;
	fscanf(r, "%c", &flag);
	assert(flag == ',');
	fscanf(r, "%d %d", &model->count, &count);
	ccv_dpm_root_classifier_t* root_classifier = (ccv_dpm_root_classifier_t*)ccmalloc(sizeof(ccv_dpm_root_classifier_t) * count);
	memset(root_classifier, 0, sizeof(ccv_dpm_root_classifier_t) * count);
	int i, j;
	size_t size = sizeof(ccv_dpm_mixture_model_t) + sizeof(ccv_dpm_root_classifier_t) * count;
	for (i = 0; i < count; i++)
	{
		int rows, cols;
		fscanf(r, "%d %d", &rows, &cols);
		fscanf(r, "%f", &root_classifier[i].beta);
		root_classifier[i].root.w = ccv_dense_matrix_new(rows, cols, CCV_32F | 31, ccmalloc(ccv_compute_dense_matrix_size(rows, cols, CCV_32F | 31)), 0);
		size += ccv_compute_dense_matrix_size(rows, cols, CCV_32F | 31);
		for (j = 0; j < rows * cols * 31; j++)
			fscanf(r, "%f", &root_classifier[i].root.w->data.f32[j]);
		fscanf(r, "%d", &root_classifier[i].count);
		ccv_dpm_part_classifier_t* part_classifier = (ccv_dpm_part_classifier_t*)ccmalloc(sizeof(ccv_dpm_part_classifier_t) * root_classifier[i].count);
		size += sizeof(ccv_dpm_part_classifier_t) * root_classifier[i].count;
		for (j = 0; j < root_classifier[i].count; j++)
		{
			fscanf(r, "%d %d %d", &part_classifier[j].x, &part_classifier[j].y, &part_classifier[j].z);
			fscanf(r, "%lf %lf %lf %lf", &part_classifier[j].dx, &part_classifier[j].dy, &part_classifier[j].dxx, &part_classifier[j].dyy);
			fscanf(r, "%d %d", &rows, &cols);
			part_classifier[j].w = ccv_dense_matrix_new(rows, cols, CCV_32F | 31, ccmalloc(ccv_compute_dense_matrix_size(rows, cols, CCV_32F | 31)), 0);
			size += ccv_compute_dense_matrix_size(rows, cols, CCV_32F | 31);
			for (k = 0; k < rows * cols * 31; k++)
				fscanf(r, "%f", &part_classifier[j].w->data.f32[k]);
		}
		root_classifier[i].part = part_classifier;
	}
	fclose(r);
}
*/

static void _ccv_dpm_mixture_model_cleanup(ccv_dpm_mixture_model_t* model)
{
	/* this is different because it doesn't compress to a continuous memory region */
	int i, j;
	for (i = 0; i < model->count; i++)
	{
		ccv_dpm_root_classifier_t* root_classifier = model->root + i;
		for (j = 0; j < root_classifier->count; j++)
		{
			ccv_dpm_part_classifier_t* part_classifier = root_classifier->part + j;
			ccv_matrix_free(part_classifier->w);
		}
		if (root_classifier->count > 0)
			ccfree(root_classifier->part);
		if (root_classifier->root.w != 0)
			ccv_matrix_free(root_classifier->root.w);
	}
	ccfree(model->root);
	model->count = 0;
	model->root = 0;
}

void ccv_dpm_mixture_model_new(char** posfiles, ccv_rect_t* bboxes, int posnum, char** bgfiles, int bgnum, int negnum, const char* dir, ccv_dpm_new_param_t params)
{
	int i, j, k, l, n, x, y;
	ccv_dpm_mixture_model_t* model = (ccv_dpm_mixture_model_t*)ccmalloc(sizeof(ccv_dpm_mixture_model_t));
	model->count = params.components;
	model->root = (ccv_dpm_root_classifier_t*)ccmalloc(sizeof(ccv_dpm_root_classifier_t) * model->count);
	memset(model->root, 0, sizeof(ccv_dpm_root_classifier_t) * model->count);
	struct feature_node* fn = (struct feature_node*)ccmalloc(sizeof(struct feature_node) * posnum);
	for (i = 0; i < posnum; i++)
	{
		assert(bboxes[i].width > 0 && bboxes[i].height > 0);
		fn[i].value = (float)bboxes[i].width / (float)bboxes[i].height;
		fn[i].index = i;
	}
	_ccv_dpm_aspect_qsort(fn, posnum, 0);
	double mean = 0;
	for (i = 0; i < posnum; i++)
		mean += fn[i].value;
	mean /= posnum;
	double variance = 0;
	for (i = 0; i < posnum; i++)
		variance = (fn[i].value - mean) * (fn[i].value - mean);
	variance /= posnum;
	printf("global mean: %lf, & variance: %lf\ninterclass mean(variance):", mean, variance);
	int* mnum = (int*)alloca(sizeof(int) * params.components);
	int outnum = posnum, innum = 0;
	for (i = 0; i < params.components; i++)
	{
		mnum[i] = (int)((double)outnum / (double)(params.components - i) + 0.5);
		double mean = 0;
		for (j = innum; j < innum + mnum[i]; j++)
			mean += fn[j].value;
		mean /= mnum[i];
		double variance = 0;
		for (j = innum; j < innum + mnum[i]; j++)
			variance = (fn[j].value - mean) * (fn[j].value - mean);
		variance /= mnum[i];
		printf(" %lf(%lf)", mean, variance);
		outnum -= mnum[i];
		innum += mnum[i];
	}
	printf("\n");
	int* areas = (int*)ccmalloc(sizeof(int) * posnum);
	for (i = 0; i < posnum; i++)
		areas[i] = bboxes[i].width * bboxes[i].height;
	_ccv_dpm_area_qsort(areas, posnum, 0);
	// so even the object is 1/4 in size, we can still detect them (in detection phase, we start at 2x image)
	int area = ccv_clamp(areas[(int)(posnum * 0.2 + 0.5)], params.min_area, params.max_area);
	ccfree(areas);
	gsl_rng_env_setup();
	gsl_rng* rng = gsl_rng_alloc(gsl_rng_default);
	gsl_rng_set(rng, *(unsigned long int*)&params);
	innum = 0;
	/* initialize root mixture model with liblinear */
	printf("initializing root mixture model\n");
	for (i = 0; i < params.components; i++)
	{
		double aspect = 0;
		for (j = innum; j < innum + mnum[i]; j++)
			aspect += fn[j].value;
		aspect /= mnum[i];
		int cols = ccv_max((int)(sqrtf(area / aspect) * aspect / CCV_DPM_WINDOW_SIZE + 0.5), 1);
		int rows = ccv_max((int)(sqrtf(area / aspect) / CCV_DPM_WINDOW_SIZE + 0.5), 1);
		printf(" - creating initial model %d(%d) at %dx%d\n", i + 1, params.components, cols, rows);
		struct problem prob;
		prob.n = 31 * cols * rows + 1;
		prob.bias = 1.0;
		prob.y = (int*)malloc(sizeof(int) * (mnum[i] + negnum) * (!!params.symmetric + 1));
		prob.x = (struct feature_node**)malloc(sizeof(struct feature_node*) * (mnum[i] + negnum) * (!!params.symmetric + 1));
		printf(" - generating positive examples ");
		fflush(stdout);
		l = 0;
		for (j = innum; j < innum + mnum[i]; j++)
		{
			ccv_rect_t bbox = bboxes[fn[j].index];
			int mcols = (int)(sqrtf(bbox.width * bbox.height * cols / (float)rows) + 0.5);
			int mrows = (int)(sqrtf(bbox.width * bbox.height * rows / (float)cols) + 0.5);
			bbox.x = bbox.x + (bbox.width - mcols) / 2;
			bbox.y = bbox.y + (bbox.height - mrows) / 2;
			bbox.width = mcols;
			bbox.height = mrows;
			if (mcols * 2 < cols * CCV_DPM_WINDOW_SIZE || mrows * 2 < rows * CCV_DPM_WINDOW_SIZE)
			// resolution is too low to be useful
				continue;
			ccv_dense_matrix_t* image = 0;
			ccv_read(posfiles[fn[j].index], &image, (params.grayscale ? CCV_IO_GRAY : 0) | CCV_IO_ANY_FILE);
			assert(image != 0);
			ccv_dense_matrix_t* up2x = 0;
			ccv_sample_up(image, &up2x, 0, 0, 0);
			ccv_matrix_free(image);
			ccv_dense_matrix_t* slice = 0;
			ccv_slice(up2x, (ccv_matrix_t**)&slice, 0, bbox.y * 2, bbox.x * 2, bbox.height * 2, bbox.width * 2);
			ccv_matrix_free(up2x);
			ccv_dense_matrix_t* resize = 0;
			ccv_resample(slice, &resize, 0, rows * CCV_DPM_WINDOW_SIZE, cols * CCV_DPM_WINDOW_SIZE, CCV_INTER_AREA);
			ccv_matrix_free(slice);
			ccv_dense_matrix_t* hog = 0;
			ccv_hog(resize, &hog, 0, 9, CCV_DPM_WINDOW_SIZE);
			struct feature_node* features = (struct feature_node*)malloc(sizeof(struct feature_node) * (31 * cols * rows + 2));
			for (k = 0; k < rows * cols * 31; k++)
			{
				features[k].index = k + 1;
				features[k].value = hog->data.f32[k];
			}
			features[31 * rows * cols].index = 31 * rows * cols + 1;
			features[31 * rows * cols].value = prob.bias;
			features[31 * rows * cols + 1].index = -1;
			ccv_matrix_free(hog);
			prob.x[l] = features;
			prob.y[l] = 1;
			++l;
			/* I use a brutal way to add symmetric support: add flipped data.
			 * It works because liblinear is super fast */
			if (params.symmetric)
			{
				ccv_flip(resize, &resize, 0, CCV_FLIP_X);
				hog = 0;
				ccv_hog(resize, &hog, 0, 9, CCV_DPM_WINDOW_SIZE);
				features = (struct feature_node*)malloc(sizeof(struct feature_node) * (31 * cols * rows + 2));
				for (k = 0; k < rows * cols * 31; k++)
				{
					features[k].index = k + 1;
					features[k].value = hog->data.f32[k];
				}
				features[31 * rows * cols].index = 31 * rows * cols + 1;
				features[31 * rows * cols].value = prob.bias;
				features[31 * rows * cols + 1].index = -1;
				ccv_matrix_free(hog);
				prob.x[l] = features;
				prob.y[l] = 1;
				++l;
			}
			ccv_matrix_free(resize);
			printf(".");
			fflush(stdout);
		}
		printf("\n - generating negative examples ");
		fflush(stdout);
		n = 0;
		while (n < negnum)
		{
			double p = (double)negnum / (double)bgnum;
			for (j = 0; j < bgnum; j++)
				if (gsl_rng_uniform(rng) < p)
				{
					ccv_dense_matrix_t* image = 0;
					ccv_read(bgfiles[j], &image, (params.grayscale ? CCV_IO_GRAY : 0) | CCV_IO_ANY_FILE);
					assert(image != 0);
					ccv_dense_matrix_t* slice = 0;
					int y = gsl_rng_uniform_int(rng, image->rows - rows * CCV_DPM_WINDOW_SIZE);
					int x = gsl_rng_uniform_int(rng, image->cols - cols * CCV_DPM_WINDOW_SIZE);
					ccv_slice(image, (ccv_matrix_t**)&slice, 0, y, x, rows * CCV_DPM_WINDOW_SIZE, cols * CCV_DPM_WINDOW_SIZE);
					ccv_matrix_free(image);
					ccv_dense_matrix_t* hog = 0;
					ccv_hog(slice, &hog, 0, 9, CCV_DPM_WINDOW_SIZE);
					struct feature_node* features = (struct feature_node*)malloc(sizeof(struct feature_node) * (31 * cols * rows + 2));
					for (k = 0; k < 31 * rows * cols; k++)
					{
						features[k].index = k + 1;
						features[k].value = hog->data.f32[k];
					}
					features[31 * rows * cols].index = 31 * rows * cols + 1;
					features[31 * rows * cols].value = prob.bias;
					features[31 * rows * cols + 1].index = -1;
					prob.x[l] = features;
					prob.y[l] = -1;
					ccv_matrix_free(hog);
					++l;
					if (params.symmetric)
					{
						ccv_flip(slice, &slice, 0, CCV_FLIP_X);
						hog = 0;
						ccv_hog(slice, &hog, 0, 9, CCV_DPM_WINDOW_SIZE);
						features = (struct feature_node*)malloc(sizeof(struct feature_node) * (31 * cols * rows + 2));
						for (k = 0; k < 31 * rows * cols; k++)
						{
							features[k].index = k + 1;
							features[k].value = hog->data.f32[k];
						}
						features[31 * rows * cols].index = 31 * rows * cols + 1;
						features[31 * rows * cols].value = prob.bias;
						features[31 * rows * cols + 1].index = -1;
						prob.x[l] = features;
						prob.y[l] = -1;
						ccv_matrix_free(hog);
						++l;
					}
					ccv_matrix_free(slice);
					++n;
					printf(".");
					fflush(stdout);
					if (n >= negnum)
						break;
				}
		}
		prob.l = l;
		printf("\n - generated %d examples with %d dimensions each\n"
			   " - running liblinear for initial linear SVM model (L2-regularized, L1-loss)\n", prob.l, prob.n);
		struct parameter linear_parameters = { .solver_type = L2R_L1LOSS_SVC_DUAL,
											   .eps = 1e-1,
											   .C = 1.0,
											   .nr_weight = 0,
											   .weight_label = 0,
											   .weight = 0 };
		const char* err = check_parameter(&prob, &linear_parameters);
		if (err)
		{
			printf(" - ERROR: cannot pass check parameter: %s\n", err);
			exit(-1);
		}
		struct model* linear = train(&prob, &linear_parameters);
		assert(linear != 0);
		printf(" - model->label[0]: %d, model->nr_class: %d, model->nr_feature: %d\n", linear->label[0], linear->nr_class, linear->nr_feature);
		ccv_dpm_root_classifier_t* root_classifier = model->root + i;
		root_classifier->root.w = ccv_dense_matrix_new(rows, cols, CCV_32F | 31, 0, 0);
		for (j = 0; j < 31 * rows * cols; j++)
			root_classifier->root.w->data.f32[j] = linear->w[j];
		root_classifier->beta = linear->w[31 * rows * cols];
		free_and_destroy_model(&linear);
		free(prob.y);
		for (j = 0; j < prob.l; j++)
			free(prob.x[j]);
		free(prob.x);
		ccv_make_matrix_immutable(root_classifier->root.w);
		_ccv_dpm_write_checkpoint(model, dir);
		innum += mnum[i];
	}
	ccfree(fn);
	if (params.components > 1)
	{
		/* TODO: coordinate-descent for lsvm */
		printf("optimizing root mixture model with coordinate-descent approach\n");
	} else {
		printf("components == 1, skipped coordinate-descent to optimize root mixture model\n");
	}
	/* initialize part filter */
	printf("initializing part filters\n");
	for (i = 0; i < params.components; i++)
	{
		printf(" - initializing part filters for model %d(%d)\n", i + 1, params.components);
		ccv_dpm_root_classifier_t* root_classifier = model->root + i;
		ccv_dense_matrix_t* w = 0;
		ccv_sample_up(root_classifier->root.w, &w, 0, 0, 0);
		ccv_dense_matrix_t* out = 0;
		ccv_visualize(root_classifier->root.w, &out, 0);
		ccv_write(out, "test/w.png", 0, CCV_IO_PNG_FILE, 0);
		ccv_matrix_free(out);
		ccv_make_matrix_mutable(w);
		root_classifier->count = params.parts;
		root_classifier->part = (ccv_dpm_part_classifier_t*)ccmalloc(sizeof(ccv_dpm_part_classifier_t) * params.parts);
		double area = w->rows * w->cols / (double)params.parts;
		for (j = 0; j < params.parts; j++)
		{
			ccv_dpm_part_classifier_t* part_classifier = root_classifier->part + j;
			int dx = 0, dy = 0, dw = 0, dh = 0;
			double dsum = -DBL_MAX;
			for (l = 1; l < ccv_min(w->rows + 1, area + 1); l++)
			{
				n = (int)(area / l + 0.5);
				if (n < 1 || n > w->cols)
					continue;
				if (l > n * 2 || n > l * 2)
					continue;
				//if (params.symmetric)
				{
				//} else {
					for (y = 0; y < w->rows - l + 1; y++)
						for (x = 0; x < w->cols - n + 1; x++)
						{
							ccv_dense_matrix_t* slice = 0;
							ccv_slice(w, (ccv_matrix_t**)&slice, 0, y, x, l, n);
							double sum = ccv_sum(slice, CCV_UNSIGNED) / (double)(l * n);
							if (sum > dsum)
							{
								dsum = sum;
								dx = x;
								dy = y;
								dw = n;
								dh = l;
							}
							ccv_matrix_free(slice);
						}
				}
			}
			ccv_dense_matrix_t* out = 0;
			ccv_visualize(w, &out, 0);
			char buf[1024];
			sprintf(buf, "test/%d.png", j);
			ccv_write(out, buf, 0, CCV_IO_PNG_FILE, 0);
			ccv_matrix_free(out);
			printf(" ---- part %d(%d) %dx%d at (%d,%d), entropy: %lf\n", j + 1, params.parts, dw, dh, dx, dy, dsum);
			part_classifier->dx = 0;
			part_classifier->dy = 0;
			part_classifier->dxx = 0.1f;
			part_classifier->dyy = 0.1f;
			part_classifier->x = dx;
			part_classifier->y = dy;
			part_classifier->z = 1;
			part_classifier->w = 0;
			ccv_slice(w, (ccv_matrix_t**)&part_classifier->w, 0, dy, dx, dh, dw);
			/* clean up the region we selected */
			float* w_ptr = (float*)ccv_get_dense_matrix_cell_by(CCV_32F | 31, w, dy, dx, 0);
			for (y = 0; y < dh; y++)
			{
				for (x = 0; x < dw * 31; x++)
					w_ptr[x] = 0;
				w_ptr += w->cols * 31;
			}
		}
		ccv_matrix_free(w);
	}
	gsl_rng_free(rng);
	_ccv_dpm_mixture_model_cleanup(model);
	ccfree(model);
}

static int _ccv_is_equal(const void* _r1, const void* _r2, void* data)
{
	const ccv_root_comp_t* r1 = (const ccv_root_comp_t*)_r1;
	const ccv_root_comp_t* r2 = (const ccv_root_comp_t*)_r2;
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
	const ccv_root_comp_t* r1 = (const ccv_root_comp_t*)_r1;
	const ccv_root_comp_t* r2 = (const ccv_root_comp_t*)_r2;
	int distance = (int)(r1->rect.width * 0.25 + 0.5);

	return r2->id == r1->id &&
		   r2->rect.x <= r1->rect.x + distance &&
		   r2->rect.x >= r1->rect.x - distance &&
		   r2->rect.y <= r1->rect.y + distance &&
		   r2->rect.y >= r1->rect.y - distance &&
		   r2->rect.width <= (int)(r1->rect.width * 1.5 + 0.5) &&
		   (int)(r2->rect.width * 1.5 + 0.5) >= r1->rect.width;
}

ccv_array_t* ccv_dpm_detect_objects(ccv_dense_matrix_t* a, ccv_dpm_mixture_model_t** _model, int count, ccv_dpm_param_t params)
{
	int c, i, j, k, x, y;
	ccv_size_t size = ccv_size(a->cols, a->rows);
	for (c = 0; c < count; c++)
	{
		ccv_dpm_mixture_model_t* model = _model[c];
		for (i = 0; i < model->count; i++)
		{
			size.width = ccv_min(model->root[i].root.w->cols * 8, size.width);
			size.height = ccv_min(model->root[i].root.w->rows * 8, size.height);
		}
	}
	int hr = a->rows / size.height;
	int wr = a->cols / size.width;
	double scale = pow(2., 1. / (params.interval + 1.));
	int next = params.interval + 1;
	int scale_upto = (int)(log((double)ccv_min(hr, wr)) / log(scale)) - next;
	if (scale_upto < 0) // image is too small to be interesting
		return 0;
	ccv_dense_matrix_t** pyr = (ccv_dense_matrix_t**)alloca((scale_upto + next * 2) * sizeof(ccv_dense_matrix_t*));
	memset(pyr, 0, (scale_upto + next * 2) * sizeof(ccv_dense_matrix_t*));
	pyr[next] = a;
	for (i = 1; i <= params.interval; i++)
		ccv_resample(pyr[next], &pyr[next + i], 0, (int)(pyr[next]->rows / pow(scale, i)), (int)(pyr[next]->cols / pow(scale, i)), CCV_INTER_AREA);
	for (i = next; i < scale_upto + next; i++)
		ccv_sample_down(pyr[i], &pyr[i + next], 0, 0, 0);
	ccv_dense_matrix_t* hog = 0;
	/* a more efficient way to generate up-scaled hog (using smaller size) */
	for (i = 0; i < next; i++)
	{
		hog = 0;
		ccv_hog(pyr[i + next], &hog, 0, 9, CCV_DPM_WINDOW_SIZE / 2 /* this is */);
		pyr[i] = hog;
	}
	hog = 0;
	ccv_hog(pyr[next], &hog, 0, 9, CCV_DPM_WINDOW_SIZE);
	pyr[next] = hog;
	for (i = next + 1; i < scale_upto + next * 2; i++)
	{
		hog = 0;
		ccv_hog(pyr[i], &hog, 0, 9, CCV_DPM_WINDOW_SIZE);
		ccv_matrix_free(pyr[i]);
		pyr[i] = hog;
	}
	ccv_array_t* idx_seq;
	ccv_array_t* seq = ccv_array_new(64, sizeof(ccv_root_comp_t));
	ccv_array_t* seq2 = ccv_array_new(64, sizeof(ccv_root_comp_t));
	ccv_array_t* result_seq = ccv_array_new(64, sizeof(ccv_root_comp_t));
	for (c = 0; c < count; c++)
	{
		ccv_dpm_mixture_model_t* model = _model[c];
		double scale_x = 1.0;
		double scale_y = 1.0;
		for (i = next; i < scale_upto + next * 2; i++)
		{
			for (j = 0; j < model->count; j++)
			{
				ccv_dpm_root_classifier_t* root = model->root + j;
				ccv_dense_matrix_t* response = 0;
				ccv_filter(pyr[i], root->root.w, &response, 0, CCV_NO_PADDING);
				ccv_dense_matrix_t* root_feature = 0;
				ccv_flatten(response, (ccv_matrix_t**)&root_feature, 0, 0);
				ccv_matrix_free(response);
				int rwh = root->root.w->rows / 2;
				int rww = root->root.w->cols / 2;
				ccv_dense_matrix_t* part_feature[CCV_DPM_PART_MAX];
				ccv_dense_matrix_t* dx[CCV_DPM_PART_MAX];
				ccv_dense_matrix_t* dy[CCV_DPM_PART_MAX];
				for (k = 0; k < root->count; k++)
				{
					ccv_dpm_part_classifier_t* part = root->part + k;
					ccv_dense_matrix_t* response = 0;
					ccv_filter(pyr[i - next], part->w, &response, 0, CCV_NO_PADDING);
					ccv_dense_matrix_t* feature = 0;
					ccv_flatten(response, (ccv_matrix_t**)&feature, 0, 0);
					ccv_matrix_free(response);
					part_feature[k] = dx[k] = dy[k] = 0;
					ccv_distance_transform(feature, &part_feature[k], 0, &dx[k], 0, &dy[k], 0, part->dx, part->dy, part->dxx, part->dyy, CCV_NEGATIVE | CCV_GSEDT);
					ccv_matrix_free(feature);
					int offy = part->y + part->w->rows / 2 - rwh * 2;
					int miny = part->w->rows / 2, maxy = part_feature[k]->rows - part->w->rows / 2;
					int offx = part->x + part->w->cols / 2 - rww * 2;
					int minx = part->w->cols / 2, maxx = part_feature[k]->cols - part->w->cols / 2;
					float* f_ptr = (float*)ccv_get_dense_matrix_cell_by(CCV_32F | CCV_C1, root_feature, rwh, 0, 0);
					for (y = rwh; y < root_feature->rows - rwh; y++)
					{
						int iy = ccv_clamp(y * 2 + offy, miny, maxy);
						for (x = rww; x < root_feature->cols - rww; x++)
						{
							int ix = ccv_clamp(x * 2 + offx, minx, maxx);
							f_ptr[x] -= ccv_get_dense_matrix_cell_value_by(CCV_32F | CCV_C1, part_feature[k], iy, ix, 0);
						}
						f_ptr += root_feature->cols;
					}
				}
				float* f_ptr = (float*)ccv_get_dense_matrix_cell_by(CCV_32F | CCV_C1, root_feature, rwh, 0, 0);
				for (y = rwh; y < root_feature->rows - rwh; y++)
				{
					for (x = rww; x < root_feature->cols - rww; x++)
						if (f_ptr[x] + root->beta > params.threshold)
						{
							ccv_root_comp_t comp;
							comp.rect = ccv_rect((int)((x - rww) * CCV_DPM_WINDOW_SIZE * scale_x + 0.5), (int)((y - rwh) * CCV_DPM_WINDOW_SIZE * scale_y + 0.5), (int)(root->root.w->cols * CCV_DPM_WINDOW_SIZE * scale_x + 0.5), (int)(root->root.w->rows * CCV_DPM_WINDOW_SIZE * scale_y + 0.5));
							comp.id = c;
							comp.neighbors = 1;
							comp.confidence = f_ptr[x] + root->beta;
							comp.pnum = root->count;
							for (k = 0; k < root->count; k++)
							{
								ccv_dpm_part_classifier_t* part = root->part + k;
								comp.part[k].id = c;
								comp.part[k].neighbors = 1;
								int pww = part->w->cols / 2, pwh = part->w->rows / 2;
								int offy = part->y + pwh - rwh * 2;
								int offx = part->x + pww - rww * 2;
								int iy = ccv_clamp(y * 2 + offy, pwh, part_feature[k]->rows - pwh);
								int ix = ccv_clamp(x * 2 + offx, pww, part_feature[k]->cols - pww);
								int ry = iy - ccv_get_dense_matrix_cell_value_by(CCV_32S | CCV_C1, dy[k], iy, ix, 0);
								int rx = ix - ccv_get_dense_matrix_cell_value_by(CCV_32S | CCV_C1, dx[k], iy, ix, 0);
								comp.part[k].rect = ccv_rect((int)((rx - pww) * 4 * scale_x + 0.5), (int)((ry - pwh) * 4 * scale_y + 0.5), (int)(part->w->cols * 4 * scale_x + 0.5), (int)(part->w->rows * 4 * scale_y + 0.5));
								comp.part[k].confidence = ccv_get_dense_matrix_cell_value_by(CCV_32F | CCV_C1, part_feature[k], iy, ix, 0);
							}
							ccv_array_push(seq, &comp);
						}
					f_ptr += root_feature->cols;
				}
				for (k = 0; k < root->count; k++)
				{
					ccv_matrix_free(part_feature[k]);
					ccv_matrix_free(dx[k]);
					ccv_matrix_free(dy[k]);
				}
				ccv_matrix_free(root_feature);
			}
			scale_x *= scale;
			scale_y *= scale;
		}
		/* the following code from OpenCV's haar feature implementation */
		if(params.min_neighbors == 0)
		{
			for (i = 0; i < seq->rnum; i++)
			{
				ccv_root_comp_t* comp = (ccv_root_comp_t*)ccv_array_get(seq, i);
				ccv_array_push(result_seq, comp);
			}
		} else {
			idx_seq = 0;
			ccv_array_clear(seq2);
			// group retrieved rectangles in order to filter out noise
			int ncomp = ccv_array_group(seq, &idx_seq, _ccv_is_equal_same_class, 0);
			ccv_root_comp_t* comps = (ccv_root_comp_t*)ccmalloc((ncomp + 1) * sizeof(ccv_root_comp_t));
			memset(comps, 0, (ncomp + 1) * sizeof(ccv_root_comp_t));

			// count number of neighbors
			for(i = 0; i < seq->rnum; i++)
			{
				ccv_root_comp_t r1 = *(ccv_root_comp_t*)ccv_array_get(seq, i);
				int idx = *(int*)ccv_array_get(idx_seq, i);

				comps[idx].rect.x += r1.rect.x;
				comps[idx].rect.y += r1.rect.y;
				comps[idx].rect.width += r1.rect.width;
				comps[idx].rect.height += r1.rect.height;
				comps[idx].id = r1.id;
				comps[idx].pnum = r1.pnum;
				if (r1.confidence > comps[idx].confidence || comps[idx].neighbors == 0)
				{
					comps[idx].confidence = r1.confidence;
					memcpy(comps[idx].part, r1.part, sizeof(ccv_comp_t) * CCV_DPM_PART_MAX);
				}

				++comps[idx].neighbors;
			}

			// calculate average bounding box
			for(i = 0; i < ncomp; i++)
			{
				int n = comps[i].neighbors;
				if(n >= params.min_neighbors)
				{
					ccv_root_comp_t comp;
					comp.rect.x = (comps[i].rect.x * 2 + n) / (2 * n);
					comp.rect.y = (comps[i].rect.y * 2 + n) / (2 * n);
					comp.rect.width = (comps[i].rect.width * 2 + n) / (2 * n);
					comp.rect.height = (comps[i].rect.height * 2 + n) / (2 * n);
					comp.neighbors = comps[i].neighbors;
					comp.id = comps[i].id;
					comp.confidence = comps[i].confidence;
					comp.pnum = comps[i].pnum;
					memcpy(comp.part, comps[i].part, sizeof(ccv_comp_t) * CCV_DPM_PART_MAX);
					ccv_array_push(seq2, &comp);
				}
			}

			// filter out small face rectangles inside large face rectangles
			for(i = 0; i < seq2->rnum; i++)
			{
				ccv_root_comp_t r1 = *(ccv_root_comp_t*)ccv_array_get(seq2, i);
				int flag = 1;

				for(j = 0; j < seq2->rnum; j++)
				{
					ccv_root_comp_t r2 = *(ccv_root_comp_t*)ccv_array_get(seq2, j);
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

	for (i = 0; i < scale_upto + next * 2; i++)
		ccv_matrix_free(pyr[i]);

	ccv_array_free(seq);
	ccv_array_free(seq2);

	ccv_array_t* result_seq2;
	/* the following code from OpenCV's haar feature implementation */
	if (params.flags & CCV_DPM_NO_NESTED)
	{
		result_seq2 = ccv_array_new(64, sizeof(ccv_root_comp_t));
		idx_seq = 0;
		// group retrieved rectangles in order to filter out noise
		int ncomp = ccv_array_group(result_seq, &idx_seq, _ccv_is_equal, 0);
		ccv_root_comp_t* comps = (ccv_root_comp_t*)ccmalloc((ncomp + 1) * sizeof(ccv_root_comp_t));
		memset(comps, 0, (ncomp + 1) * sizeof(ccv_root_comp_t));

		// count number of neighbors
		for(i = 0; i < result_seq->rnum; i++)
		{
			ccv_root_comp_t r1 = *(ccv_root_comp_t*)ccv_array_get(result_seq, i);
			int idx = *(int*)ccv_array_get(idx_seq, i);

			if (comps[idx].neighbors == 0 || comps[idx].confidence < r1.confidence)
			{
				comps[idx].confidence = r1.confidence;
				comps[idx].neighbors = 1;
				comps[idx].rect = r1.rect;
				comps[idx].id = r1.id;
				comps[idx].pnum = r1.pnum;
				memcpy(comps[idx].part, r1.part, sizeof(ccv_comp_t) * CCV_DPM_PART_MAX);
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

	return result_seq2;
}

ccv_dpm_mixture_model_t* ccv_load_dpm_mixture_model(const char* directory)
{
	FILE* r = fopen(directory, "r");
	if (r == 0)
		return 0;
	int count;
	char flag;
	fscanf(r, "%c", &flag);
	assert(flag == '.');
	fscanf(r, "%d", &count);
	ccv_dpm_root_classifier_t* root_classifier = (ccv_dpm_root_classifier_t*)ccmalloc(sizeof(ccv_dpm_root_classifier_t) * count);
	memset(root_classifier, 0, sizeof(ccv_dpm_root_classifier_t) * count);
	int i, j, k;
	size_t size = sizeof(ccv_dpm_mixture_model_t) + sizeof(ccv_dpm_root_classifier_t) * count;
	/* the format is easy, but I tried to copy all data into one memory region */
	for (i = 0; i < count; i++)
	{
		int rows, cols;
		fscanf(r, "%d %d", &rows, &cols);
		fscanf(r, "%f", &root_classifier[i].beta);
		root_classifier[i].root.w = ccv_dense_matrix_new(rows, cols, CCV_32F | 31, ccmalloc(ccv_compute_dense_matrix_size(rows, cols, CCV_32F | 31)), 0);
		size += ccv_compute_dense_matrix_size(rows, cols, CCV_32F | 31);
		for (j = 0; j < rows * cols * 31; j++)
			fscanf(r, "%f", &root_classifier[i].root.w->data.f32[j]);
		ccv_make_matrix_immutable(root_classifier[i].root.w);
		fscanf(r, "%d", &root_classifier[i].count);
		ccv_dpm_part_classifier_t* part_classifier = (ccv_dpm_part_classifier_t*)ccmalloc(sizeof(ccv_dpm_part_classifier_t) * root_classifier[i].count);
		size += sizeof(ccv_dpm_part_classifier_t) * root_classifier[i].count;
		for (j = 0; j < root_classifier[i].count; j++)
		{
			fscanf(r, "%d %d %d", &part_classifier[j].x, &part_classifier[j].y, &part_classifier[j].z);
			fscanf(r, "%lf %lf %lf %lf", &part_classifier[j].dx, &part_classifier[j].dy, &part_classifier[j].dxx, &part_classifier[j].dyy);
			fscanf(r, "%d %d", &rows, &cols);
			part_classifier[j].w = ccv_dense_matrix_new(rows, cols, CCV_32F | 31, ccmalloc(ccv_compute_dense_matrix_size(rows, cols, CCV_32F | 31)), 0);
			size += ccv_compute_dense_matrix_size(rows, cols, CCV_32F | 31);
			for (k = 0; k < rows * cols * 31; k++)
				fscanf(r, "%f", &part_classifier[j].w->data.f32[k]);
			ccv_make_matrix_immutable(part_classifier[j].w);
		}
		root_classifier[i].part = part_classifier;
	}
	fclose(r);
	unsigned char* m = (unsigned char*)ccmalloc(size);
	ccv_dpm_mixture_model_t* model = (ccv_dpm_mixture_model_t*)m;
	m += sizeof(ccv_dpm_mixture_model_t);
	model->count = count;
	model->root = (ccv_dpm_root_classifier_t*)m;
	m += sizeof(ccv_dpm_root_classifier_t) * model->count;
	memcpy(model->root, root_classifier, sizeof(ccv_dpm_root_classifier_t) * model->count);
	for (i = 0; i < model->count; i++)
	{
		ccv_dpm_part_classifier_t* part_classifier = model->root[i].part;
		model->root[i].part = (ccv_dpm_part_classifier_t*)m;
		m += sizeof(ccv_dpm_part_classifier_t) * model->root[i].count;
		memcpy(model->root[i].part, part_classifier, sizeof(ccv_dpm_part_classifier_t) * model->root[i].count);
	}
	for (i = 0; i < model->count; i++)
	{
		ccv_dense_matrix_t* w = model->root[i].root.w;
		model->root[i].root.w = (ccv_dense_matrix_t*)m;
		m += ccv_compute_dense_matrix_size(w->rows, w->cols, w->type);
		memcpy(model->root[i].root.w, w, ccv_compute_dense_matrix_size(w->rows, w->cols, w->type));
		model->root[i].root.w->data.u8 = (unsigned char*)(model->root[i].root.w + 1);
		ccfree(w);
		for (j = 0; j < model->root[i].count; j++)
		{
			w = model->root[i].part[j].w;
			model->root[i].part[j].w = (ccv_dense_matrix_t*)m;
			m += ccv_compute_dense_matrix_size(w->rows, w->cols, w->type);
			memcpy(model->root[i].part[j].w, w, ccv_compute_dense_matrix_size(w->rows, w->cols, w->type));
			model->root[i].part[j].w->data.u8 = (unsigned char*)(model->root[i].part[j].w + 1);
			ccfree(w);
		}
	}
	return model;
}

void ccv_dpm_mixture_model_free(ccv_dpm_mixture_model_t* model)
{
	ccfree(model);
}
