package Image::CCV;
use Exporter 'import';
use vars qw($VERSION @EXPORT);
BEGIN {  # for Inline.pm, below
	$VERSION = '0.01'; 
};

@EXPORT = qw(sift detect_faces );
=head1 NAME

Image::CCV - Crazy-cool Computer Vision bindings for Perl
=cut
# TODO: Make ccv_array_t into a class, so automatic destruction works
# TODO: ccv_sift_param_t currently leaks. Add a DESTROY method.
# TODO: Add tests
# TODO: Add Troubleshooting.pm
# TODO: Add FAQ.pm
# TODO: Add Examples.pm

#include "ccv_amalgamated.c"
use Inline
    C => <<'CCV',
/* Make the ccv library conveniently available */
#include "ccv-src/lib/3rdparty/sha1.h"
#include "ccv-src/lib/3rdparty/sha1.c"
#include "ccv-src/lib/ccv.h"
#include "ccv-src/lib/ccv_basic.c"
#include "ccv-src/lib/ccv_algebra.c"
#include "ccv-src/lib/ccv_cache.c"
#include "ccv-src/lib/ccv_memory.c"
#include "ccv-src/lib/ccv_util.c"
#include "ccv-src/lib/ccv_io.c"
#include "ccv-src/lib/ccv_sift.c"
#include "ccv-src/lib/ccv_bbf.c"

#include <ctype.h>

void
detect_faces(char* filename, char* training_data)
{
	Inline_Stack_Vars;
	Inline_Stack_Reset;
	int i;
	ccv_enable_default_cache();
	ccv_dense_matrix_t* image = 0;
	ccv_bbf_classifier_cascade_t* cascade = ccv_load_bbf_classifier_cascade(training_data);
	ccv_unserialize(filename, &image, CCV_SERIAL_GRAY | CCV_SERIAL_ANY_FILE);
	if (image != 0)
	{
		ccv_bbf_param_t params = { .interval = 5, .min_neighbors = 2, .flags = 0, .size = ccv_size(24, 24) };
		ccv_array_t* seq = ccv_bbf_detect_objects(image, &cascade, 1, params);
		for (i = 0; i < seq->rnum; i++)
		{
			ccv_comp_t* comp = (ccv_comp_t*)ccv_array_get(seq, i);
			// Create the new 5-item array
			AV* res = newAV();
			av_push( res, newSVnv( comp->rect.x ));
			av_push( res, newSVnv( comp->rect.y ));
			av_push( res, newSVnv( comp->rect.width ));
			av_push( res, newSVnv( comp->rect.height ));
			av_push( res, newSVnv( comp->confidence ));
                        Inline_Stack_Push(sv_2mortal(newRV_noinc((SV*) res)));
		}
		ccv_array_free(seq);
		ccv_matrix_free(image);
	}
	ccv_bbf_classifier_cascade_free(cascade);
	ccv_disable_cache();
	Inline_Stack_Done;
	return;
}

ccv_sift_param_t* myccv_pack_parameters(int noctaves, int nlevels, int up2x, int edge_threshold, int norm_threshold, int peak_threshold)
{
	ccv_sift_param_t* res;
	res = malloc(sizeof(*res));
	
	res->noctaves = noctaves;
	res->nlevels = nlevels;
	res->up2x = up2x;
	res->edge_threshold = edge_threshold;
	res->norm_threshold = norm_threshold;
	res->peak_threshold = peak_threshold;
	
	return res;
}

/* Should this just become a tiearray interface?! */
void myccv_keypoints_to_list(ccv_array_t* keypoints)
{
      Inline_Stack_Vars;
      Inline_Stack_Reset;

      AV* res = newAV();
      int i;
      for (i = 0; i < keypoints->rnum; i++) {
          ccv_keypoint_t* kp = (ccv_keypoint_t*)ccv_array_get(keypoints, i);
          AV* point = newAV();
          
          av_push( point, newSVnv( kp->x ));
          av_push( point, newSVnv( kp->y ));
      };
      
      Inline_Stack_Push(sv_2mortal(newRV_noinc((SV*) res)));
      Inline_Stack_Done;
      return;
}

void myccv_get_descriptor(char* file, ccv_sift_param_t* param)
{
	Inline_Stack_Vars;
	Inline_Stack_Reset;

	ccv_dense_matrix_t* data = 0;
	ccv_unserialize(file, &data, CCV_SERIAL_GRAY | CCV_SERIAL_ANY_FILE);
	assert(data);
	
	ccv_array_t* keypoints = 0;
	ccv_dense_matrix_t* descriptor = 0;
	ccv_sift(data, &keypoints, &descriptor, 0, *param);

	/* XXX We should blesss those into proper classes for automatic deallocation */
        Inline_Stack_Push(sv_2mortal(newSVpv((void *)descriptor,0)));
	Inline_Stack_Push(sv_2mortal(newSVpv((void *)keypoints,0)));
	
	Inline_Stack_Done;
	return;
}

void myccv_sift(char* object_file, char* scene_file, ccv_sift_param_t* param)
{
        Inline_Stack_Vars;
        Inline_Stack_Reset;

	ccv_enable_default_cache();
	ccv_dense_matrix_t* object = 0;
	ccv_dense_matrix_t* image = 0;
	ccv_unserialize(object_file, &object, CCV_SERIAL_GRAY | CCV_SERIAL_ANY_FILE);
	assert(object);
	ccv_unserialize(scene_file, &image, CCV_SERIAL_GRAY | CCV_SERIAL_ANY_FILE);
	assert(image);
	ccv_array_t* obj_keypoints = 0;
	ccv_dense_matrix_t* obj_desc = 0;
	ccv_sift(object, &obj_keypoints, &obj_desc, 0, *param);
	ccv_array_t* image_keypoints = 0;
	ccv_dense_matrix_t* image_desc = 0;
	ccv_sift(image, &image_keypoints, &image_desc, 0, *param);
	int i, j, k;
	int match = 0;
	for (i = 0; i < obj_keypoints->rnum; i++)
	{
		float* odesc = obj_desc->data.fl + i * 128;
		int minj = -1;
		double mind = 1e6, mind2 = 1e6;
		for (j = 0; j < image_keypoints->rnum; j++)
		{
			float* idesc = image_desc->data.fl + j * 128;
			double d = 0;
			for (k = 0; k < 128; k++)
			{
				d += (odesc[k] - idesc[k]) * (odesc[k] - idesc[k]);
				if (d > mind2)
					break;
			}
			if (d < mind)
			{
				mind2 = mind;
				mind = d;
				minj = j;
			} else if (d < mind2) {
				mind2 = d;
			}
		}
		if (mind < mind2 * 0.36)
		{
			ccv_keypoint_t* op = (ccv_keypoint_t*)ccv_array_get(obj_keypoints, i);
			ccv_keypoint_t* kp = (ccv_keypoint_t*)ccv_array_get(image_keypoints, minj);
			// Create the new 4-item array
			AV* res = newAV();
			av_push( res, newSVnv( op->x ));
			av_push( res, newSVnv( op->y ));
			av_push( res, newSVnv( kp->x ));
			av_push( res, newSVnv( kp->y ));
                        Inline_Stack_Push(sv_2mortal(newRV_noinc((SV*) res)));
			// printf("%f %f => %f %f\n", op->x, op->y, kp->x, kp->y);
			match++;
		}
	}
	printf("%dx%d on %dx%d\n", object->cols, object->rows, image->cols, image->rows);
	printf("%d keypoints out of %d are matched\n", match, obj_keypoints->rnum);
	ccv_array_free(obj_keypoints);
	ccv_array_free(image_keypoints);
	ccv_matrix_free(obj_desc);
	ccv_matrix_free(image_desc);
	ccv_matrix_free(object);
	ccv_matrix_free(image);
	ccv_disable_cache();
	Inline_Stack_Done;
	return;
}
// 14
CCV
    INC => '-Ic:/Projekte/CCV/ccv/lib',
    #LIBS => '-Lc:/Projekte/CCV/ccv/lib -Lc:/strawberry/c/i686-w64-mingw32/lib -lccv -ljpeg -lpng -lws2_32',
    LIBS => '-ljpeg -lpng -lws2_32',
    CCFLAGS => ' -mms-bitfields -O2 -msse2 -DHAVE_ZLIB -DHAVE_LIBJPEG -DHAVE_LIBPNG',
    NAME => __PACKAGE__,
    VERSION => $VERSION,
    ;

sub default_ccv_params {
    my ($params) = @_;
    $params ||= {};

    my %default = (
	noctaves => 5,
	nlevels => 5,
	up2x => 1,
	edge_threshold => 5,
	norm_threshold => 0,
	peak_threshold => 0,
    );
    
    for (keys %default) {
    	if(! exists $params->{ $_ }) {
            $params->{ $_ } = $default{ $_ }
    	};
    };
    
    if( ref $params ne 'ccv_sift_param_tPtr') {
    	$params = myccv_pack_parameters(
    	    @{$params}{qw<
    	        noctaves
    	        nlevels
    	        up2x
    	        edge_threshold
    	        norm_threshold
    	        peak_threshold
    	    >}
    	);
    };
    
    $params
};

sub get_sift_descriptor {
    my ($filename, $params) = @_;
    
    $params = default_ccv_params( $params );
    
    my ($keypoints, $descriptor) = myccv_get_descriptor($filename);
    return {
    	keypoints => $keypoints,
    	descriptor => $descriptor,
    }
}

sub sift {
    my ($object, $scene, $params) = @_;
    
    $params = default_ccv_params( $params );
    
    myccv_sift( $object, $scene, $params);
};
1;

=head1 REPOSITORY

The public repository of this module is 
L<http://github.com/Corion/image-ccv>.

=head1 SUPPORT

The public support forum of this module is
L<http://perlmonks.org/>.

=head1 TALKS

I've given one lightning talk about this module at Perl conferences:

L<http://corion.net/talks/Image-CCV-lightning/Image-CCV-lightning.html|German Perl Workshop, German>

=head1 BUG TRACKER

Please report bugs in this module via the RT CPAN bug queue at
L<https://rt.cpan.org/Public/Dist/Display.html?Name=Image-CCV>
or via mail to L<image-ccv-Bugs@rt.cpan.org>.

=head1 AUTHOR

Max Maischein C<corion@cpan.org>

=head1 COPYRIGHT (c)

Copyright 2009-2012 by Max Maischein C<corion@cpan.org>.

=head1 LICENSE

This module is released under the same terms as Perl itself. The CCV library
distributed with it comes with its own license(s). Please study these
before redistributing.

=cut
