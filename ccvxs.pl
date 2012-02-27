#!perl -w
use strict;
use Imager;
use Imager::Fill;
use List::Util qw(max);

use vars qw($VERSION);
$VERSION = '0.01';

# TODO: Add face recognition wrapper
# TODO: Make ccv_array_t into a class, so automatic destruction works
# TODO: ccv_sift_param_t currently leaks. Add a DESTROY method.
# TODO: Turn C code into XS

#include "ccv_amalgamated.c"
use Inline
    C => <<'CCV',
#include "ccv/lib/3rdparty/sha1.h"
#include "ccv/lib/3rdparty/sha1.c"
#include "ccv/lib/ccv.h"
#include "ccv/lib/ccv_basic.c"
#include "ccv/lib/ccv_algebra.c"
#include "ccv/lib/ccv_cache.c"
#include "ccv/lib/ccv_memory.c"
#include "ccv/lib/ccv_util.c"
#include "ccv/lib/ccv_io.c"
#include "ccv/lib/ccv_sift.c"
/**/
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

my $scene  = "IMG_1229_bw_small.png";
my $object = "IMG_1230_bw_sofa.png";
#my $scene  = "ccv/samples/scene.png";
#my $object = "ccv/samples/basmati.png";

#my $scene  = "IMG_0766_small_bw.png";
#my $scene = "onion-skew-240x253.png";
#my $scene = "4585979660_f9a4c39cf5.png";
#my $scene = "2123317682_9e93436f77.png";
#my $object = "4585979660_f9a4c39cf5.png";
#my $object = "onion-240x240.png";
#my $scene = "IMG_0766-bw.png";
#my $object = "yapceu-logo200-bw.png";

#my $scene = "IMG_4026_bw.png";
#my $object = "IMG_4031_bw.png";

my @coords = sift( $object, $scene, );
print "@$_\n" for @coords;

my $scene_image = Imager->new( file => $scene );
my $object_image = Imager->new( file => $object );

my $xsize = $scene_image->getwidth + $object_image->getwidth;
my $ysize = max( $scene_image->getheight, $object_image->getheight);

my $out = Imager->new(
    xsize => $xsize,
    ysize => $ysize,
);

# paste the two input images side by side
$out->rubthrough(
    src => $scene_image,
    tx => 0, ty => 0,
    src_minx => 0,
    src_maxx => $scene_image->getwidth-1,
    src_miny => 0,
    src_maxy => $scene_image->getheight-1,
);

my $obj_ofs_x = $scene_image->getwidth;
my $obj_ofs_y = 0;

$out->rubthrough(
    src => $object_image,
    tx => $obj_ofs_x, ty => $obj_ofs_y,
    src_minx => 0,
    src_maxx => $object_image->getwidth-1,
    src_miny => 0,
    src_maxy => $object_image->getheight-1,
);

my @points = @coords;

my $green = Imager::Color->new( 0, 255, 0 );
for (@points) {
    $out->line(
        color => $green,
        x1 => $_->[0]+$obj_ofs_x,
        y1 => $_->[1]+$obj_ofs_y,
        x2 => $_->[2],
        y2 => $_->[3],
    );
};

$out->write( file => 'out.png' )
    or die $out->errstr;