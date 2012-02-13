#!perl -w
use strict;
use Imager;
use Imager::Fill;
use List::Util qw(max);

use vars qw($VERSION);
$VERSION = '0.01';

use Inline
    C => <<'CCV',
#include "ccv.h"
void sift(char* object_file, char* scene_file)
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
	// unsigned int elapsed_time = get_current_time();
	ccv_sift_param_t param;
	param.noctaves = 3;
	param.nlevels = 8;
	param.up2x = 1;
	param.edge_threshold = 10;
	param.norm_threshold = 1;
	param.peak_threshold = 1;
	ccv_array_t* obj_keypoints = 0;
	ccv_dense_matrix_t* obj_desc = 0;
	ccv_sift(object, &obj_keypoints, &obj_desc, 0, param);
	ccv_array_t* image_keypoints = 0;
	ccv_dense_matrix_t* image_desc = 0;
	ccv_sift(image, &image_keypoints, &image_desc, 0, param);
	// elapsed_time = get_current_time() - elapsed_time;
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
			av_push( res, newSViv( op->x ));
			av_push( res, newSViv( op->y ));
			av_push( res, newSViv( kp->x ));
			av_push( res, newSViv( kp->y ));
                        Inline_Stack_Push(sv_2mortal(newRV_noinc((SV*) res)));
			// printf("%f %f => %f %f\n", op->x, op->y, kp->x, kp->y);
			match++;
		}
	}
	//printf("%dx%d on %dx%d\n", object->cols, object->rows, image->cols, image->rows);
	//printf("%d keypoints out of %d are matched\n", match, obj_keypoints->rnum);
	//printf("elpased time : %d\n", elapsed_time);
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

CCV
    INC => '-Ic:/Projekte/CCV/ccv/lib',
    LIBS => '-Lc:/Projekte/CCV/ccv/lib -Lc:/strawberry/c/i686-w64-mingw32/lib -lccv -ljpeg -lpng -lws2_32',
    CCFLAGS => '-msse2',
    ;


my $ccv_base = "ccv";
#my $scene  = "onion-skew-240x253.png";
my $scene  = "IMG_1229_bw_small.png";
my $object = "IMG_1230_bw_sofa.png";
#my $scene  = "$ccv_base/samples/scene.png";
#my $object = "$ccv_base/samples/basmati.png";
my $ccv = "$ccv_base/bin/siftmatch.exe";

my @coords = sift( $object, $scene );
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

# Now draw the connections between the sifted points
#my @points = map { /^([\d.]+) ([\d.]+) => ([\d.]+) ([\d.]+)$/ or die $_; [$1,$2,$3,$4]} grep { /=>/} @coords;
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