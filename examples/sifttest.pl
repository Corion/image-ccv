#!perl -w
use strict;
use Imager;
use Imager::Fill;
use List::Util qw(max);
use Image::CCV qw(sift);

use vars qw($VERSION);
$VERSION = '0.03';

my $scene  = "images/IMG_1229_bw_small.png";
my $object = "images/IMG_1230_bw_sofa.png";

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