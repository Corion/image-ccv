#!perl -w
use strict;
use Imager;
use Imager::Fill;
use List::Util qw(max);
use Image::CCV qw(detect_faces);

#my $scene  = "images/face_IMG_0762_bw.png";
#my $scene  = "images/2123317682_9e93436f77.png";
#my $scene  = "images/IMG_0766-bw.png";
my $scene = "images/IMG_0732_bw.png";

my @coords = detect_faces( $scene );
#print "@$_\n" for @coords;

my $out = Imager->new( file => $scene );

for (@coords) {
    my ($x,$y,$width,$height,$confidence) = @$_;
    warn "($x,$y): ${width}x$height @ $confidence\n";
    my $color = Imager::Color->new( (1-$confidence/100) *255, $confidence/100 *255, 0 );
    
    # Draw a nice box
    $out->box(
        color => $color,
        xmin => $x,
        ymin => $y,
        xmax => $x+$width,
        ymax => $y+$height,
        aa => 1,
    );
};

$out->write( file => 'face.png' )
    or die $out->errstr;