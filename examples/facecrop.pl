#!perl -w
use strict;
use Getopt::Long;
use Pod::Usage;
use List::Util qw(max);
use Imager;
use Imager::Fill;
use Image::CCV qw(detect_faces);

use vars qw($VERSION);
$VERSION = '0.02';

=head1 NAME

facecrop.pl - create crop from image using the largest face area

=head1 SYNTAX

  facecrop.pl filename.png

  facecrop.pl filename.png -o thumb_filename.png

=cut

GetOptions(
    'o:s'        => \my $out_file,
    'width|w:s'  => \my $max_width,
    'height|h:s' => \my $max_height,
    'scale|s:s'  => \my $scale,
) or pod2usage();
$scale ||= 1.5; # default chosen by wild guess

for my $scene (@ARGV) {
    my @coords = detect_faces( $scene );
    if(! @coords) {
        die "No face found\n";
    };

    # Now, find the largest face (area) in this image
    # We ignore the confidence value
    my $max = $coords[0];
    for (@coords) {
        if( $_->[2] * $_->[3] > $max->[2] * $max->[3] ) {
            $max = $_
        }
    };
    
    if( $out_file ) {
        # Scale the frame a bit up
        my $w = $max->[2] * $scale;
        my $h = $max->[3] * $scale;
        my $l = max( 0, $max->[0] - $max->[2]*(($scale -1) / 2));
        my $t = max( 0, $max->[1] - $max->[3]*(($scale -1) / 2) );
        my $out = Imager->new( file => $scene )
                        ->crop( left => $l, top => $t,
                                width => $w, height => $h
                              );
        if( $max_width || $max_height ) {
            $max_width  ||= $max_height;
            $max_height ||= $max_width;
            $out = $out->scale(
                xpixels => $max_width,
                ypixels => $max_height,
                type => 'nonprop'
            );
        };
        $out->write( file => $out_file )
            or die $out->errstr;
    } else {
        my ($x,$y,$width,$height,$confidence) = @$max;
        print "($x,$y): ${width}x$height @ $confidence\n";
    }
}