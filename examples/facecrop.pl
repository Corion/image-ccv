#!perl -w
use strict;
use Getopt::Long;
use Pod::Usage;
use List::Util qw(max);
use Imager;
use Imager::Fill;
use Image::CCV qw(detect_faces);

use vars qw($VERSION);
$VERSION = '0.03';

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
    'largest'    => \my $only_largest,
    'draw-box'   => \my $draw_box,
    'verbose'    => \my $verbose,
) or pod2usage();
$scale ||= 1.5; # default chosen by wild guess

for my $scene (@ARGV) {
    my @coords = detect_faces( $scene );
    if(! @coords) {
        die "No face found\n";
    };

    if( $only_largest ) {
        # Now, find the largest face (area) in this image
        # We ignore the confidence value
        my $max = $coords[0];
        for (@coords) {
            if( $_->[2] * $_->[3] > $max->[2] * $max->[3] ) {
                $max = $_
            }
        };
        @coords = ($max);
    };
    
    if( $verbose ) {
        print sprintf "%d Gesichter gefunden\n", 0+@coords;
    };
    
    my $index = 1;
    for my $face (@coords) {
        if( $out_file ) {
            my $out = Imager->new( file => $scene );            
            my ($x,$y,$width,$height,$confidence) = @$face;
            
            if( $draw_box ) {
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
            
            # Scale the frame a bit up
            my $w = $face->[2] * $scale;
            my $h = $face->[3] * $scale;
            my $l = max( 0, $face->[0] - $face->[2]*(($scale -1) / 2));
            my $t = max( 0, $face->[1] - $face->[3]*(($scale -1) / 2) );
            
            $out = $out->crop( 
                       left => $l, top => $t,
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
            
            my $out_name = sprintf $out_file, $index++;
            $out->write( file => $out_name )
                or die $out->errstr;
            print "$out_name\n";
            
        } else {
            my ($x,$y,$width,$height,$confidence) = @$face;
            print "($x,$y): ${width}x$height @ $confidence\n";
        }
    }
}