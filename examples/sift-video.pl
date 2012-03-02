#!perl -w
use strict;
use Imager;
use Imager::Fill;
use List::Util qw(max);
use Image::CCV qw(sift);
use Getopt::Long;

use vars qw($VERSION);
$VERSION = '0.01';

my $scene  = "images/IMG_1229_bw_small.png";
my $workdir;

# Get number of frames in input video

my %processed;
my @images;
my @output_images;

for my $number (0..$ffmpeg->frames) {
    my $name = File::Spec->catfile( $workdir, $number );
    # Extract the next frame
    $ffmpeg->extract_frame( $number, $name );

    # Process the latest frame in respect to all other frames
    my $result = process_frame( $name, \@images );
    $result->write(...);
    push @output_images, $result_name;

    push @images, $name;
    # Remove all frames that don't fit in the current processing window
    unlink splice @images, $frames;
}

$ffmpeg_out->combine(@output_images);