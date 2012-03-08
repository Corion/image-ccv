#!perl -w
use strict;
use warnings;

# This is mainly a safeguard-test to check that the hardcoded
# class names XS/Inline::C generate match up with what my
# Perl code expects
use Test::More tests => 2;

use Image::CCV;

my @faces = detect_faces('t/face_IMG_0762_bw_small.png');

is 0+@faces, 1, "We find one face";

is_deeply $faces[0], [
  '37',
  '33',
  '26',
  '26',
  '5.34655570983887'
], "We detect the face at the expected co-ordinates";

# Most likely, this should be more lenient, especially towards
# the confidence value