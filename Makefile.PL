# -*- mode: perl; c-basic-offset: 4; indent-tabs-mode: nil; -*-

use strict;
use ExtUtils::MakeMaker qw(WriteMakefile);
use Config;
# See lib/ExtUtils/MakeMaker.pm for details of how to influence
# the contents of the Makefile that is written.


# TODO: Add command line switches in addition to %ENV and %Config

warn <<ALPHA_WARNING;
*** ALPHA VERSION!
*** Currently Image::CCV blindly assumes that you have
*** libjpeg and libpng available. If you don't have
*** them  available, things will simply fail badly for you.
ALPHA_WARNING

my @tests = map { glob $_ } 't/*.t', 't/*/*.t';
my $module = 'Image::CCV';
(my $module_pm = "lib/$module.pm") =~ s!::!/!g;

# I should maybe use something like Shipwright...
#regen_README($module_pm);
regen_EXAMPLES();

my $ccflags = $ENV{CCFLAGS}
            || $Config::Config{ ccflags }
            ;

# TODO: Add a probe for png.h
# TODO: Add a probe for jpeg.h
# TODO: Add a probe for alloca.h
my $LIBS = $ENV{LIBS} 
           || '-ljpeg -lpng'
           ;

my @LIBS = ($LIBS);

if( $^O =~ /mswin32/i ) {
    # I'm sorry, but I don't know how else get -lws2_32 to link :-/
    $LIBS = "$LIBS -LC:\\strawberry\\c\\i686-w64-mingw32\\lib -lws2_32";
    #push @LIBS, ":nosearch $LIBS -lws2_32";
};
warn "LIBS: $LIBS\n";

WriteMakefile1(
    MIN_PERL_VERSION => '5.006',
    META_MERGE => {
        resources => {
            repository => 'http://github.com/Corion/image-ccv',
        },
    },
    BUILD_REQUIRES => {
                        'DynaLoader' => '0',
                        'Exporter'   => '0',
                        'Carp'       => '0',
    },
    'NAME'          => $module,
    'LICENSE'       => 'perl',
    'VERSION_FROM'  => $module_pm,
    'PREREQ_PM'     => {
                       'Imager' => 0.78,
                       'Imager::File::PNG' => 0, # Our tests want to read/write PNG
                       'Imager::File::JPEG' => 0, # just to make sure you have libjpeg somewhere
                        'List::Util' => '0', # just for the example programs
                      }, # e.g., Module::Name => 1.1
    'TYPEMAPS' => [
        #'C:\\strawberry\\perl\\lib\\ExtUtils\\typemap',
        'typemap'
    ],
    'DEFINE'  => '-DHAVE_ZLIB -DHAVE_LIBJPEG -DHAVE_LIBPNG',
    'OBJECT'  => '$(O_FILES)', # link all the C files too
    #'CCFLAGS' => "-msse2 -mms-bitfields $ccflags",
    'CCFLAGS' => "-msse2 $ccflags",
    #'INC' => '-I"C:/Projekte/CCV"',
    'LIBS' => [
        $LIBS,
    ],
    ABSTRACT_FROM   => $module_pm, # retrieve abstract from module
    AUTHOR          => 'Max Maischein <corion@cpan.org>',
    test            => { TESTS => join( ' ', @tests ) },
);

1;

sub WriteMakefile1 {  #Written by Alexandr Ciornii, version 0.21. Added by eumm-upgrade.
    my %params=@_;
    my $eumm_version=$ExtUtils::MakeMaker::VERSION;
    $eumm_version=eval $eumm_version;
    die "EXTRA_META is deprecated" if exists $params{EXTRA_META};
    die "License not specified" if not exists $params{LICENSE};
    if ($params{BUILD_REQUIRES} and $eumm_version < 6.5503) {
        #EUMM 6.5502 has problems with BUILD_REQUIRES
        $params{PREREQ_PM}={ %{$params{PREREQ_PM} || {}} , %{$params{BUILD_REQUIRES}} };
        delete $params{BUILD_REQUIRES};
    }
    delete $params{CONFIGURE_REQUIRES} if $eumm_version < 6.52;
    delete $params{MIN_PERL_VERSION} if $eumm_version < 6.48;
    delete $params{META_MERGE} if $eumm_version < 6.46;
    delete $params{META_ADD} if $eumm_version < 6.46;
    delete $params{LICENSE} if $eumm_version < 6.31;
    delete $params{AUTHOR} if $] < 5.005;
    delete $params{ABSTRACT_FROM} if $] < 5.005;
    delete $params{BINARY_LOCATION} if $] < 5.005;
    
    WriteMakefile(%params);
}

sub regen_README {
    eval {
        require Pod::Readme;

        my $parser = Pod::Readme->new();

        # Read POD from Module.pm and write to README
        $parser->parse_from_file($_[0], 'README');
    };
    eval {
        require Pod::Markdown;

        my $parser = Pod::Markdown->new();

        # Read POD from Module.pm and write to README
        $parser->parse_from_file($_[0]);
        open my $fh, '>', 'README.mkdn'
            or die "Couldn't open 'README.mkdn': $!";
        print $fh $parser->as_markdown;
    };
}


sub regen_EXAMPLES {
    my $perl = $^X;
    (my $target = $module_pm) =~ s!\.pm$!/Examples.pm!;
    
    if( -d 'examples' ) {
        warn "(Re)Creating $target\n";
        if ($perl =~/\s/) {
            $perl = qq{"$perl"};
        };
        my $examples = `$perl -w examples/gen_examples_pod.pl`;
        if ($examples) {
            $examples =~ s/\r\n/\n/g;
            if (open my $out, '>', $target) {
                binmode $out;
                print $out $examples;
            };
        };
    };
};