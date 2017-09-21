#!perl -w
use strict;
use ExtUtils::MakeMaker qw(WriteMakefile);
use Config;

use lib qw(inc);
use Devel::CheckLib;

# See lib/ExtUtils/MakeMaker.pm for details of how to influence
# the contents of the Makefile that is written.

# Normalize version strings like 6.30_02 to 6.3002,
# so that we can do numerical comparisons on it.
my $eumm_version = $ExtUtils::MakeMaker::VERSION;
$eumm_version =~ s/_//;

# Prompt the user here for any paths and other configuration

my @tests = map { glob $_ } 't/*.t', 't/*/*.t';
my $module = 'Image::CCV';
(my $main_file = "lib/$module.pm" ) =~ s!::!/!g;
(my $distbase = $module) =~ s!::!-!g;

# I should maybe use something like Shipwright...
#regen_README($main_file);
regen_EXAMPLES();

my $ccflags = $ENV{CCFLAGS}
            || $Config::Config{ ccflags }
            ;

# TODO: Add command line switches in addition to %ENV and %Config
#       Thanks to Devel::CheckLib, LIBS= and INC= already exist
# TODO: Move OS checking to Devel::CheckOS

my $DEBUG = 0;

my $is_Win32 = ($^O =~ /mswin32/i );

check_lib_or_exit( header => 'png.h', lib => 'png', function => '(void)&png_create_write_struct; return 0;',
    debug => $DEBUG,
);
check_lib_or_exit( header => ['stdio.h', 'jpeglib.h'], lib => 'jpeg' );

if(! $is_Win32) {
    check_lib_or_exit( header => 'alloca.h', function => '(void)&alloca; return 0;' );
};

my @LIBS = split /\s+/, $ENV{LIBS};
@LIBS = qw(-ljpeg -lpng) unless @LIBS;
           ;

if( $is_Win32 ) {    
    # I'm sorry, but I don't know how else get -lws2_32 to link :-/
    #$LIBS = "$LIBS"; # 
    push @LIBS, '-lws2_32';
    
};
# These should be the default everywhere, but aren't on my versions of Perl
# (strawberry and Debian ./Configure -de)
$ccflags = "-msse2 -fPIC $ccflags";

#my @LIBS = ($LIBS);

# -*- mode: perl; c-basic-offset: 4; indent-tabs-mode: nil; -*-

my %module = (
    NAME                => $module,
    AUTHOR              => q{Max Maischein <corion@cpan.org>},
    VERSION_FROM        => $main_file,
    ABSTRACT_FROM       => $main_file,
    META_MERGE => {
        "meta-spec" => { version => 2 },
        resources => {
            repository => {
                web => 'https://github.com/Corion/image-ccv',
                url => 'git://github.com/Corion/image-ccv.git',
                type => 'git',
            }
        },
        dynamic_config   => 0, # we promise to keep META.* up-to-date
        x_static_install => 0, # we need a C compiler and run stuff on install
    },

    'LICENSE'=> 'perl',

    MIN_PERL_VERSION => '5.006',
    BUILD_REQUIRES => {
                        'DynaLoader' => '0',
                        'Exporter'   => '0',
                        'Carp'       => '0',
                        'warnings'   => '0',
                        'ExtUtils::MakeMaker' => 0,
    },
    CONFIGURE_REQUIRES => {
        'Devel::CheckLib' => 0,
    },
    'NAME'          => $module,
    'LICENSE'       => 'perl',
    'VERSION_FROM'  => $main_file,
    'PREREQ_PM'     => {
                       'Imager' => 0.78,
                        'List::Util' => '0', # just for the example programs
                      }, # e.g., Module::Name => 1.1
    'TYPEMAPS' => [
        #'C:\\strawberry\\perl\\lib\\ExtUtils\\typemap',
        'typemap'
    ],
    'DEFINE'  => '-DHAVE_ZLIB -DHAVE_LIBJPEG -DHAVE_LIBPNG',
    'OBJECT'  => '$(O_FILES)', # link all the C files too
    #'CCFLAGS' => "-msse2 -mms-bitfields $ccflags",
    'CCFLAGS' => $ccflags,
    #'INC' => '-I"C:/Projekte/CCV"',
    'LIBS' => join( " ", @LIBS),
    ABSTRACT_FROM   => $main_file, # retrieve abstract from module
    AUTHOR          => 'Max Maischein <corion@cpan.org>',
    test            => { TESTS => join( ' ', @tests ) },
);

# This is so that we can do
#     require 'Makefile.PL'
# and then call get_module_info

sub get_module_info { %module }

if( ! caller ) {
    # I should maybe use something like Shipwright...
    regen_README($main_file);
    regen_EXAMPLES();
    WriteMakefile1(get_module_info);
};

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
    (my $target = $main_file) =~ s!\.pm$!/Examples.pm!;
    
    if( -d 'examples' ) {
        warn "(Re)Creating $target\n";
        if ($perl =~/\s/) {
            $perl = qq{"$perl"};
        };
        my $examples = `$perl -w examples/gen_examples_pod.pl $module`;
        if ($examples) {
            $examples =~ s/\r\n/\n/g;
            if (open my $out, '>', $target) {
                binmode $out;
                print $out $examples;
            };
        };
    };
};
