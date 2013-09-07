<<<<<<< HEAD
include config.mk

#CC += -faddress-sanitizer -fno-omit-frame-pointer
CFLAGS := -O3 -ffast-math -Wall $(CFLAGS)# -fprofile-arcs -ftest-coverage

.PHONY: all clean
=======
CC = `cat .cc.conf`# -faddress-sanitizer -fno-omit-frame-pointer
CFLAGS = -O3 -ffast-math -mtune=native -Wall `cat .def.conf`

config:
	@./configure
	@make all

force: clean
	@./configure force
	@make all
>>>>>>> added a simple config script

all: libccv.a

clean:
	rm -f *.o 3rdparty/sha1/*.o 3rdparty/sfmt/*.o 3rdparty/kissfft/*.o 3rdparty/dsfmt/*.o libccv.a

libccv.a: ccv_cache.o ccv_memory.o 3rdparty/sha1/sha1.o 3rdparty/kissfft/kiss_fft.o 3rdparty/kissfft/kiss_fftnd.o 3rdparty/kissfft/kiss_fftr.o 3rdparty/kissfft/kiss_fftndr.o 3rdparty/kissfft/kissf_fft.o 3rdparty/kissfft/kissf_fftnd.o 3rdparty/kissfft/kissf_fftr.o 3rdparty/kissfft/kissf_fftndr.o 3rdparty/dsfmt/dSFMT.o 3rdparty/sfmt/SFMT.o ccv_io.o ccv_numeric.o ccv_algebra.o ccv_util.o ccv_basic.o ccv_resample.o ccv_transform.o ccv_classic.o ccv_daisy.o ccv_sift.o ccv_bbf.o ccv_mser.o ccv_swt.o ccv_dpm.o ccv_tld.o ccv_ferns.o ccv_icf.o
	$(AR) rcs $@ $^

ccv_io.o: ccv_io.c ccv.h ccv_internal.h io/*.c
	$(CC) $< -o $@ -c $(CFLAGS)

%.o: %.c ccv.h ccv_internal.h
	$(CC) $< -o $@ -c $(CFLAGS)