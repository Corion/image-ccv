/* Make the ccv library conveniently available */
#include <stdint.h>
#ifdef WIN32
#include "winsock2.h"
#endif
#include "ccv-src/lib/3rdparty/sha1/sha1.h"
#include "ccv-src/lib/3rdparty/sha1/sha1.c"
#include "ccv-src/lib/ccv.h"
#include "ccv-src/lib/ccv_basic.c"
#include "ccv-src/lib/ccv_algebra.c"
#include "ccv-src/lib/ccv_cache.c"
#include "ccv-src/lib/ccv_memory.c"
#include "ccv-src/lib/ccv_util.c"
#include "ccv-src/lib/ccv_io.c"
#include "ccv-src/lib/ccv_sift.c"
#include "ccv-src/lib/ccv_bbf.c"
#include <ctype.h>
