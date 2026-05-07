#include "oblas.h"
#include <errno.h>
#include <stdlib.h>

#if defined(_MSC_VER)
#include <malloc.h>
#endif

void *oalloc(size_t nmemb, size_t size, size_t align) {
  size_t aligned_sz = ((size / align) + ((size % align) ? 1 : 0)) * align;
  size_t total = nmemb * aligned_sz;

#if defined(_MSC_VER)
  /* Argument order is (size, align), opposite of posix_memalign. */
  void *aligned = _aligned_malloc(total, align);
  if (!aligned) {
    exit(ENOMEM);
  }
  return aligned;
#else
  void *aligned = NULL;
  if (posix_memalign(&aligned, align, total) != 0) {
    exit(ENOMEM);
  }
  return aligned;
#endif
}

void ofree(void *ptr) {
  if (!ptr) return;
#if defined(_MSC_VER)
  _aligned_free(ptr);
#else
  free(ptr);
#endif
}

#ifdef OBLAS_SSE
#include "oblas_sse.c"
#else
#ifdef OBLAS_AVX
#include "oblas_avx.c"
#else
#ifdef OBLAS_NEON
#include "oblas_neon.c"
#else
#include "oblas_classic.c"
#endif
#endif
#endif
