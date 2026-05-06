#ifndef OCTET_BLAS_H
#define OCTET_BLAS_H

#include <stdint.h>

#include "octmat.h"
#include "octtables.h"

/* MSVC compatibility shims — provide GCC/Clang builtins used by the
   bit-twiddling routines in oblas_classic.c and gf2.c. Guarded so a TU
   that includes both oblas.h and gf2.h does not redefine them. Safe to
   include from C and C++. */
#if defined(_MSC_VER) && !defined(__clang__) && !defined(_OBLAS_GCC_BUILTIN_SHIMS)
#define _OBLAS_GCC_BUILTIN_SHIMS
#include <intrin.h>
static __inline int __builtin_ctz(uint32_t x) {
  unsigned long r = 0;
  _BitScanForward(&r, x);
  return (int)r;
}
static __inline int __builtin_popcount(uint32_t x) {
  return (int)__popcnt(x);
}
#endif

#define OCTET_MUL(u, v) OCT_EXP[OCT_LOG[u] + OCT_LOG[v]]
#define OCTET_DIV(u, v) OCT_EXP[OCT_LOG[u] - OCT_LOG[v] + 255]
#define OCTET_SWAP(u, v)                                                       \
  do {                                                                         \
    uint8_t __tmp = (u);                                                       \
    (u) = (v);                                                                 \
    (v) = __tmp;                                                               \
  } while (0)

#define ALIGNED_COLS(k)                                                        \
  (((k) / OCTMAT_ALIGN) + (((k) % OCTMAT_ALIGN) ? 1 : 0)) * OCTMAT_ALIGN

typedef uint8_t octet;

/* Aligned allocator. On POSIX uses posix_memalign; on MSVC uses
   _aligned_malloc. Memory returned MUST be freed with ofree(), not
   plain free(), or the heap will be corrupted on Windows. */
void *oalloc(size_t nmemb, size_t size, size_t align);

/* Companion deallocator for memory returned by oalloc(). Matches
   _aligned_free under MSVC and plain free() elsewhere. Calling
   ofree(NULL) is a no-op. */
void ofree(void *ptr);

void ocopy(uint8_t *a, uint8_t *b, size_t i, size_t j, size_t k);
void oswaprow(uint8_t *a, size_t i, size_t j, size_t k);
void oswapcol(uint8_t *a, size_t i, size_t j, size_t k, size_t l);
void oaxpy(uint8_t *a, uint8_t *b, size_t i, size_t j, size_t k, uint8_t u);
void oaddrow(uint8_t *a, uint8_t *b, size_t i, size_t j, size_t k);
void oscal(uint8_t *a, size_t i, size_t k, uint8_t u);
void ozero(uint8_t *restrict a, size_t i, size_t k);
void ogemm(uint8_t *a, uint8_t *b, uint8_t *c, size_t n, size_t k, size_t m);
size_t onnz(uint8_t *a, size_t i, size_t s, size_t e, size_t k);
void oaxpy_b32(uint8_t *a, uint32_t *b, size_t i, size_t k, uint8_t u);

#endif
