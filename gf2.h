#ifndef GF2MAT_H
#define GF2MAT_H

#include <stdint.h>
#include <stdio.h>

/* MSVC compatibility shims for the GCC/Clang bit-twiddling builtins
   used in gf2.c. Mirrors the equivalent block in oblas.h; both share
   the _OBLAS_GCC_BUILTIN_SHIMS guard so a TU that includes both
   headers does not redefine them. */
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

#define gf2word uint32_t
#define gf2wsz (sizeof(gf2word) * 8)

#define gf2_at(g, i, j)                                                        \
  (!!((g->bits + (i)*g->stride)[(j) / gf2wsz] & (1 << ((j) % gf2wsz))))

typedef struct {
  size_t rows;
  size_t cols;
  size_t stride;
  gf2word *bits;
} gf2mat;

gf2mat *gf2mat_new(size_t rows, size_t cols);
void gf2mat_free(gf2mat *gf2);
void gf2mat_print(gf2mat *gf2, FILE *stream);
gf2mat *gf2mat_copy(gf2mat *gf2);

void gf2mat_axpy(gf2mat *gf2, int i, uint8_t *dst, int beta);
void gf2mat_fill(gf2mat *gf2, int i, uint8_t *dst);

int gf2mat_get(gf2mat *gf2, int i, int j);
void gf2mat_set(gf2mat *gf2, int i, int j, uint8_t b);

void gf2mat_xor(gf2mat *a, gf2mat *b, int i, int j);
void gf2mat_and(gf2mat *a, gf2mat *b, int i, int j);

int gf2mat_nnz(gf2mat *gf2, int i, int s, int e);

void gf2mat_swaprow(gf2mat *gf2, int i, int j);
void gf2mat_swapcol(gf2mat *gf2, int i, int j);
void gf2mat_zero(gf2mat *gf2, int i);

#endif
