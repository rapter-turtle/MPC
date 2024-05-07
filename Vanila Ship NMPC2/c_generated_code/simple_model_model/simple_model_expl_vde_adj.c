/* This file was automatically generated by CasADi 3.6.3.
 *  It consists of: 
 *   1) content generated by CasADi runtime: not copyrighted
 *   2) template code copied from CasADi source: permissively licensed (MIT-0)
 *   3) user code: owned by the user
 *
 */
#ifdef __cplusplus
extern "C" {
#endif

/* How to prefix internal symbols */
#ifdef CASADI_CODEGEN_PREFIX
  #define CASADI_NAMESPACE_CONCAT(NS, ID) _CASADI_NAMESPACE_CONCAT(NS, ID)
  #define _CASADI_NAMESPACE_CONCAT(NS, ID) NS ## ID
  #define CASADI_PREFIX(ID) CASADI_NAMESPACE_CONCAT(CODEGEN_PREFIX, ID)
#else
  #define CASADI_PREFIX(ID) simple_model_expl_vde_adj_ ## ID
#endif

#include <math.h>

#ifndef casadi_real
#define casadi_real double
#endif

#ifndef casadi_int
#define casadi_int int
#endif

/* Add prefix to internal symbols */
#define casadi_c0 CASADI_PREFIX(c0)
#define casadi_c1 CASADI_PREFIX(c1)
#define casadi_c2 CASADI_PREFIX(c2)
#define casadi_clear CASADI_PREFIX(clear)
#define casadi_copy CASADI_PREFIX(copy)
#define casadi_f0 CASADI_PREFIX(f0)
#define casadi_s0 CASADI_PREFIX(s0)
#define casadi_s1 CASADI_PREFIX(s1)
#define casadi_s2 CASADI_PREFIX(s2)
#define casadi_s3 CASADI_PREFIX(s3)

/* Symbol visibility in DLLs */
#ifndef CASADI_SYMBOL_EXPORT
  #if defined(_WIN32) || defined(__WIN32__) || defined(__CYGWIN__)
    #if defined(STATIC_LINKED)
      #define CASADI_SYMBOL_EXPORT
    #else
      #define CASADI_SYMBOL_EXPORT __declspec(dllexport)
    #endif
  #elif defined(__GNUC__) && defined(GCC_HASCLASSVISIBILITY)
    #define CASADI_SYMBOL_EXPORT __attribute__ ((visibility ("default")))
  #else
    #define CASADI_SYMBOL_EXPORT
  #endif
#endif

void casadi_clear(casadi_real* x, casadi_int n) {
  casadi_int i;
  if (x) {
    for (i=0; i<n; ++i) *x++ = 0;
  }
}

void casadi_copy(const casadi_real* x, casadi_int n, casadi_real* y) {
  casadi_int i;
  if (y) {
    if (x) {
      for (i=0; i<n; ++i) *y++ = *x++;
    } else {
      for (i=0; i<n; ++i) *y++ = 0.;
    }
  }
}

static const casadi_int casadi_s0[12] = {8, 1, 0, 8, 0, 1, 2, 3, 4, 5, 6, 7};
static const casadi_int casadi_s1[6] = {2, 1, 0, 2, 0, 1};
static const casadi_int casadi_s2[3] = {0, 0, 0};
static const casadi_int casadi_s3[12] = {10, 1, 0, 8, 0, 1, 2, 5, 6, 7, 8, 9};

static const casadi_real casadi_c0[3] = {2.5125628140703520e-04, 2.5125628140703520e-04, 5.0753692331117091e-05};
static const casadi_real casadi_c1[3] = {0., 0., 1.};
static const casadi_real casadi_c2[9] = {50., 0., 0., 0., 200., 0., 0., 0., 1281.};

/* simple_model_expl_vde_adj:(i0[8],i1[8],i2[2],i3[])->(o0[10x1,8nz]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_int i, j, k;
  casadi_real *rr, *ss, *tt;
  const casadi_real *cs;
  casadi_real w0, *w1=w+4, *w2=w+13, *w3=w+16, *w4=w+24, *w5=w+27, w6, w7, w8, w9, w10, *w11=w+35, *w12=w+38, *w13=w+41, *w14=w+50, *w15=w+53, w16, w17, w18, w19, w20, w21, *w22=w+62, *w23=w+65, w24, *w25=w+69;
  /* #0: @0 = -3980 */
  w0 = -3980.;
  /* #1: @1 = zeros(3x3) */
  casadi_clear(w1, 9);
  /* #2: @2 = 
  [[0.000251256, 00, 00], 
   [00, 0.000251256, 00], 
   [00, 00, 5.07537e-05]] */
  casadi_copy(casadi_c0, 3, w2);
  /* #3: @2 = nonzeros(@2) */
  /* #4: @3 = input[1][0] */
  casadi_copy(arg[1], 8, w3);
  /* #5: {@4, @5, @6, @7} = vertsplit(@3) */
  casadi_copy(w3, 3, w4);
  casadi_copy(w3+3, 3, w5);
  w6 = w3[6];
  w7 = w3[7];
  /* #6: @2 = (@2*@4) */
  for (i=0, rr=w2, cs=w4; i<3; ++i) (*rr++) *= (*cs++);
  /* #7: @4 = (-@2) */
  for (i=0, rr=w4, cs=w2; i<3; ++i) *rr++ = (- *cs++ );
  /* #8: @8 = input[0][0] */
  w8 = arg[0] ? arg[0][0] : 0;
  /* #9: @9 = input[0][1] */
  w9 = arg[0] ? arg[0][1] : 0;
  /* #10: @10 = input[0][2] */
  w10 = arg[0] ? arg[0][2] : 0;
  /* #11: @11 = vertcat(@8, @9, @10) */
  rr=w11;
  *rr++ = w8;
  *rr++ = w9;
  *rr++ = w10;
  /* #12: @12 = @11' */
  casadi_copy(w11, 3, w12);
  /* #13: @1 = mac(@4,@12,@1) */
  for (i=0, rr=w1; i<3; ++i) for (j=0; j<3; ++j, ++rr) for (k=0, ss=w4+j, tt=w12+i*1; k<1; ++k) *rr += ss[k*3]**tt++;
  /* #14: @13 = @1' */
  for (i=0, rr=w13, cs=w1; i<3; ++i) for (j=0; j<3; ++j) rr[i+j*3] = *cs++;
  /* #15: {@12, @14, @15} = horzsplit(@13) */
  casadi_copy(w13, 3, w12);
  casadi_copy(w13+3, 3, w14);
  casadi_copy(w13+6, 3, w15);
  /* #16: @15 = @15' */
  /* #17: {@10, @16, NULL} = horzsplit(@15) */
  w10 = w15[0];
  w16 = w15[1];
  /* #18: @16 = (@0*@16) */
  w16  = (w0*w16);
  /* #19: @17 = 3980 */
  w17 = 3980.;
  /* #20: @14 = @14' */
  /* #21: {NULL, NULL, @18} = horzsplit(@14) */
  w18 = w14[2];
  /* #22: @18 = (@17*@18) */
  w18  = (w17*w18);
  /* #23: @16 = (@16+@18) */
  w16 += w18;
  /* #24: @14 = zeros(3x1) */
  casadi_clear(w14, 3);
  /* #25: @18 = input[0][5] */
  w18 = arg[0] ? arg[0][5] : 0;
  /* #26: @19 = cos(@18) */
  w19 = cos( w18 );
  /* #27: @20 = sin(@18) */
  w20 = sin( w18 );
  /* #28: @20 = (-@20) */
  w20 = (- w20 );
  /* #29: @21 = 0 */
  w21 = 0.;
  /* #30: @15 = horzcat(@19, @20, @21) */
  rr=w15;
  *rr++ = w19;
  *rr++ = w20;
  *rr++ = w21;
  /* #31: @15 = @15' */
  /* #32: @19 = sin(@18) */
  w19 = sin( w18 );
  /* #33: @20 = cos(@18) */
  w20 = cos( w18 );
  /* #34: @21 = 0 */
  w21 = 0.;
  /* #35: @22 = horzcat(@19, @20, @21) */
  rr=w22;
  *rr++ = w19;
  *rr++ = w20;
  *rr++ = w21;
  /* #36: @22 = @22' */
  /* #37: @23 = [[0, 0, 1]] */
  casadi_copy(casadi_c1, 3, w23);
  /* #38: @23 = @23' */
  /* #39: @13 = horzcat(@15, @22, @23) */
  rr=w13;
  for (i=0, cs=w15; i<3; ++i) *rr++ = *cs++;
  for (i=0, cs=w22; i<3; ++i) *rr++ = *cs++;
  for (i=0, cs=w23; i<3; ++i) *rr++ = *cs++;
  /* #40: @14 = mac(@13,@5,@14) */
  for (i=0, rr=w14; i<1; ++i) for (j=0; j<3; ++j, ++rr) for (k=0, ss=w13+j, tt=w5+i*3; k<3; ++k) *rr += ss[k*3]**tt++;
  /* #41: @15 = zeros(3x1) */
  casadi_clear(w15, 3);
  /* #42: @19 = 0 */
  w19 = 0.;
  /* #43: @20 = 0 */
  w20 = 0.;
  /* #44: @21 = -3980 */
  w21 = -3980.;
  /* #45: @24 = (@21*@9) */
  w24  = (w21*w9);
  /* #46: @22 = horzcat(@19, @20, @24) */
  rr=w22;
  *rr++ = w19;
  *rr++ = w20;
  *rr++ = w24;
  /* #47: @22 = @22' */
  /* #48: @19 = 0 */
  w19 = 0.;
  /* #49: @20 = 0 */
  w20 = 0.;
  /* #50: @17 = (@17*@8) */
  w17 *= w8;
  /* #51: @23 = horzcat(@19, @20, @17) */
  rr=w23;
  *rr++ = w19;
  *rr++ = w20;
  *rr++ = w17;
  /* #52: @23 = @23' */
  /* #53: @19 = 3980 */
  w19 = 3980.;
  /* #54: @9 = (@19*@9) */
  w9  = (w19*w9);
  /* #55: @0 = (@0*@8) */
  w0 *= w8;
  /* #56: @8 = 0 */
  w8 = 0.;
  /* #57: @25 = horzcat(@9, @0, @8) */
  rr=w25;
  *rr++ = w9;
  *rr++ = w0;
  *rr++ = w8;
  /* #58: @25 = @25' */
  /* #59: @13 = horzcat(@22, @23, @25) */
  rr=w13;
  for (i=0, cs=w22; i<3; ++i) *rr++ = *cs++;
  for (i=0, cs=w23; i<3; ++i) *rr++ = *cs++;
  for (i=0, cs=w25; i<3; ++i) *rr++ = *cs++;
  /* #60: @15 = mac(@13,@4,@15) */
  for (i=0, rr=w15; i<1; ++i) for (j=0; j<3; ++j, ++rr) for (k=0, ss=w13+j, tt=w4+i*3; k<3; ++k) *rr += ss[k*3]**tt++;
  /* #61: @14 = (@14+@15) */
  for (i=0, rr=w14, cs=w15; i<3; ++i) (*rr++) += (*cs++);
  /* #62: @15 = zeros(3x1) */
  casadi_clear(w15, 3);
  /* #63: @13 = 
  [[50, -0, -0], 
   [-0, 200, -0], 
   [-0, -0, 1281]] */
  casadi_copy(casadi_c2, 9, w13);
  /* #64: @1 = @13' */
  for (i=0, rr=w1, cs=w13; i<3; ++i) for (j=0; j<3; ++j) rr[i+j*3] = *cs++;
  /* #65: @4 = (-@2) */
  for (i=0, rr=w4, cs=w2; i<3; ++i) *rr++ = (- *cs++ );
  /* #66: @15 = mac(@1,@4,@15) */
  for (i=0, rr=w15; i<1; ++i) for (j=0; j<3; ++j, ++rr) for (k=0, ss=w1+j, tt=w4+i*3; k<3; ++k) *rr += ss[k*3]**tt++;
  /* #67: @14 = (@14+@15) */
  for (i=0, rr=w14, cs=w15; i<3; ++i) (*rr++) += (*cs++);
  /* #68: {@9, @0, @8} = vertsplit(@14) */
  w9 = w14[0];
  w0 = w14[1];
  w8 = w14[2];
  /* #69: @16 = (@16+@9) */
  w16 += w9;
  /* #70: output[0][0] = @16 */
  if (res[0]) res[0][0] = w16;
  /* #71: @19 = (@19*@10) */
  w19 *= w10;
  /* #72: @12 = @12' */
  /* #73: {NULL, NULL, @10} = horzsplit(@12) */
  w10 = w12[2];
  /* #74: @21 = (@21*@10) */
  w21 *= w10;
  /* #75: @19 = (@19+@21) */
  w19 += w21;
  /* #76: @19 = (@19+@0) */
  w19 += w0;
  /* #77: output[0][1] = @19 */
  if (res[0]) res[0][1] = w19;
  /* #78: output[0][2] = @8 */
  if (res[0]) res[0][2] = w8;
  /* #79: @8 = cos(@18) */
  w8 = cos( w18 );
  /* #80: @1 = zeros(3x3) */
  casadi_clear(w1, 9);
  /* #81: @11 = @11' */
  /* #82: @1 = mac(@5,@11,@1) */
  for (i=0, rr=w1; i<3; ++i) for (j=0; j<3; ++j, ++rr) for (k=0, ss=w5+j, tt=w11+i*1; k<1; ++k) *rr += ss[k*3]**tt++;
  /* #83: @13 = @1' */
  for (i=0, rr=w13, cs=w1; i<3; ++i) for (j=0; j<3; ++j) rr[i+j*3] = *cs++;
  /* #84: {@5, @11, NULL} = horzsplit(@13) */
  casadi_copy(w13, 3, w5);
  casadi_copy(w13+3, 3, w11);
  /* #85: @11 = @11' */
  /* #86: {@19, @0, NULL} = horzsplit(@11) */
  w19 = w11[0];
  w0 = w11[1];
  /* #87: @8 = (@8*@19) */
  w8 *= w19;
  /* #88: @19 = sin(@18) */
  w19 = sin( w18 );
  /* #89: @19 = (@19*@0) */
  w19 *= w0;
  /* #90: @8 = (@8-@19) */
  w8 -= w19;
  /* #91: @19 = cos(@18) */
  w19 = cos( w18 );
  /* #92: @5 = @5' */
  /* #93: {@0, @21, NULL} = horzsplit(@5) */
  w0 = w5[0];
  w21 = w5[1];
  /* #94: @19 = (@19*@21) */
  w19 *= w21;
  /* #95: @8 = (@8-@19) */
  w8 -= w19;
  /* #96: @18 = sin(@18) */
  w18 = sin( w18 );
  /* #97: @18 = (@18*@0) */
  w18 *= w0;
  /* #98: @8 = (@8-@18) */
  w8 -= w18;
  /* #99: output[0][3] = @8 */
  if (res[0]) res[0][3] = w8;
  /* #100: {@8, @18, @0} = vertsplit(@2) */
  w8 = w2[0];
  w18 = w2[1];
  w0 = w2[2];
  /* #101: output[0][4] = @8 */
  if (res[0]) res[0][4] = w8;
  /* #102: @8 = 4 */
  w8 = 4.;
  /* #103: @8 = (@8*@0) */
  w8 *= w0;
  /* #104: @18 = (@18+@8) */
  w18 += w8;
  /* #105: output[0][5] = @18 */
  if (res[0]) res[0][5] = w18;
  /* #106: output[0][6] = @6 */
  if (res[0]) res[0][6] = w6;
  /* #107: output[0][7] = @7 */
  if (res[0]) res[0][7] = w7;
  return 0;
}

CASADI_SYMBOL_EXPORT int simple_model_expl_vde_adj(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int simple_model_expl_vde_adj_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int simple_model_expl_vde_adj_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void simple_model_expl_vde_adj_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int simple_model_expl_vde_adj_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void simple_model_expl_vde_adj_release(int mem) {
}

CASADI_SYMBOL_EXPORT void simple_model_expl_vde_adj_incref(void) {
}

CASADI_SYMBOL_EXPORT void simple_model_expl_vde_adj_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int simple_model_expl_vde_adj_n_in(void) { return 4;}

CASADI_SYMBOL_EXPORT casadi_int simple_model_expl_vde_adj_n_out(void) { return 1;}

CASADI_SYMBOL_EXPORT casadi_real simple_model_expl_vde_adj_default_in(casadi_int i) {
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* simple_model_expl_vde_adj_name_in(casadi_int i) {
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    case 3: return "i3";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* simple_model_expl_vde_adj_name_out(casadi_int i) {
  switch (i) {
    case 0: return "o0";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* simple_model_expl_vde_adj_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s0;
    case 2: return casadi_s1;
    case 3: return casadi_s2;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* simple_model_expl_vde_adj_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s3;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int simple_model_expl_vde_adj_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 7;
  if (sz_res) *sz_res = 5;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 72;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif