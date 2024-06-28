/* This file was automatically generated by CasADi.
   The CasADi copyright holders make no ownership claim of its contents. */
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
#define casadi_clear CASADI_PREFIX(clear)
#define casadi_copy CASADI_PREFIX(copy)
#define casadi_f0 CASADI_PREFIX(f0)
#define casadi_mtimes CASADI_PREFIX(mtimes)
#define casadi_s0 CASADI_PREFIX(s0)
#define casadi_s1 CASADI_PREFIX(s1)
#define casadi_s2 CASADI_PREFIX(s2)
#define casadi_s3 CASADI_PREFIX(s3)
#define casadi_s4 CASADI_PREFIX(s4)
#define casadi_trans CASADI_PREFIX(trans)

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

void casadi_clear(casadi_real* x, casadi_int n) {
  casadi_int i;
  if (x) {
    for (i=0; i<n; ++i) *x++ = 0;
  }
}

void casadi_trans(const casadi_real* x, const casadi_int* sp_x, casadi_real* y,
    const casadi_int* sp_y, casadi_int* tmp) {
  casadi_int ncol_x, nnz_x, ncol_y, k;
  const casadi_int* row_x, *colind_y;
  ncol_x = sp_x[1];
  nnz_x = sp_x[2 + ncol_x];
  row_x = sp_x + 2 + ncol_x+1;
  ncol_y = sp_y[1];
  colind_y = sp_y+2;
  for (k=0; k<ncol_y; ++k) tmp[k] = colind_y[k];
  for (k=0; k<nnz_x; ++k) {
    y[tmp[row_x[k]]++] = x[k];
  }
}

void casadi_mtimes(const casadi_real* x, const casadi_int* sp_x, const casadi_real* y, const casadi_int* sp_y, casadi_real* z, const casadi_int* sp_z, casadi_real* w, casadi_int tr) {
  casadi_int ncol_x, ncol_y, ncol_z, cc;
  const casadi_int *colind_x, *row_x, *colind_y, *row_y, *colind_z, *row_z;
  ncol_x = sp_x[1];
  colind_x = sp_x+2; row_x = sp_x + 2 + ncol_x+1;
  ncol_y = sp_y[1];
  colind_y = sp_y+2; row_y = sp_y + 2 + ncol_y+1;
  ncol_z = sp_z[1];
  colind_z = sp_z+2; row_z = sp_z + 2 + ncol_z+1;
  if (tr) {
    for (cc=0; cc<ncol_z; ++cc) {
      casadi_int kk;
      for (kk=colind_y[cc]; kk<colind_y[cc+1]; ++kk) {
        w[row_y[kk]] = y[kk];
      }
      for (kk=colind_z[cc]; kk<colind_z[cc+1]; ++kk) {
        casadi_int kk1;
        casadi_int rr = row_z[kk];
        for (kk1=colind_x[rr]; kk1<colind_x[rr+1]; ++kk1) {
          z[kk] += x[kk1] * w[row_x[kk1]];
        }
      }
    }
  } else {
    for (cc=0; cc<ncol_y; ++cc) {
      casadi_int kk;
      for (kk=colind_z[cc]; kk<colind_z[cc+1]; ++kk) {
        w[row_z[kk]] = z[kk];
      }
      for (kk=colind_y[cc]; kk<colind_y[cc+1]; ++kk) {
        casadi_int kk1;
        casadi_int rr = row_y[kk];
        for (kk1=colind_x[rr]; kk1<colind_x[rr+1]; ++kk1) {
          w[row_x[kk1]] += x[kk1]*y[kk];
        }
      }
      for (kk=colind_z[cc]; kk<colind_z[cc+1]; ++kk) {
        z[kk] = w[row_z[kk]];
      }
    }
  }
}

static const casadi_int casadi_s0[9] = {3, 3, 0, 1, 2, 3, 0, 1, 2};
static const casadi_int casadi_s1[7] = {3, 1, 0, 3, 0, 1, 2};
static const casadi_int casadi_s2[12] = {8, 1, 0, 8, 0, 1, 2, 3, 4, 5, 6, 7};
static const casadi_int casadi_s3[6] = {2, 1, 0, 2, 0, 1};
static const casadi_int casadi_s4[12] = {10, 1, 0, 8, 0, 1, 2, 5, 6, 7, 8, 9};

static const casadi_real casadi_c0[3] = {2.5125628140703520e-04, 2.5125628140703520e-04, 5.0753692331117091e-05};
static const casadi_real casadi_c1[9] = {50., 0., 0., 0., 200., 0., 0., 0., 1281.};

/* simple_model_expl_vde_adj:(i0[8],i1[8],i2[2],i3[3])->(o0[10x1,8nz]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_int i, j, k;
  casadi_real *rr, *ss, *tt;
  const casadi_real *cs;
  casadi_real w0, w1, *w2=w+5, *w3=w+13, w4, w5, w6, w7, w8, w9, *w10=w+22, *w11=w+31, *w12=w+34, *w13=w+37, w14, w15, w16, *w17=w+43, *w18=w+52, *w19=w+55, w20, w21, w22, w23, w24, w25, *w26=w+64, *w27=w+67;
  /* #0: @0 = input[0][5] */
  w0 = arg[0] ? arg[0][5] : 0;
  /* #1: @1 = sin(@0) */
  w1 = sin( w0 );
  /* #2: @2 = input[1][0] */
  casadi_copy(arg[1], 8, w2);
  /* #3: {@3, @4, @5, @6, @7, @8} = vertsplit(@2) */
  casadi_copy(w2, 3, w3);
  w4 = w2[3];
  w5 = w2[4];
  w6 = w2[5];
  w7 = w2[6];
  w8 = w2[7];
  /* #4: @1 = (@1*@5) */
  w1 *= w5;
  /* #5: @9 = cos(@0) */
  w9 = cos( w0 );
  /* #6: @9 = (@9*@4) */
  w9 *= w4;
  /* #7: @1 = (@1+@9) */
  w1 += w9;
  /* #8: @9 = -3980 */
  w9 = -3980.;
  /* #9: @10 = zeros(3x3) */
  casadi_clear(w10, 9);
  /* #10: @11 = zeros(3x1) */
  casadi_clear(w11, 3);
  /* #11: @12 = 
  [[0.000251256, 00, 00], 
   [00, 0.000251256, 00], 
   [00, 00, 5.07537e-05]] */
  casadi_copy(casadi_c0, 3, w12);
  /* #12: @13 = @12' */
  casadi_trans(w12,casadi_s0, w13, casadi_s0, iw);
  /* #13: @11 = mac(@13,@3,@11) */
  casadi_mtimes(w13, casadi_s0, w3, casadi_s1, w11, casadi_s1, w, 0);
  /* #14: @13 = (-@11) */
  for (i=0, rr=w13, cs=w11; i<3; ++i) *rr++ = (- *cs++ );
  /* #15: @14 = input[0][0] */
  w14 = arg[0] ? arg[0][0] : 0;
  /* #16: @15 = input[0][1] */
  w15 = arg[0] ? arg[0][1] : 0;
  /* #17: @16 = input[0][2] */
  w16 = arg[0] ? arg[0][2] : 0;
  /* #18: @12 = vertcat(@14, @15, @16) */
  rr=w12;
  *rr++ = w14;
  *rr++ = w15;
  *rr++ = w16;
  /* #19: @12 = @12' */
  /* #20: @10 = mac(@13,@12,@10) */
  for (i=0, rr=w10; i<3; ++i) for (j=0; j<3; ++j, ++rr) for (k=0, ss=w13+j, tt=w12+i*1; k<1; ++k) *rr += ss[k*3]**tt++;
  /* #21: @17 = @10' */
  for (i=0, rr=w17, cs=w10; i<3; ++i) for (j=0; j<3; ++j) rr[i+j*3] = *cs++;
  /* #22: {@12, @18, @19} = horzsplit(@17) */
  casadi_copy(w17, 3, w12);
  casadi_copy(w17+3, 3, w18);
  casadi_copy(w17+6, 3, w19);
  /* #23: @19 = @19' */
  /* #24: {@20, @21, NULL} = horzsplit(@19) */
  w20 = w19[0];
  w21 = w19[1];
  /* #25: @21 = (@9*@21) */
  w21  = (w9*w21);
  /* #26: @1 = (@1+@21) */
  w1 += w21;
  /* #27: @21 = 3980 */
  w21 = 3980.;
  /* #28: @18 = @18' */
  /* #29: {NULL, NULL, @22} = horzsplit(@18) */
  w22 = w18[2];
  /* #30: @22 = (@21*@22) */
  w22  = (w21*w22);
  /* #31: @1 = (@1+@22) */
  w1 += w22;
  /* #32: @18 = zeros(3x1) */
  casadi_clear(w18, 3);
  /* #33: @22 = 0 */
  w22 = 0.;
  /* #34: @23 = 0 */
  w23 = 0.;
  /* #35: @24 = -3980 */
  w24 = -3980.;
  /* #36: @25 = (@24*@15) */
  w25  = (w24*w15);
  /* #37: @19 = horzcat(@22, @23, @25) */
  rr=w19;
  *rr++ = w22;
  *rr++ = w23;
  *rr++ = w25;
  /* #38: @19 = @19' */
  /* #39: @22 = 0 */
  w22 = 0.;
  /* #40: @23 = 0 */
  w23 = 0.;
  /* #41: @21 = (@21*@14) */
  w21 *= w14;
  /* #42: @26 = horzcat(@22, @23, @21) */
  rr=w26;
  *rr++ = w22;
  *rr++ = w23;
  *rr++ = w21;
  /* #43: @26 = @26' */
  /* #44: @22 = 3980 */
  w22 = 3980.;
  /* #45: @23 = (@22*@15) */
  w23  = (w22*w15);
  /* #46: @9 = (@9*@14) */
  w9 *= w14;
  /* #47: @21 = 0 */
  w21 = 0.;
  /* #48: @27 = horzcat(@23, @9, @21) */
  rr=w27;
  *rr++ = w23;
  *rr++ = w9;
  *rr++ = w21;
  /* #49: @27 = @27' */
  /* #50: @17 = horzcat(@19, @26, @27) */
  rr=w17;
  for (i=0, cs=w19; i<3; ++i) *rr++ = *cs++;
  for (i=0, cs=w26; i<3; ++i) *rr++ = *cs++;
  for (i=0, cs=w27; i<3; ++i) *rr++ = *cs++;
  /* #51: @18 = mac(@17,@13,@18) */
  for (i=0, rr=w18; i<1; ++i) for (j=0; j<3; ++j, ++rr) for (k=0, ss=w17+j, tt=w13+i*3; k<3; ++k) *rr += ss[k*3]**tt++;
  /* #52: @13 = zeros(3x1) */
  casadi_clear(w13, 3);
  /* #53: @17 = 
  [[50, -0, -0], 
   [-0, 200, -0], 
   [-0, -0, 1281]] */
  casadi_copy(casadi_c1, 9, w17);
  /* #54: @10 = @17' */
  for (i=0, rr=w10, cs=w17; i<3; ++i) for (j=0; j<3; ++j) rr[i+j*3] = *cs++;
  /* #55: @19 = (-@11) */
  for (i=0, rr=w19, cs=w11; i<3; ++i) *rr++ = (- *cs++ );
  /* #56: @13 = mac(@10,@19,@13) */
  for (i=0, rr=w13; i<1; ++i) for (j=0; j<3; ++j, ++rr) for (k=0, ss=w10+j, tt=w19+i*3; k<3; ++k) *rr += ss[k*3]**tt++;
  /* #57: @18 = (@18+@13) */
  for (i=0, rr=w18, cs=w13; i<3; ++i) (*rr++) += (*cs++);
  /* #58: {@23, @9, @21} = vertsplit(@18) */
  w23 = w18[0];
  w9 = w18[1];
  w21 = w18[2];
  /* #59: @1 = (@1+@23) */
  w1 += w23;
  /* #60: output[0][0] = @1 */
  if (res[0]) res[0][0] = w1;
  /* #61: @1 = cos(@0) */
  w1 = cos( w0 );
  /* #62: @1 = (@1*@5) */
  w1 *= w5;
  /* #63: @23 = sin(@0) */
  w23 = sin( w0 );
  /* #64: @23 = (@23*@4) */
  w23 *= w4;
  /* #65: @1 = (@1-@23) */
  w1 -= w23;
  /* #66: @22 = (@22*@20) */
  w22 *= w20;
  /* #67: @1 = (@1+@22) */
  w1 += w22;
  /* #68: @12 = @12' */
  /* #69: {NULL, NULL, @22} = horzsplit(@12) */
  w22 = w12[2];
  /* #70: @24 = (@24*@22) */
  w24 *= w22;
  /* #71: @1 = (@1+@24) */
  w1 += w24;
  /* #72: @1 = (@1+@9) */
  w1 += w9;
  /* #73: output[0][1] = @1 */
  if (res[0]) res[0][1] = w1;
  /* #74: @1 = 3.5 */
  w1 = 3.5000000000000000e+00;
  /* #75: @9 = cos(@0) */
  w9 = cos( w0 );
  /* #76: @9 = (@9*@5) */
  w9 *= w5;
  /* #77: @9 = (@1*@9) */
  w9  = (w1*w9);
  /* #78: @6 = (@6+@9) */
  w6 += w9;
  /* #79: @9 = 3.5 */
  w9 = 3.5000000000000000e+00;
  /* #80: @24 = sin(@0) */
  w24 = sin( w0 );
  /* #81: @24 = (@24*@4) */
  w24 *= w4;
  /* #82: @24 = (@9*@24) */
  w24  = (w9*w24);
  /* #83: @6 = (@6-@24) */
  w6 -= w24;
  /* #84: @6 = (@6+@21) */
  w6 += w21;
  /* #85: output[0][2] = @6 */
  if (res[0]) res[0][2] = w6;
  /* #86: @6 = sin(@0) */
  w6 = sin( w0 );
  /* #87: @21 = (@15*@5) */
  w21  = (w15*w5);
  /* #88: @6 = (@6*@21) */
  w6 *= w21;
  /* #89: @6 = (-@6) */
  w6 = (- w6 );
  /* #90: @21 = sin(@0) */
  w21 = sin( w0 );
  /* #91: @1 = (@1*@16) */
  w1 *= w16;
  /* #92: @1 = (@1*@5) */
  w1 *= w5;
  /* #93: @21 = (@21*@1) */
  w21 *= w1;
  /* #94: @6 = (@6-@21) */
  w6 -= w21;
  /* #95: @21 = cos(@0) */
  w21 = cos( w0 );
  /* #96: @5 = (@14*@5) */
  w5  = (w14*w5);
  /* #97: @21 = (@21*@5) */
  w21 *= w5;
  /* #98: @6 = (@6+@21) */
  w6 += w21;
  /* #99: @21 = cos(@0) */
  w21 = cos( w0 );
  /* #100: @9 = (@9*@16) */
  w9 *= w16;
  /* #101: @9 = (@9*@4) */
  w9 *= w4;
  /* #102: @21 = (@21*@9) */
  w21 *= w9;
  /* #103: @6 = (@6-@21) */
  w6 -= w21;
  /* #104: @21 = cos(@0) */
  w21 = cos( w0 );
  /* #105: @15 = (@15*@4) */
  w15 *= w4;
  /* #106: @21 = (@21*@15) */
  w21 *= w15;
  /* #107: @6 = (@6-@21) */
  w6 -= w21;
  /* #108: @21 = sin(@0) */
  w21 = sin( w0 );
  /* #109: @14 = (@14*@4) */
  w14 *= w4;
  /* #110: @21 = (@21*@14) */
  w21 *= w14;
  /* #111: @6 = (@6-@21) */
  w6 -= w21;
  /* #112: @21 = cos(@0) */
  w21 = cos( w0 );
  /* #113: @14 = input[3][0] */
  w14 = arg[3] ? arg[3][0] : 0;
  /* #114: {@4, @15, NULL} = vertsplit(@3) */
  w4 = w3[0];
  w15 = w3[1];
  /* #115: @9 = (@14*@15) */
  w9  = (w14*w15);
  /* #116: @21 = (@21*@9) */
  w21 *= w9;
  /* #117: @6 = (@6-@21) */
  w6 -= w21;
  /* #118: @21 = sin(@0) */
  w21 = sin( w0 );
  /* #119: @9 = input[3][1] */
  w9 = arg[3] ? arg[3][1] : 0;
  /* #120: @15 = (@9*@15) */
  w15  = (w9*w15);
  /* #121: @21 = (@21*@15) */
  w21 *= w15;
  /* #122: @6 = (@6-@21) */
  w6 -= w21;
  /* #123: @21 = sin(@0) */
  w21 = sin( w0 );
  /* #124: @14 = (@14*@4) */
  w14 *= w4;
  /* #125: @21 = (@21*@14) */
  w21 *= w14;
  /* #126: @6 = (@6-@21) */
  w6 -= w21;
  /* #127: @0 = cos(@0) */
  w0 = cos( w0 );
  /* #128: @9 = (@9*@4) */
  w9 *= w4;
  /* #129: @0 = (@0*@9) */
  w0 *= w9;
  /* #130: @6 = (@6+@0) */
  w6 += w0;
  /* #131: output[0][3] = @6 */
  if (res[0]) res[0][3] = w6;
  /* #132: {@6, @0, @9} = vertsplit(@11) */
  w6 = w11[0];
  w0 = w11[1];
  w9 = w11[2];
  /* #133: output[0][4] = @6 */
  if (res[0]) res[0][4] = w6;
  /* #134: @6 = -4 */
  w6 = -4.;
  /* #135: @6 = (@6*@9) */
  w6 *= w9;
  /* #136: @0 = (@0+@6) */
  w0 += w6;
  /* #137: output[0][5] = @0 */
  if (res[0]) res[0][5] = w0;
  /* #138: output[0][6] = @7 */
  if (res[0]) res[0][6] = w7;
  /* #139: output[0][7] = @8 */
  if (res[0]) res[0][7] = w8;
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

CASADI_SYMBOL_EXPORT casadi_real simple_model_expl_vde_adj_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* simple_model_expl_vde_adj_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    case 3: return "i3";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* simple_model_expl_vde_adj_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* simple_model_expl_vde_adj_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s2;
    case 1: return casadi_s2;
    case 2: return casadi_s3;
    case 3: return casadi_s1;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* simple_model_expl_vde_adj_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s4;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int simple_model_expl_vde_adj_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 7;
  if (sz_res) *sz_res = 7;
  if (sz_iw) *sz_iw = 4;
  if (sz_w) *sz_w = 70;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
