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
  #define CASADI_PREFIX(ID) simple_model_expl_ode_fun_ ## ID
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
#define casadi_s0 CASADI_PREFIX(s0)
#define casadi_s1 CASADI_PREFIX(s1)
#define casadi_s2 CASADI_PREFIX(s2)

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

static const casadi_int casadi_s0[12] = {8, 1, 0, 8, 0, 1, 2, 3, 4, 5, 6, 7};
static const casadi_int casadi_s1[6] = {2, 1, 0, 2, 0, 1};
static const casadi_int casadi_s2[3] = {0, 0, 0};

static const casadi_real casadi_c0[3] = {2.5125628140703520e-04, 2.5125628140703520e-04, 5.0753692331117091e-05};
static const casadi_real casadi_c1[9] = {50., 0., 0., 0., 200., 0., 0., 0., 1281.};

/* simple_model_expl_ode_fun:(i0[8],i1[2],i2[])->(o0[8]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_int i, j, k;
  casadi_real *rr, *ss, *tt;
  const casadi_real *cs;
  casadi_real *w0=w+3, w1, w2, w3, *w4=w+9, *w5=w+12, *w6=w+15, *w7=w+24, w8, w9, w10, *w11=w+30, *w12=w+33, *w13=w+36, *w14=w+39, w15;
  /* #0: @0 = 
  [[0.000251256, 00, 00], 
   [00, 0.000251256, 00], 
   [00, 00, 5.07537e-05]] */
  casadi_copy(casadi_c0, 3, w0);
  /* #1: @0 = nonzeros(@0) */
  /* #2: @1 = input[0][6] */
  w1 = arg[0] ? arg[0][6] : 0;
  /* #3: @2 = input[0][7] */
  w2 = arg[0] ? arg[0][7] : 0;
  /* #4: @3 = -4 */
  w3 = -4.;
  /* #5: @3 = (@3*@2) */
  w3 *= w2;
  /* #6: @4 = vertcat(@1, @2, @3) */
  rr=w4;
  *rr++ = w1;
  *rr++ = w2;
  *rr++ = w3;
  /* #7: @5 = zeros(3x1) */
  casadi_clear(w5, 3);
  /* #8: @6 = 
  [[50, -0, -0], 
   [-0, 200, -0], 
   [-0, -0, 1281]] */
  casadi_copy(casadi_c1, 9, w6);
  /* #9: @1 = input[0][0] */
  w1 = arg[0] ? arg[0][0] : 0;
  /* #10: @2 = input[0][1] */
  w2 = arg[0] ? arg[0][1] : 0;
  /* #11: @3 = input[0][2] */
  w3 = arg[0] ? arg[0][2] : 0;
  /* #12: @7 = vertcat(@1, @2, @3) */
  rr=w7;
  *rr++ = w1;
  *rr++ = w2;
  *rr++ = w3;
  /* #13: @5 = mac(@6,@7,@5) */
  for (i=0, rr=w5; i<1; ++i) for (j=0; j<3; ++j, ++rr) for (k=0, ss=w6+j, tt=w7+i*3; k<3; ++k) *rr += ss[k*3]**tt++;
  /* #14: @4 = (@4-@5) */
  for (i=0, rr=w4, cs=w5; i<3; ++i) (*rr++) -= (*cs++);
  /* #15: @5 = zeros(3x1) */
  casadi_clear(w5, 3);
  /* #16: @8 = 0 */
  w8 = 0.;
  /* #17: @9 = 0 */
  w9 = 0.;
  /* #18: @10 = -3980 */
  w10 = -3980.;
  /* #19: @10 = (@10*@2) */
  w10 *= w2;
  /* #20: @11 = horzcat(@8, @9, @10) */
  rr=w11;
  *rr++ = w8;
  *rr++ = w9;
  *rr++ = w10;
  /* #21: @11 = @11' */
  /* #22: @8 = 0 */
  w8 = 0.;
  /* #23: @9 = 0 */
  w9 = 0.;
  /* #24: @10 = 3980 */
  w10 = 3980.;
  /* #25: @10 = (@10*@1) */
  w10 *= w1;
  /* #26: @12 = horzcat(@8, @9, @10) */
  rr=w12;
  *rr++ = w8;
  *rr++ = w9;
  *rr++ = w10;
  /* #27: @12 = @12' */
  /* #28: @8 = 3980 */
  w8 = 3980.;
  /* #29: @8 = (@8*@2) */
  w8 *= w2;
  /* #30: @9 = -3980 */
  w9 = -3980.;
  /* #31: @9 = (@9*@1) */
  w9 *= w1;
  /* #32: @10 = 0 */
  w10 = 0.;
  /* #33: @13 = horzcat(@8, @9, @10) */
  rr=w13;
  *rr++ = w8;
  *rr++ = w9;
  *rr++ = w10;
  /* #34: @13 = @13' */
  /* #35: @6 = horzcat(@11, @12, @13) */
  rr=w6;
  for (i=0, cs=w11; i<3; ++i) *rr++ = *cs++;
  for (i=0, cs=w12; i<3; ++i) *rr++ = *cs++;
  for (i=0, cs=w13; i<3; ++i) *rr++ = *cs++;
  /* #36: @14 = @6' */
  for (i=0, rr=w14, cs=w6; i<3; ++i) for (j=0; j<3; ++j) rr[i+j*3] = *cs++;
  /* #37: @5 = mac(@14,@7,@5) */
  for (i=0, rr=w5; i<1; ++i) for (j=0; j<3; ++j, ++rr) for (k=0, ss=w14+j, tt=w7+i*3; k<3; ++k) *rr += ss[k*3]**tt++;
  /* #38: @4 = (@4-@5) */
  for (i=0, rr=w4, cs=w5; i<3; ++i) (*rr++) -= (*cs++);
  /* #39: @0 = (@0*@4) */
  for (i=0, rr=w0, cs=w4; i<3; ++i) (*rr++) *= (*cs++);
  /* #40: output[0][0] = @0 */
  casadi_copy(w0, 3, res[0]);
  /* #41: @8 = 1 */
  w8 = 1.;
  /* #42: @9 = input[0][5] */
  w9 = arg[0] ? arg[0][5] : 0;
  /* #43: @10 = cos(@9) */
  w10 = cos( w9 );
  /* #44: @10 = (@1*@10) */
  w10  = (w1*w10);
  /* #45: @8 = (@8-@10) */
  w8 -= w10;
  /* #46: @10 = sin(@9) */
  w10 = sin( w9 );
  /* #47: @10 = (@2*@10) */
  w10  = (w2*w10);
  /* #48: @8 = (@8+@10) */
  w8 += w10;
  /* #49: @10 = 3.5 */
  w10 = 3.5000000000000000e+00;
  /* #50: @10 = (@10*@3) */
  w10 *= w3;
  /* #51: @15 = sin(@9) */
  w15 = sin( w9 );
  /* #52: @10 = (@10*@15) */
  w10 *= w15;
  /* #53: @8 = (@8+@10) */
  w8 += w10;
  /* #54: output[0][1] = @8 */
  if (res[0]) res[0][3] = w8;
  /* #55: @8 = sin(@9) */
  w8 = sin( w9 );
  /* #56: @1 = (@1*@8) */
  w1 *= w8;
  /* #57: @1 = (-@1) */
  w1 = (- w1 );
  /* #58: @8 = cos(@9) */
  w8 = cos( w9 );
  /* #59: @2 = (@2*@8) */
  w2 *= w8;
  /* #60: @1 = (@1-@2) */
  w1 -= w2;
  /* #61: @2 = 3.5 */
  w2 = 3.5000000000000000e+00;
  /* #62: @2 = (@2*@3) */
  w2 *= w3;
  /* #63: @9 = cos(@9) */
  w9 = cos( w9 );
  /* #64: @2 = (@2*@9) */
  w2 *= w9;
  /* #65: @1 = (@1-@2) */
  w1 -= w2;
  /* #66: output[0][2] = @1 */
  if (res[0]) res[0][4] = w1;
  /* #67: output[0][3] = @3 */
  if (res[0]) res[0][5] = w3;
  /* #68: @3 = input[1][0] */
  w3 = arg[1] ? arg[1][0] : 0;
  /* #69: output[0][4] = @3 */
  if (res[0]) res[0][6] = w3;
  /* #70: @3 = input[1][1] */
  w3 = arg[1] ? arg[1][1] : 0;
  /* #71: output[0][5] = @3 */
  if (res[0]) res[0][7] = w3;
  return 0;
}

CASADI_SYMBOL_EXPORT int simple_model_expl_ode_fun(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int simple_model_expl_ode_fun_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int simple_model_expl_ode_fun_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void simple_model_expl_ode_fun_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int simple_model_expl_ode_fun_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void simple_model_expl_ode_fun_release(int mem) {
}

CASADI_SYMBOL_EXPORT void simple_model_expl_ode_fun_incref(void) {
}

CASADI_SYMBOL_EXPORT void simple_model_expl_ode_fun_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int simple_model_expl_ode_fun_n_in(void) { return 3;}

CASADI_SYMBOL_EXPORT casadi_int simple_model_expl_ode_fun_n_out(void) { return 1;}

CASADI_SYMBOL_EXPORT casadi_real simple_model_expl_ode_fun_default_in(casadi_int i) {
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* simple_model_expl_ode_fun_name_in(casadi_int i) {
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* simple_model_expl_ode_fun_name_out(casadi_int i) {
  switch (i) {
    case 0: return "o0";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* simple_model_expl_ode_fun_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s1;
    case 2: return casadi_s2;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* simple_model_expl_ode_fun_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int simple_model_expl_ode_fun_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 6;
  if (sz_res) *sz_res = 2;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 49;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
