/* This file was automatically generated by CasADi 3.6.5.
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
  #define CASADI_PREFIX(ID) jit_ ## ID
#endif

#include <math.h>

#ifndef casadi_real
#define casadi_real double
#endif

#ifndef casadi_int
#define casadi_int long long int
#endif

/* Add prefix to internal symbols */
#define casadi_f0 CASADI_PREFIX(f0)
#define casadi_s0 CASADI_PREFIX(s0)
#define casadi_s1 CASADI_PREFIX(s1)

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

static const casadi_int casadi_s0[10] = {6, 1, 0, 6, 0, 1, 2, 3, 4, 5};
static const casadi_int casadi_s1[8] = {4, 1, 0, 4, 0, 1, 2, 3};

/* quaternion_fk:(i0[6])->(o0[4]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a10, a11, a12, a13, a14, a2, a3, a4, a5, a6, a7, a8, a9;
  a0=arg[0]? arg[0][0] : 0;
  a1=2.;
  a2=(a0/a1);
  a2=cos(a2);
  a3=arg[0]? arg[0][1] : 0;
  a4=(a3/a1);
  a4=cos(a4);
  a5=(a2*a4);
  a6=arg[0]? arg[0][2] : 0;
  a7=(a6/a1);
  a7=cos(a7);
  a8=(a5*a7);
  a3=(a3/a1);
  a3=sin(a3);
  a2=(a2*a3);
  a6=(a6/a1);
  a6=sin(a6);
  a9=(a2*a6);
  a8=(a8-a9);
  a9=arg[0]? arg[0][3] : 0;
  a10=(a9/a1);
  a10=cos(a10);
  a11=(a8*a10);
  a0=(a0/a1);
  a0=sin(a0);
  a3=(a0*a3);
  a12=(a3*a7);
  a0=(a0*a4);
  a4=(a0*a6);
  a12=(a12+a4);
  a9=(a9/a1);
  a9=sin(a9);
  a4=(a12*a9);
  a11=(a11+a4);
  a4=arg[0]? arg[0][4] : 0;
  a13=(a4/a1);
  a13=cos(a13);
  a14=(a11*a13);
  a5=(a5*a6);
  a2=(a2*a7);
  a5=(a5+a2);
  a2=(a5*a10);
  a0=(a0*a7);
  a3=(a3*a6);
  a0=(a0-a3);
  a3=(a0*a9);
  a2=(a2+a3);
  a4=(a4/a1);
  a4=sin(a4);
  a3=(a2*a4);
  a14=(a14-a3);
  a3=arg[0]? arg[0][5] : 0;
  a6=(a3/a1);
  a6=sin(a6);
  a7=(a14*a6);
  a8=(a8*a9);
  a12=(a12*a10);
  a8=(a8-a12);
  a12=(a8*a13);
  a0=(a0*a10);
  a5=(a5*a9);
  a0=(a0-a5);
  a5=(a0*a4);
  a12=(a12-a5);
  a3=(a3/a1);
  a3=cos(a3);
  a1=(a12*a3);
  a7=(a7+a1);
  if (res[0]!=0) res[0][0]=a7;
  a11=(a11*a4);
  a2=(a2*a13);
  a11=(a11+a2);
  a2=(a11*a3);
  a8=(a8*a4);
  a0=(a0*a13);
  a8=(a8+a0);
  a0=(a8*a6);
  a2=(a2+a0);
  if (res[0]!=0) res[0][1]=a2;
  a8=(a8*a3);
  a11=(a11*a6);
  a8=(a8-a11);
  if (res[0]!=0) res[0][2]=a8;
  a14=(a14*a3);
  a12=(a12*a6);
  a14=(a14-a12);
  if (res[0]!=0) res[0][3]=a14;
  return 0;
}

CASADI_SYMBOL_EXPORT int quaternion_fk(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int quaternion_fk_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int quaternion_fk_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void quaternion_fk_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int quaternion_fk_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void quaternion_fk_release(int mem) {
}

CASADI_SYMBOL_EXPORT void quaternion_fk_incref(void) {
}

CASADI_SYMBOL_EXPORT void quaternion_fk_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int quaternion_fk_n_in(void) { return 1;}

CASADI_SYMBOL_EXPORT casadi_int quaternion_fk_n_out(void) { return 1;}

CASADI_SYMBOL_EXPORT casadi_real quaternion_fk_default_in(casadi_int i) {
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* quaternion_fk_name_in(casadi_int i) {
  switch (i) {
    case 0: return "i0";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* quaternion_fk_name_out(casadi_int i) {
  switch (i) {
    case 0: return "o0";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* quaternion_fk_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* quaternion_fk_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s1;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int quaternion_fk_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 1;
  if (sz_res) *sz_res = 1;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}

CASADI_SYMBOL_EXPORT int quaternion_fk_work_bytes(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 1*sizeof(const casadi_real*);
  if (sz_res) *sz_res = 1*sizeof(casadi_real*);
  if (sz_iw) *sz_iw = 0*sizeof(casadi_int);
  if (sz_w) *sz_w = 0*sizeof(casadi_real);
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
