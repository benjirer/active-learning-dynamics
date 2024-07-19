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
static const casadi_int casadi_s1[12] = {8, 1, 0, 8, 0, 1, 2, 3, 4, 5, 6, 7};

/* dual_quaternion_fk:(i0[6])->(o0[8]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a2, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a3, a30, a31, a32, a4, a5, a6, a7, a8, a9;
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
  a9=(a2*a3);
  a6=(a6/a1);
  a6=sin(a6);
  a10=(a9*a6);
  a8=(a8-a10);
  a10=arg[0]? arg[0][3] : 0;
  a11=(a10/a1);
  a11=cos(a11);
  a12=(a8*a11);
  a0=(a0/a1);
  a0=sin(a0);
  a13=(a0*a3);
  a14=(a13*a7);
  a15=(a0*a4);
  a16=(a15*a6);
  a14=(a14+a16);
  a10=(a10/a1);
  a10=sin(a10);
  a16=(a14*a10);
  a12=(a12+a16);
  a16=arg[0]? arg[0][4] : 0;
  a17=(a16/a1);
  a17=cos(a17);
  a18=(a12*a17);
  a19=(a5*a6);
  a20=(a9*a7);
  a19=(a19+a20);
  a20=(a19*a11);
  a21=(a15*a7);
  a22=(a13*a6);
  a21=(a21-a22);
  a22=(a21*a10);
  a20=(a20+a22);
  a16=(a16/a1);
  a16=sin(a16);
  a22=(a20*a16);
  a18=(a18-a22);
  a22=arg[0]? arg[0][5] : 0;
  a23=(a22/a1);
  a23=sin(a23);
  a24=(a18*a23);
  a25=(a8*a10);
  a26=(a14*a11);
  a25=(a25-a26);
  a26=(a25*a17);
  a27=(a21*a11);
  a28=(a19*a10);
  a27=(a27-a28);
  a28=(a27*a16);
  a26=(a26-a28);
  a22=(a22/a1);
  a22=cos(a22);
  a1=(a26*a22);
  a24=(a24+a1);
  if (res[0]!=0) res[0][0]=a24;
  a12=(a12*a16);
  a20=(a20*a17);
  a12=(a12+a20);
  a20=(a12*a22);
  a25=(a25*a16);
  a27=(a27*a17);
  a25=(a25+a27);
  a27=(a25*a23);
  a20=(a20+a27);
  if (res[0]!=0) res[0][1]=a20;
  a25=(a25*a22);
  a12=(a12*a23);
  a25=(a25-a12);
  if (res[0]!=0) res[0][2]=a25;
  a18=(a18*a22);
  a26=(a26*a23);
  a18=(a18-a26);
  if (res[0]!=0) res[0][3]=a18;
  a18=2.0165000000000000e-01;
  a26=(a18*a10);
  a25=(a14*a26);
  a18=(a18*a11);
  a12=(a8*a18);
  a25=(a25+a12);
  a12=3.7499999999999999e-02;
  a20=(a12*a11);
  a27=(a19*a20);
  a25=(a25+a27);
  a12=(a12*a10);
  a27=(a21*a12);
  a25=(a25-a27);
  a27=1.6925000000000001e-01;
  a24=(a27*a7);
  a1=(a5*a24);
  a27=(a27*a6);
  a28=(a9*a27);
  a1=(a1+a28);
  a28=1.4599999999999999e-01;
  a29=(a28*a2);
  a30=(a29*a4);
  a31=9.4000000000000000e-02;
  a2=(a31*a2);
  a32=(a2*a3);
  a30=(a30-a32);
  a32=(a30*a7);
  a29=(a29*a3);
  a2=(a2*a4);
  a29=(a29+a2);
  a2=(a29*a6);
  a32=(a32-a2);
  a1=(a1+a32);
  a32=(a1*a11);
  a28=(a28*a0);
  a2=(a3*a28);
  a31=(a31*a0);
  a0=(a4*a31);
  a2=(a2-a0);
  a0=(a7*a2);
  a31=(a31*a3);
  a28=(a28*a4);
  a31=(a31+a28);
  a28=(a6*a31);
  a0=(a0+a28);
  a28=(a24*a13);
  a4=(a27*a15);
  a28=(a28-a4);
  a0=(a0+a28);
  a28=(a0*a10);
  a32=(a32+a28);
  a25=(a25+a32);
  a32=(a25*a17);
  a28=(a8*a20);
  a4=(a14*a12);
  a28=(a28-a4);
  a4=(a19*a18);
  a28=(a28-a4);
  a4=(a21*a26);
  a28=(a28-a4);
  a5=(a5*a27);
  a9=(a9*a24);
  a5=(a5-a9);
  a30=(a30*a6);
  a29=(a29*a7);
  a30=(a30+a29);
  a5=(a5+a30);
  a30=(a5*a11);
  a13=(a13*a27);
  a15=(a15*a24);
  a13=(a13+a15);
  a2=(a2*a6);
  a31=(a31*a7);
  a2=(a2-a31);
  a13=(a13+a2);
  a2=(a13*a10);
  a30=(a30-a2);
  a28=(a28+a30);
  a30=(a28*a16);
  a32=(a32-a30);
  a30=(a32*a22);
  a0=(a11*a0);
  a1=(a10*a1);
  a0=(a0-a1);
  a1=(a18*a14);
  a2=(a26*a8);
  a1=(a1-a2);
  a2=(a12*a19);
  a1=(a1-a2);
  a2=(a20*a21);
  a1=(a1-a2);
  a0=(a0+a1);
  a1=(a17*a0);
  a8=(a8*a12);
  a14=(a14*a20);
  a8=(a8+a14);
  a19=(a19*a26);
  a8=(a8-a19);
  a21=(a21*a18);
  a8=(a8+a21);
  a13=(a13*a11);
  a5=(a5*a10);
  a13=(a13+a5);
  a8=(a8+a13);
  a13=(a16*a8);
  a1=(a1-a13);
  a13=(a1*a23);
  a30=(a30+a13);
  if (res[0]!=0) res[0][4]=a30;
  a0=(a0*a16);
  a8=(a8*a17);
  a0=(a0+a8);
  a8=(a0*a22);
  a25=(a25*a16);
  a28=(a28*a17);
  a25=(a25+a28);
  a28=(a25*a23);
  a8=(a8+a28);
  if (res[0]!=0) res[0][5]=a8;
  a25=(a25*a22);
  a0=(a0*a23);
  a25=(a25-a0);
  if (res[0]!=0) res[0][6]=a25;
  a22=(a22*a1);
  a23=(a23*a32);
  a22=(a22-a23);
  if (res[0]!=0) res[0][7]=a22;
  return 0;
}

CASADI_SYMBOL_EXPORT int dual_quaternion_fk(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int dual_quaternion_fk_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int dual_quaternion_fk_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void dual_quaternion_fk_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int dual_quaternion_fk_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void dual_quaternion_fk_release(int mem) {
}

CASADI_SYMBOL_EXPORT void dual_quaternion_fk_incref(void) {
}

CASADI_SYMBOL_EXPORT void dual_quaternion_fk_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int dual_quaternion_fk_n_in(void) { return 1;}

CASADI_SYMBOL_EXPORT casadi_int dual_quaternion_fk_n_out(void) { return 1;}

CASADI_SYMBOL_EXPORT casadi_real dual_quaternion_fk_default_in(casadi_int i) {
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* dual_quaternion_fk_name_in(casadi_int i) {
  switch (i) {
    case 0: return "i0";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* dual_quaternion_fk_name_out(casadi_int i) {
  switch (i) {
    case 0: return "o0";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* dual_quaternion_fk_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* dual_quaternion_fk_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s1;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int dual_quaternion_fk_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 1;
  if (sz_res) *sz_res = 1;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}

CASADI_SYMBOL_EXPORT int dual_quaternion_fk_work_bytes(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 1*sizeof(const casadi_real*);
  if (sz_res) *sz_res = 1*sizeof(casadi_real*);
  if (sz_iw) *sz_iw = 0*sizeof(casadi_int);
  if (sz_w) *sz_w = 0*sizeof(casadi_real);
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
