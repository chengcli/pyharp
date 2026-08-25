#pragma once

// C/C++
#include <cmath>

// base
#include <configure.h>

namespace harp {

template <typename T>
struct Complex {
  T r;
  T i;

  DISPATCH_MACRO Complex() : r(0), i(0) {}
  DISPATCH_MACRO Complex(T real, T imag = 0) : r(real), i(imag) {}
};

template <typename T>
DISPATCH_MACRO Complex<T> operator+(Complex<T> a, Complex<T> b) {
  return {a.r + b.r, a.i + b.i};
}

template <typename T>
DISPATCH_MACRO Complex<T> operator+(Complex<T> a, T b) {
  return {a.r + b, a.i};
}

template <typename T>
DISPATCH_MACRO Complex<T> operator+(T a, Complex<T> b) {
  return {a + b.r, b.i};
}

template <typename T>
DISPATCH_MACRO Complex<T> operator-(Complex<T> a, Complex<T> b) {
  return {a.r - b.r, a.i - b.i};
}

template <typename T>
DISPATCH_MACRO Complex<T> operator-(Complex<T> a, T b) {
  return {a.r - b, a.i};
}

template <typename T>
DISPATCH_MACRO Complex<T> operator-(T a, Complex<T> b) {
  return {a - b.r, -b.i};
}

template <typename T>
DISPATCH_MACRO Complex<T> operator-(Complex<T> a) {
  return {-a.r, -a.i};
}

template <typename T>
DISPATCH_MACRO Complex<T> operator*(Complex<T> a, Complex<T> b) {
  return {a.r * b.r - a.i * b.i, a.r * b.i + a.i * b.r};
}

template <typename T>
DISPATCH_MACRO Complex<T> operator*(Complex<T> a, T b) {
  return {a.r * b, a.i * b};
}

template <typename T>
DISPATCH_MACRO Complex<T> operator*(T a, Complex<T> b) {
  return {a * b.r, a * b.i};
}

template <typename T>
DISPATCH_MACRO Complex<T> operator/(Complex<T> a, Complex<T> b) {
  T const den = b.r * b.r + b.i * b.i;
  return {(a.r * b.r + a.i * b.i) / den, (a.i * b.r - a.r * b.i) / den};
}

template <typename T>
DISPATCH_MACRO Complex<T> operator/(Complex<T> a, T b) {
  return {a.r / b, a.i / b};
}

template <typename T>
DISPATCH_MACRO Complex<T> operator/(T a, Complex<T> b) {
  T const den = b.r * b.r + b.i * b.i;
  return {a * b.r / den, -a * b.i / den};
}

template <typename T>
DISPATCH_MACRO Complex<T> complex_conj(Complex<T> a) {
  return {a.r, -a.i};
}

template <typename T>
DISPATCH_MACRO T complex_norm(Complex<T> a) {
  return a.r * a.r + a.i * a.i;
}

template <typename T>
DISPATCH_MACRO T complex_abs(Complex<T> a) {
  return sqrt(complex_norm(a));
}

template <typename T>
DISPATCH_MACRO Complex<T> complex_exp(Complex<T> a) {
  T const e = exp(a.r);
  return {static_cast<T>(e * cos(a.i)), static_cast<T>(e * sin(a.i))};
}

}  // namespace harp
