#pragma once

#include <vector>

std::vector<float> cpu_vecprod(const std::vector<float>& x,
                               const std::vector<float>& y) {
  assert(x.size() == y.size());

  std::vector<float> r(x.size());

  for (size_t i = 0; i < x.size(); i++) {
    r[i] = x[i]*y[i];
  }
  return r;
}

void cpu_mul2(std::vector<float>& x) {
  for (size_t i = 0; i < x.size(); i++) {
  	x[i] *= 2;
  }
}

float cpu_reduce(const std::vector<float>& x) {
  float sum = 0;
  for (size_t i = 0; i < x.size(); i++) {
    sum += x[i];
  }
  return sum;
}

std::vector<float> cpu_scan(const std::vector<float>& x) {
  std::vector<float> out (x.size(), 0.0);
  if (x.size() > 0) out[0] = x[0];
  for (size_t i = 1; i < out.size(); i++) {
    out[i] = x[i] + out[i-1];
  }
  return out;
}

std::vector<float> cpu_segscan(const std::vector<float>& x,
			    const std::vector<float>& flags) {
  std::vector<float> out (x.size(), 0.0);
  if (x.size() > 0) out[0] = x[0];
  for (size_t i = 1; i < out.size(); i++) {
    if (flags[i] == 1.0) {
      out[i] = x[i];
    } else {
      out[i] = x[i] + out[i-1];
    }
  }
  return out;
}

std::vector<float> cpu_spmv(const std::vector<float>& a, const std::vector<float>& x, int m, int n) {
  assert(x.size() == n);
  assert(a.size() == m*n);

  std::vector<float> b(n);

  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < n; j++) {
      b[i] += a[i*n + j]*x[j];
    }
  }
  return b;
}
