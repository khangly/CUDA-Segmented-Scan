#pragma once

#include <vector>
#include <unistd.h>
#include <cassert>

std::vector<float> gen_vec(size_t n) {
  std::vector<float> v;
  v.reserve(n);

  for (size_t i = 0; i < n; i++) {
    // v.push_back(drand48());
    v.push_back((float) i);
  }

  return v;
}

std::vector<float> gen_mat(size_t m, size_t n) {
  return gen_vec(m*n);
}

bool compare(const std::vector<float>& x, const std::vector<float>& y) {
  float eps = 1.0e-4;
  assert(x.size() == y.size());
  for (size_t i = 0; i < x.size(); i++) {
    float diff = fabsf(x[i] - y[i]);
    float a = fabsf(x[i]);
    float b = fabsf(y[i]);
    float max = std::max(a, b);

    if (diff > max*eps) {
      printf("mismatch at %d\n",i);
      return false;
    }
  }
  return true;
}


void print(const std::vector <float>& v) {

  if (v.size() == 0) printf("{}");
  else {
    printf("{");
    for(int i=0; i < v.size()-1; i++) {
      printf("%1.1f, ", v[i]);
    }
    printf("%1.1f}", v[v.size()-1]);
  }

}

void checkResult (std::vector <float>& x,
		  std::vector <float>& rv, std::vector <float>& rv_cpu,
		  float duration_gpu, float duration_cpu) {
  bool correct = compare(rv, rv_cpu);

  if (correct) {
    printf("scan OK!\n");
    printf("%lf ms GPU, %lf ms CPU\n",
	   duration_gpu*1000, duration_cpu*1000);
  } else {
    printf("scan FAILURE!\n");
  }
    printf("input:     ");
    print(x);
    printf("\nexpected: ");
    print(rv_cpu);
    printf("\nfound:    ");
    print(rv);
    printf("\n");
    // }
}
