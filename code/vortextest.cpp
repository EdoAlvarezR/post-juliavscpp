// compile as `g++ -std=c++14 -o vortextest vortextest.cpp`
#include "vortextest.h"

int main() {                        // Program entry point

  int ntests = 100;
  int n = 3;
  float_t lambda = 1.5;

  std::vector<real_t> res;

  Particles particles = generatelattice(n, lambda);

  if (true) printparticles(&(particles[0]), n*n*n);

  auto t_start = std::chrono::high_resolution_clock::now();
  P2P(&(particles[0]), n*n*n);
  auto t_end = std::chrono::high_resolution_clock::now();

  if (true) printUJ(&(particles[0]), n*n*n);

  res = benchmarkP2P(ntests, &(particles[0]), n*n*n);
  cout << "\nSamples:\t" << ntests << "\n";
  cout << "min time:\t" << res[0] << " ms" << "\n";
  cout << "ave time:\t" << res[1] << " ms" << "\n";
  cout << "max time:\t" << res[2] << " ms" << "\n";

  // float_t min_t = benchmarkP2P_wrap(ntests, n, lambda);

  cout << "hello, world" << endl;
  return 0;
}
