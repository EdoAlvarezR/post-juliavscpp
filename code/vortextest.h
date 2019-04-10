// Math functions
#include <cmath>
#include <vector>
#include "vec.h"

// IO stream and timer
#include <iostream>
#include <iomanip>
#include <chrono>

using namespace std;


typedef double real_t;                        // Floating point precision

typedef vec<3,real_t> vec3;                   // Vector of 3 real_t types
typedef vec<9,real_t> vec9;                   // Vector of 9 real_t types

const real_t const4 = 1/(4*M_PI);

//! Particle struct
struct Particle {
  // Properties
  vec3 X;                                     // Position
  vec3 Gamma;                                 // Vectorial circulation
  real_t sigma;                               // Vortex blob radius

  // Calculations
  vec3 U;                                     // Velocity
  vec9 J;                                     // Jacobian of velocity
};

typedef std::vector<Particle> Particles;      // Vector of bodies


// Generate a square box of sides l with n particles per edge and core overlap
// lambda=sigma/dx.
Particles generatelattice(int n, real_t lambda, real_t l=1.0){

  Particles particles(n*n*n);
  real_t dx = l/(n-1);              // Spacing
  real_t sigma = dx*lambda;     // Smoothing radius

  int ind = 0;

  for (int k=0; k<n; ++k){
    for (int j=0; j<n; ++j){
      for (int i=0; i<n; ++i){

        particles[ind].X[0] = dx*i;
        particles[ind].X[1] = dx*j;
        particles[ind].X[2] = dx*k;

        for (int h=0; h<3; ++h) particles[ind].Gamma[h] = 1.0;

        particles[ind].sigma = sigma;

        for (int h=0; h<3; ++h) particles[ind].U[h] = 0.0;
        for (int h=0; h<9; ++h) particles[ind].J[h] = 0.0;

        ind++;
      }
    }
  }

  return particles;
}


// Regularized particle-to-particle Winckelmann's kernel
void P2P(Particle * P, const int numParticles) {

  real_t r, ros, aux, g_sgm, dgdr;
  vec3 dX;

  for (int i=0; i<numParticles; i++) {
    for (int j=0; j<numParticles; j++) {

      dX[0] = P[i].X[0] - P[j].X[0];
      dX[1] = P[i].X[1] - P[j].X[1];
      dX[2] = P[i].X[2] - P[j].X[2];
      r = sqrt(dX[0]*dX[0] + dX[1]*dX[1] + dX[2]*dX[2]);

      if (r!=0){
          ros = r/P[j].sigma;

          // Evaluate g_σ and ∂gσ∂r
          aux = pow(ros*ros + 1.0, 2.5);
          g_sgm = ros*ros*ros * (ros*ros + 2.5) / aux;
          dgdr = 7.5 * ros*ros / ( aux * (ros*ros + 1.0) );

          // u(x) = ∑g_σ(x−xp) K(x−xp) × Γp
          aux = (- const4 / (r*r*r)) * g_sgm;
          P[i].U[0] += ( dX[1]*P[j].Gamma[2] - dX[2]*P[j].Gamma[1] ) * aux;
          P[i].U[1] += ( dX[2]*P[j].Gamma[0] - dX[0]*P[j].Gamma[2] ) * aux;
          P[i].U[2] += ( dX[0]*P[j].Gamma[1] - dX[1]*P[j].Gamma[0] ) * aux;

          // ∂u∂xj(x) = ∑[ ∂gσ∂xj(x−xp) * K(x−xp)×Γp + gσ(x−xp) * ∂K∂xj(x−xp)×Γp ]
          aux = (- const4 / (r*r*r)) * (dgdr/(P[j].sigma*r) - 3.0*g_sgm/(r*r));
          for (int k=0; k<3; k++){
            P[i].J[3*k + 0] += ( dX[1]*P[j].Gamma[2] - dX[2]*P[j].Gamma[1] )* aux*dX[k];
            P[i].J[3*k + 1] += ( dX[2]*P[j].Gamma[0] - dX[0]*P[j].Gamma[2] )* aux*dX[k];
            P[i].J[3*k + 2] += ( dX[0]*P[j].Gamma[1] - dX[1]*P[j].Gamma[0] )* aux*dX[k];
          }

          // Adds the Kronecker delta term
          aux = - const4 * g_sgm / (r*r*r);
          // k=1
          P[i].J[3*0 + 1] -= aux * P[j].Gamma[2];
          P[i].J[3*0 + 2] += aux * P[j].Gamma[1];
          // k=2
          P[i].J[3*1 + 0] += aux * P[j].Gamma[2];
          P[i].J[3*1 + 2] -= aux * P[j].Gamma[0];
          // k=3
          P[i].J[3*2 + 0] -= aux * P[j].Gamma[1];
          P[i].J[3*2 + 1] += aux * P[j].Gamma[0];
      }

    }
  }

}

std::vector<real_t> benchmarkP2P(int ntests, Particle * P, const int numParticles){

  auto t_start = std::chrono::high_resolution_clock::now();
  auto t_end = std::chrono::high_resolution_clock::now();

  float_t t;
  float_t min_t = 1e+16;
  float_t max_t = 1e-16;
  float_t ave_t = 0;

  for (int i=0; i<ntests; i++){

    t_start = std::chrono::high_resolution_clock::now();
    P2P(P, numParticles);
    t_end = std::chrono::high_resolution_clock::now();

    t = std::chrono::duration<float_t, std::milli>(t_end-t_start).count();

    if (t<=min_t) min_t=t;
    if (t>=max_t) max_t=t;
    ave_t += t;
  }

  ave_t = ave_t/ntests;

  return {min_t, ave_t, max_t};
}

void printparticles(Particle * P, int numParticles){
  std::cout << "\n--------------- PARTICLE POSITION -------------------------------\n";
  std::cout << "#\tX1\tX2\tX3\tsigma\n";
  // Prints the index of every particle
  for (int i=0; i<numParticles; ++i){
    std::cout << std::fixed << std::setprecision(5)
              << i+1 << "\t"
              << P[i].X[0] << "\t"
              << P[i].X[1] << "\t"
              << P[i].X[2] << "\t"
              << P[i].sigma << "\t"
              << "\n";
  }
}


void printUJ(Particle * P, int numParticles){
  std::cout << "\n--------------- U and J ----------------------------------------\n";
  std::cout << "#\tU1\tU2\tU3\tJ1\tJ2\tJ3\tJ4\tJ5\tJ6\tJ7\tJ8\tJ9\n";
  // Prints the index of every particle
  for (int i=0; i<numParticles; ++i){
    std::cout << std::fixed << std::setprecision(2)
              << i+1 << "\t"
              << P[i].U[0] << "\t"
              << P[i].U[1] << "\t"
              << P[i].U[2] << "\t"
              << P[i].J[0] << "\t"
              << P[i].J[1] << "\t"
              << P[i].J[2] << "\t"
              << P[i].J[3] << "\t"
              << P[i].J[4] << "\t"
              << P[i].J[5] << "\t"
              << P[i].J[6] << "\t"
              << P[i].J[7] << "\t"
              << P[i].J[8] << "\t"
              << "\n";
  }
}


real_t benchmarkP2P_wrap(int ntests, int n, float_t lambda, bool verbose=false) {

  std::vector<real_t> res;

  Particles particles = generatelattice(n, lambda);

  res = benchmarkP2P(ntests, &(particles[0]), n*n*n);

  if (verbose){
    cout << "Samples:\t" << ntests << "\n";
    cout << "min time:\t" << res[0] << " ms" << "\n";
    cout << "ave time:\t" << res[1] << " ms" << "\n";
    cout << "max time:\t" << res[2] << " ms" << "\n";
  }

  return res[0];
}
