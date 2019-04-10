// To compile run `build.sh`

// Julia wrapper
#include "jlcxx/jlcxx.hpp"

#include "vortextest.h"
// #include "vortextest.cpp"

// Exposing the functions to Julia
JLCXX_MODULE define_julia_module(jlcxx::Module& mod)
{
    mod.method("benchmarkP2P_wrap", &benchmarkP2P_wrap);
}
