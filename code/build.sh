
SRC=/home/edoalvar/Dropbox/FLOWResearch/LabNotebook/Posts/juliacpp/code
BLD=$SRC/build

# Generates the compiled object libhello.cpp.o:
c++ -DJULIA_ENABLE_THREADING -Dhello_EXPORTS \
-I/home/edoalvar/.julia/packages/CxxWrap/KcmSi/deps/usr/include \
-I/home/edoalvar/Programs/julia-1.0.3/include/julia/ \
-std=c++14 -fPIC -o $BLD/vortextest_jlcxx.cpp.o -c $SRC/vortextest_jlcxx.cpp

# Generate the shared library libhello.so:
c++  -fPIC  -std=c++14  \
-shared -Wl,-soname,vortextest_jlcxx.so \
-o $BLD/vortextest_jlcxx.so \
$BLD/vortextest_jlcxx.cpp.o \
-Wl,-rpath,: \
/home/edoalvar/.julia/packages/CxxWrap/KcmSi/deps/usr/lib/libcxxwrap_julia.so.0.5.1 \
/home/edoalvar/Programs/julia-1.0.3/lib/libjulia.so
