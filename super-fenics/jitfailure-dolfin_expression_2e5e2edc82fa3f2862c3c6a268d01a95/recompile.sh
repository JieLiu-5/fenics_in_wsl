#!/bin/bash
# Execute this file to recompile locally
c++ -Wall -shared -fPIC -std=c++11 -O3 -fno-math-errno -fno-trapping-math -ffinite-math-only -I/usr/lib/slepcdir/3.7.4/x86_64-linux-gnu-real/include -I/usr/include/scotch -I/usr/lib/x86_64-linux-gnu/hdf5/openmpi/include -I/usr/include/suitesparse -I/usr/include/superlu -I/usr/include/hypre -I/usr/lib/petscdir/3.7.7/x86_64-linux-gnu-real/include -I/usr/lib/x86_64-linux-gnu/openmpi/include -I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi/opal/mca/event/libevent2022/libevent/include -I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi/opal/mca/event/libevent2022/libevent -I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi -I/usr/include/hdf5/openmpi -I/usr/include/eigen3 -I/usr/lib/python3/dist-packages/ffc/backends/ufc -I/home/wsl_jie/.cache/dijitso/include dolfin_expression_2e5e2edc82fa3f2862c3c6a268d01a95.cpp -L/usr/lib/x86_64-linux-gnu/openmpi/lib -L/usr/lib/petscdir/3.7.7/x86_64-linux-gnu-real/lib -L/usr/lib/slepcdir/3.7.4/x86_64-linux-gnu-real/lib -L/usr/lib/x86_64-linux-gnu/hdf5/openmpi -L/home/wsl_jie/.cache/dijitso/lib -Wl,-rpath,/home/wsl_jie/.cache/dijitso/lib -lmpi -lmpi_cxx -lpetsc_real -lslepc_real -lm -ldl -lz -lsz -lhdf5 -lboost_timer -ldolfin -olibdijitso-dolfin_expression_2e5e2edc82fa3f2862c3c6a268d01a95.so