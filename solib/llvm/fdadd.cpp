#include <stdlib.h>

// g++ -Wall -fPIC -O2 -c add.cpp && g++ -shared -o add.so add.o

#ifdef __cplusplus
extern "C" {
#endif

double fpadd(double a, double b) {
    return a + b + 100;
}

#ifdef __cplusplus
}
#endif
