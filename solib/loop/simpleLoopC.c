#include <stddef.h>
#include <sys/types.h>

// gcc  -O0 -Wall -Wextra -pedantic -Warray-bounds  -fPIC -shared -o simpleLoopC.so simpleLoopC.c

void loopTestC(const size_t n, const double values[], double output[])
{
  size_t i;

  for (i=0; i<n; i++) {
      output[i] = values[i];
  }
}
