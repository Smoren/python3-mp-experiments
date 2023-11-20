#include <stdio.h>
#include <math.h>

// gcc -shared -Wl,-soname,distance -o distance.so -fPIC distance.c

double l2norm(double x, double y) {
    return sqrt(x*x + y*y);
}

double l2distance(double x1, double y1, double x2, double y2) {
    return l2norm(x2-x1, y2-y1);
}

//int main() {
//    double x1 = 1, y1 = 2, x2 = 3, y2 = 5;
//    double result = l2distance(x1, y1, x2, y2);
//    printf("distance: %f\n", result);
//
//    return 0;
//}
