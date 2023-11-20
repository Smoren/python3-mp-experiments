#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <set>

// g++ -Wall -fPIC -O2 -c isin.cpp && g++ -shared -o isin.so isin.o

#ifdef __cplusplus
extern "C" {
#endif

bool* isin(int where[], int where_size, int what[], int what_size) {
    bool* result = (bool*)malloc(sizeof(bool) * where_size);
    std::set<int> s;
    for (int i = 0; i < what_size; i++) {
        s.insert(what[i]);
    }
    for (int i = 0; i < where_size; i++) {
        result[i] = s.count(where[i]) > 0;
    }
    return result;
}

#ifdef __cplusplus
}
#endif

//int main() {
//    int where[] = {1, 2, 3, 4, 5};
//    int where_size = 5;
//    int what[] = {1, 3, 5};
//    int what_size = 3;
//    bool* in = isin(where, where_size, what, what_size);
//
//    for (int i = 0; i < where_size; i++) {
//        printf("%d: %d\n", i, in[i]);
//    }
//
//    return 0;
//}
