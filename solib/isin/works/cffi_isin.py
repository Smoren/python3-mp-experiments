import cffi

ffi = cffi.FFI()
defs = "void isin(int64_t where[], int64_t where_size, int64_t what[], int64_t what_size, bool result[]);"
ffi.cdef(defs, override=True)
source = """
#include <stdbool.h>

void isin(int64_t where[], int64_t where_size, int64_t what[], int64_t what_size, bool result[]) {
    result = (bool*)malloc(sizeof(bool) * where_size);
    for (int64_t i = 0; i < where_size; i++) {
        result[i] = false;
    }

    if (what_size == 0) return;

    int64_t what_min = what[0], what_max = what[0];
    for (int64_t i = 1; i < what_size; i++) {
        if (what[i] > what_max) what_max = what[i];
        else if (what[i] < what_min) what_min = what[i];
    }
    int64_t what_range = what_max - what_min;

    int64_t* what_normalized = (int64_t*)malloc(sizeof(int64_t) * what_size + 1);
    for (int64_t i = 0; i < what_size; i++) {
        what_normalized[i] = what[i] - what_min;
    }

    bool* isin_helper_ar = (bool*)malloc(sizeof(bool) * what_range + 1);
    for (int64_t i = 0; i <= what_range; i++) {
        isin_helper_ar[i] = false;
    }
    for (int64_t i = 0; i < what_size; i++) {
        isin_helper_ar[what_normalized[i]] = true;
    }

    for (int64_t i = 0; i < where_size; i++) {
        if (where[i] > what_max || where[i] < what_min) continue;
        result[i] = isin_helper_ar[where[i] - what_min];
    }

    free(what_normalized);
    free(isin_helper_ar);

    for (int64_t i = 0; i < where_size; i++) {
        printf("result[%d] = %d\\n", i, (int)result[i]);
    }
}
"""
ffi.set_source(module_name="cffi_isin", source=source)
ffi.compile()
