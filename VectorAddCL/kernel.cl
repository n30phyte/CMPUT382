__kernel void vector_add(__global const float *A, __global const float *B, __global float *C) {
    int idx = get_global_id(0);
    C[idx] = A[idx] + B[idx];
}