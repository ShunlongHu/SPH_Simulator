__kernel void add_vec_gpu(__global const int *a, __global const int *b, __global int *res, const int len) {
	const int idx = (get_global_id(0)*256+get_global_id(1));
	if (idx < len)
		res[idx] = a[idx] + b[idx];
}