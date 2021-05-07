void atomic_add_f(volatile global float* values, const float B)
//ints can be used as theyre faster but floats are better for memory
{
	union { float f; uint i; } oldNum; //share memory location to reduce memory use and not get
	union { float f; uint i; } newNum; 
	do
	{
		oldNum.f = *values;
		newNum.f = oldNum.f + B;
	} while (atom_cmpxchg((volatile global uint*)values, oldNum.i, newNum.i) != oldNum.i);
}

void atomic_max_f(volatile global float* A, const float B)
{
	union { float f; uint i; } oldNum;
	union { float f; uint i; } newNum;
	do
	{
		oldNum.f = *A;
		newNum.f = max(oldNum.f, B);
	} while (atom_cmpxchg((volatile global uint*)A, oldNum.i, newNum.i) != oldNum.i);
}

void atomic_min_f(volatile global float* A, const float B)
{
	union { float f; uint i; } oldNum;
	union { float f; uint i; } newNum;
	do
	{
		oldNum.f = *A;
		newNum.f = min(oldNum.f, B);
	} while (atom_cmpxchg((volatile global uint*)A, oldNum.i, newNum.i) != oldNum.i);
}

kernel void reduce_add(global const float* A, global float* B, local float* scratch) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	//values from global to local memory
	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);//check treads have copied from global to local

	for (int i = N / 2; i > 0; i /= 2) { 
		if (lid < i)					 //make sure i stays within range.
			scratch[lid] += scratch[lid + i];

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	//add local groupd to first element of the array
	//copy the cache to output array
	if (!lid) {
		atomic_add_f(&B[0], scratch[lid]);
	}
}

kernel void reduce_min(global const float* A, global float* B, local float* scratch) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	//values from global to local memory
	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);//check treads have copied from global to local

	for (int i = N / 2; i > 0; i /= 2) { 
		if (lid < i) {
			//check size difference
			if (scratch[lid + i] < scratch[lid])
				scratch[lid] = scratch[lid + i];
		}


		barrier(CLK_LOCAL_MEM_FENCE);
	}

	
	if (!lid) {
		atomic_min_f(&B[0], scratch[lid]);
	}
}

kernel void reduce_max(global const float* A, global float* B, local float* scratch) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);

	for (int i = N / 2; i > 0; i /= 2) { //using coalesced access , i = stride
		if (lid < i) {
			if (scratch[lid + i] > scratch[lid])
				scratch[lid] = scratch[lid + i];
		}

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (!lid) {
		atomic_max_f(&B[0], scratch[lid]);
	}
}

kernel void mean_variance_squared(global const float* A, global float* B, local float* scratch, float mean) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	// variance calculation
	float variance = (A[id] - mean);

	
	scratch[lid] = (variance * variance);

	barrier(CLK_LOCAL_MEM_FENCE); //syncs local memory

	for (int i = N/2; i > 0; i /= 2){
		if (lid < i)
			scratch[lid] += scratch[lid + i];

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	//if this is first element in local memory
	//atomic_add to put elements to output buffer
	if (!lid)
		atomic_add_f(&B[0], scratch[lid]);
}


//this is the bitonic sort, it uses 116779 loops to sort it... 

kernel void bitonic_sort_f(global const float* in, global float* out, local float* scratch, int merge)
{
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int gid = get_group_id(0);
	int N = get_local_size(0);

	int max_group = (get_global_size(0) / N) - 1;
	int offset_id = id + ((N / 2) * merge);

	if (merge && gid == 0)
	{
		out[id] = in[id];

		barrier(CLK_GLOBAL_MEM_FENCE);
	}

	scratch[lid] = in[offset_id];

	barrier(CLK_LOCAL_MEM_FENCE);

	for (int l = 1; l < N; l <<= 1)
	{
		bool direction = ((lid & (l << 1)) != 0);

		for (int inc = l; inc > 0; inc >>= 1)
		{
			int j = lid ^ inc;
			float first_data = scratch[lid];
			float second_data = scratch[j];

			bool smaller = (second_data < first_data) || (second_data == first_data && j < lid);
			bool swap = smaller ^ (j < lid) ^ direction;

			barrier(CLK_LOCAL_MEM_FENCE);

			scratch[lid] = (swap) ? second_data : first_data;

			barrier(CLK_LOCAL_MEM_FENCE);
		}
	}

	out[offset_id] = scratch[lid];

	barrier(CLK_GLOBAL_MEM_FENCE);

	if (merge && gid == max_group)
		out[offset_id] = in[offset_id];
}

// int calulations if float was not to be used

kernel void bitonic_sort(global const int* in, global int* out, local int* scratch, int merge) {
	int id = get_global_id(0);
	int lid = get_local_id(0);

	//get the work group id
	int gid = get_group_id(0);

	int N = get_local_size(0);

	int maxGroup = (get_global_size(0) / N) - 1;

	
	//when N = 1024, offset_id alternates between 0 and 512. 
	int offset_id = id + ((N / 2) * merge);

	
	if (merge && gid == 0)
		out[id] = in[id];
	
	scratch[lid] = in[offset_id];

	//access local memory in a commutative manner
	for (int l = 1; l < N; l <<= 1){
		//set the direction bool for this run of bitonic sort
		bool direction = ((lid & (l << 1)) != 0);

		for (int inc = l; inc > 0; inc >>= 1)
		{
			//compares the two data points and store in first_data and second_data.
			int j = lid ^ inc;
			int first_data = scratch[lid];
			int second_data = scratch[j];

			 
			bool smaller = (second_data < first_data) || (second_data == first_data && j < lid);
			bool swap = smaller ^ (j < lid) ^ direction;

			//place the smallest value within the scratch buffer if swapping
			scratch[lid] = (swap) ? second_data : first_data;
		}
	}

	out[offset_id] = scratch[lid];

	//if a merge is taking place between groups and this group is the last, copy the last N/2 values from the input to the output buffer
	if (merge && gid == maxGroup)
		out[offset_id] = in[offset_id];
}

kernel void reduce_add_int(global const int* A, global int* B, local int* scratch) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	//cahce values from global memory to local memory
	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);//wait for copying

	for (int i = N / 2; i > 0; i /= 2) { 
		if (lid < i)					 
			scratch[lid] += scratch[lid + i];

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	
	// cache copied to output array
	if (!lid) {
		atomic_add(&B[0], scratch[lid]);
	}
}

kernel void reduce_min_int(global const int* A, global int* B, local int* scratch) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);

	for (int i = N / 2; i > 0; i /= 2) { 
		if (lid < i) {
			
			if (scratch[lid + i] < scratch[lid])
				scratch[lid] = scratch[lid + i];
		}

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	
	if (!lid) {
		atomic_min(&B[0], scratch[lid]);
	}
}

kernel void reduce_max_int(global const int* A, global int* B, local int* scratch) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);

	for (int i = N / 2; i > 0; i /= 2) {
		if (lid < i) {
			if (scratch[lid + i] > scratch[lid])
				scratch[lid] = scratch[lid + i];
		}

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (!lid) {
		atomic_max(&B[0], scratch[lid]);
	}
}

kernel void mean_variance_squared_int(global const int* A, global int* B, local int* scratch, float mean) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	
	float variance = (A[id] - mean);

	
	scratch[lid] = (int)(variance * variance) / 10; 

	barrier(CLK_LOCAL_MEM_FENCE); 

	for (int i = N / 2; i > 0; i /= 2) {
		if (lid < i)
			scratch[lid] += scratch[lid + i];

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	
	if (!lid)
		atomic_add_f(&B[0], scratch[lid]);
}