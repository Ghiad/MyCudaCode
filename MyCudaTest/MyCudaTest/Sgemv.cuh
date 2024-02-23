#include<cuda_runtime.h>
#include<cuda.h>
#include<bits/stdc++.h>
#include<device_launch_parameters.h>
#include<cusparse.h>

void Sgemv(const IndexType row_num, const IndexType* A_row_offset,
	IndexType* A_col_index, ValueType* A_value, const ValueType* x, ValueType* y);