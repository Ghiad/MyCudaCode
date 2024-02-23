## 主函数在kernel.cu中，其他文件是不同步骤的实现
主要参数,m=n=k=2048,bm=bn=128,bk=tx=ty=8。\
Sgemm1是navieSgemm，没有任何的优化,为cublas的11%\
Sgemm2是采用了Shared memory，减少global memory读取的实现,一个线程计算一个数，为cublas的12%\
Sgemm3是采用了寄存器，减少shared memory读取，使用float4读取数据的实现，一个线程计算tx*ty个数,为cublas的75%\
Sgemm4是采用了双缓冲预取的操作实现,为cublas的82%\
Sgemm5是减少shared memory bank conflict的实现,为cublas的89%
