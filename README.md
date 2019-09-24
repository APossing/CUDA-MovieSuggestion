CUDA-MovieSuggestion

This movie suggester was built on Cuda 10.1 Runtime. The code is definitely not my cleanest work, but the point of this project was purely to develop and think about the parallel functions. 

Some improvement I would like to look into for the future
-	Getting block sizes optimal
-	Ping-Ponging with the GPU so all the data transfer doesn’t happen up front and everything is busy, this will lead to a faster solution and can solve a problem that requires more than the 8gb I have available on my GPU.

**Check out [Kernel](MovieSuggestion/MovieSuggestion/kernel.cu) for suggestion implematation**
