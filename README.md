# CS203_Fa20_Project: Threaded, Vectorized Matrix Multiplications.

## Overview

Matrix multiplication is the basic element in linear algebra and many compute-intensive applications. The renaissance of deep neural networks makes matrix multiplications more important these days. 
In this project you will be responsible for improving the performance of a vectorized, multithreaded matrix multiplication function. The baseline function already uses block algorithm, Intel's SSE4.1 SIMD instructions and partition tasks into different parallel threads. 

This project will require you to implement a function mythreaded_vector_blockmm that beats the performance of the original threaded_vector_blockmm function. You may use any architectural features available on the given machine to implement the mythreaded_vector_blockmm function. 

As hints, you may start from the following papers.

1. A. Douillet and G. R. Gao, "Software-Pipelining on Multi-Core Architectures," 16th International Conference on Parallel Architecture and Compilation Techniques (PACT 2007), Brasov, 2007, pp. 39-48, doi: 10.1109/PACT.2007.4336198.

2. Md Kamruzzaman, Steven Swanson, and Dean M. Tullsen. 2010. Software data spreading: leveraging distributed caches to improve single thread performance. In Proceedings of the 31st ACM SIGPLAN Conference on Programming Language Design and Implementation (PLDI '10). Association for Computing Machinery, New York, NY, USA, 460–470. DOI:https://doi.org/10.1145/1806596.1806648

3. Raehyun Kim, Jaeyoung Choi, and Myungho Lee. 2019. Optimizing parallel GEMM routines using auto-tuning with Intel AVX-512. In Proceedings of the International Conference on High Performance Computing in Asia-Pacific Region (HPC Asia 2019). Association for Computing Machinery, New York, NY, USA, 101–110. DOI:https://doi.org/10.1145/3293320.3293334


## How to start

You will be given the source code containing the threaded_vector_blockmm function and a mythreaded_vector_blockmm that is currently identical to threaded_vector_blockmm of the project. You should clone this repo using 

git clone https://github.com/hungweitseng/CS203_Fa20_Project

You should develop this project under a Linux machine. We will run your project on  ti-05.cs.ucr.edu. As a CSE student, you should have access to it. If not, please acquire an account from CSE's IT at systems@cs.ucr.edu. Please don't contact us for account issues. 

Once you clone the repo, you will find a few files in there. The build the executable file, you need to use "make" command in each directory. Once you finish making the files, you will see an executable file called "project". It accepts a few arguments -- the array size, the N in block algorithm, and the nuber_of_threads available. By default, we will test your program using

./project 4096 512 4

Your task will be to design and implement the mythreaded_vector_blockmm function located in the myblockmm.c file. The effectiveness of your mythreaded_vector_blockmm will be tested against the threaded_vector_blockmm in baseline.c. You will also compete against your fellow classmates for amazing awards and prizes! You will get full credit if you can speedup the mythreaded_vector_blockmm by 2x. If not, your grades will be the speedup_over_baseline * 40 + 20. You have the complete freedom to implement any mechanism that you think might help improve the performance of mythreaded_vector_blockmm in the myblockmm.c file. However, as you're only allowed to turnin myblockmm.c, any modification you made in other files will be omitted. 

## Deliverables

You only need to turn your modified "myblockmm.c" file and please put your name and student ID # in the name and SID char arrays in the myblockmm.c file.

## Due 

11:59pm, 12/7/2020. No extension. No late submission accepted.

