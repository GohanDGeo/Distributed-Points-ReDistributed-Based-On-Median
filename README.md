# Distributed-Points-ReDistributed-Based-On-Median
Second project for the course Parallel and Distributed System, ECE AUTH, 2021-2022


Task: 
Given a system of p processes and a dataset of N points, with each process having N/p points,
choose a random point from one process (pivot), and redistribute the points so that each process has N/p points
with distances from the pivot less (or equal) from the distances of the points in the next process. 
So, the maximum distance from process i will be less or equal from the minimum distance in process i+1.

The /data folder contains only one test.bin file, only containing 8 points, purely for testing. 
Works with any column-major data file in bin format, as long as the first and secnod 64bits represent the number of points and the dimension of the points, and the rest are double/float64 represent numbers.

The code has been tested using the mnist, fashionMnist and CIFAR10 datasets. 
To test it on those datasets, just put the bin files in the /data folder, and change the bin file name in the code.

To run:

```
 mpiCC distrPointsMedian.cpp  -o exec
 export OMPI_MCA_btl_vader_single_copy_mechanism=none ; mpiexec -np <p> ./exec
```
where <p> the number of processes to run the code for.

If the number of points N is not divisible by the number of processes p, an error is thrown and the program terminates.

 Author: Georgios Koutroumpis
