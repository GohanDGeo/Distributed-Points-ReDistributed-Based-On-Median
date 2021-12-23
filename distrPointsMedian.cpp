/*
PARALLEL AND DISTRIBUTED SYSTEMS, ECE, AUTH, 2021-2022
PROJECT 2
KOUTROUMPIS GEORGIOS, AEM: 9668

Task: Given a system of p processes and a dataset of N points, with each process having N/p points,
      choose a random point from one process (pivot), and redistribute the points so that each process has N/p points
      with distances from the pivot less (or equal) from the distances of the points in the next process. 
      So, the maximum distance from process i will be less or equal from the minimum distance in process i+1.
*/

#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <mpi.h>
#include <bits/stdc++.h>
#include<cmath>
#include <stdlib.h>

using namespace std;

//A struct containing the index highStart, which indicates the index where the values greater
//than the median begin, and the counter medianCount, which counts how many values are equal to the median
struct indices{
    int highStart;
    int medianCount;
};

//Prints the input vector v
template <typename T>
void printVector(vector<T> v)
{
    for (int i = 0; i < v.size(); i++)
    {
        cout << v[i] << " ";
    }
    cout << "\n\n";
}

//Swaps elements with idx a and b, from a vector v
//@args:
//v-> vector holding double-type values, to swap in
//a-> index of first value
//b-> index of second value
void swap(vector<double>& v , int a, int b)
{
    iter_swap(v.begin() + a, v.begin() + b);
}


//Partition vector v, with from idx l to idx r.
//After this function is done, the element in pos r, will be in the sorted place,
//within vector v.
//@args:
//v-> vector of double-type elements
//l-> the left index from which the function partitions
//r-> the right index from which the function partitions
int partition(vector<double>& v, int l, int r)
{
    double x = v[r];
    int i = l;

    for(int j = l; j <= r - 1; j++)
    {
        if(v[j] <= x)
        {
            swap(v, i, j);
            i++;
        }
    }
    swap(v, i, r);
    return i;
}

//Randomly partition vector v, with from idx l to idx r.
//After this function is done, the element in the randomly chosen position {pivot}, will be in the sorted place,
//within vector v.
//@args:
//v-> vector of double-type elements
//l-> the left index from which the function partitions
//r-> the right index from which the function partitions
int randomPartition(vector<double>& v, int l, int r)
{
    int n = r - l + 1;
    int pivot = rand() % n;
    swap(v, l + pivot, r);
    return partition(v, l, r);
}

//A utility function used to find the median of a vector v.
//Algorithm taken from: https://www.geeksforgeeks.org/median-of-an-unsorted-array-in-liner-time-on/
//@args:
//v-> vector of double-type elements, from which the median is found
//l-> the left index of the part of the vector currently searching in
//r-> the right index of the part of the vector currently searching in
//a-> value which will end up holding on of the two center values of v
//b-> value which will end up holding on of the two center values of v
//(a, b will hold the **two** center values, as v will always have an even number of elements)
void medianUtil(vector<double>& v, int l, int r, int k, double& a, double& b)
{

    if (l <= r) {
 
        // Find the partition index
        int partitionIndex
            = randomPartition(v, l, r);
 
        // If partition index = k, then
        // we found the median of odd
        // number element in v
        if (partitionIndex == k) {
            b = v[partitionIndex];
            if (a != -1)
                return;
        }
 
        // If index = k - 1, then we get
        // a & b as middle element of
        // v
        else if (partitionIndex == k - 1) {
            a = v[partitionIndex];
            if (b != -1)
                return;
        }
 
        // If partitionIndex >= k then
        // find the index in first half
        // of the v
        if (partitionIndex >= k)
            return medianUtil(v, l,
                              partitionIndex - 1,
                              k, a, b);
 
        // If partitionIndex <= k then
        // find the index in second half
        // of the v
        else
            return medianUtil(v,
                              partitionIndex + 1,
                              r, k, a, b);
    }
 
    return;
}

//Function to find median of vector v
//Algorithm taken from: https://www.geeksforgeeks.org/median-of-an-unsorted-array-in-liner-time-on/
//@args:
//v-> vector holding double-type elements, from which to find median
double findMedian(vector<double>& v)
{
    int n = v.size();
    double ans, a = -1, b = -1;
 
    // If n is odd
    if (n % 2 == 1) {
        medianUtil(v, 0, n - 1,
                   n / 2, a, b);
        ans = b;
    }
 
    // If n is even
    else {
        medianUtil(v, 0, n - 1,
                   n / 2, a, b);
        ans = (a + b) / 2;
    }
 
    return ans;
}

//Function which finds the kth smallest element in vector v.
//Can be used to find the left center element of vector v and use that as the median.
//The algorithm below works when using this function to determine the "median"
//(even though because we have an even amount of points, it's not the real median)
//Kept here for the record.
double kthSmallest(vector<double>& v, int l, int r, int k)
{
    if(k > 0 && k <= r - l + 1)
    {
        int idx = partition(v, l, r);

        if(idx - l == k - 1)
            return v[idx];
        
        if(idx - l > k - 1)
            return kthSmallest(v, l, idx - 1, k);
        
        return kthSmallest(v, idx + 1, r, k - idx + l - 1);
    }
    
    return INT_MAX;
}

//Partitions the vector dist and data, so that the left part of dist has distances less than the median,
//the middle part equal to the median, and the right part greater than the median. And so the vector data
//will have on the left points with distance less than the median, middle equal to the median, right greater
//than the median.
//@args:
//dist-> the vector containing the distances of each point
//data-> the vector containing the points (column-major)
//median-> the value which the partitioning is based on
//Algorithm modified from: https://www.geeksforgeeks.org/three-way-partitioning-of-an-array-around-a-given-range/
indices threeWayPartition(vector<double>& dist, vector<double>& data, int dim, double median)
{
    int n = dist.size();
    // Initialize ext available positions for
    // smaller (than range) and greater lements
    int start = 0, end = n-1;
 
    // Traverse elements from left
    for (int i=0; i<=end;)
    {
        // If current element is smaller than
        // range, put it on next available smaller
        // position.
        if (dist[i] < median)
        {
          //if i and start are same in that case we can't swap
          //swap only if i is greater than start
          if(i==start)
          {
            start++;
            i++;
          }
          else
          {
            swap(dist, i, start);
            swap_ranges(data.begin() + start*dim, data.begin() + (start+1)*dim, data.begin() + i*dim);
            i++;
            start++;
          }
        }
 
        // If current element is greater than
        // range, put it on next available greater
        // position.
        else if (dist[i] > median)
        {
            swap(dist, i, end);
            swap_ranges(data.begin() + i*dim, data.begin() + (i+1)*dim, data.begin() + end*dim);
            end--;
        }
        else
            i++;
    }

    //The struct containing the info to be returned
    indices info;
    //If the start index is less or equal to the end index,
    //values equal to the median have been found.
    //Their count is the # of elements between the 2 indices.
    //Else indicate that no elements equal to the median have been found,
    //by setting the value to -1.
    if(start <= end)
        info.medianCount = (end-start+1);
    else 
        info.medianCount = -1;

    //The beginning index of points greater than the median is always the idx end+1
    info.highStart = (end+1);

    return info;
}


//Function which returns the eucledian distance between two points a and b.
//The two elements must have same dimensions.
//@args:
//a-> first point, in the form of a vector, holding double-type elements
//b-> second point, in the form of a vector, holding double-type elements
double euclDistance(vector<double> a, vector<double> b)
{
    //Get the dimension size
    int n = a.size();

    //Initialize the sum of square differences
    double sqrSum = 0;

    //Sum over the square differences
    for (int i = 0; i < n; i++)
    {
        sqrSum += pow(a[i] - b[i], 2);
    }

    //Return the distance
    return  sqrt(sqrSum);
}

//A recursive function, which by the end, will have processes hold points with ascendingly biggest distances.
//Points will NOT be sorted within the process.
//In each world, process with rank == 0 is the main rank.
//@args:
//curr_world-> The current MPI Comm world
//data-> The vector which holds the points of the process
//dist-> The vector which holds the distances of the points
//dim-> The dimensions of the points
//p_per_p-> The number of points per process
void distributeByMedian(MPI_Comm curr_world, vector<double>& data, vector<double>& dist, int dim, int p_per_p)
{
    // Get the number of processes in curr_world
    int world_size;
    MPI_Comm_size(curr_world, &world_size);

    //If the world size is equal to 1, return, the recursion is finished, from this side.
    if(world_size == 1)
    {
        return;
    }

    // Get the rank of this process in curr_world
    int rank;
    MPI_Comm_rank(curr_world, &rank);

    //Vector which will hold all the distances in the current world.
    //Only to be used by the main process of the world.
    vector<double> all_dist;
    if (rank == 0)
    {
        //Initialize the vector 
        all_dist = vector<double>(p_per_p*world_size);
    }
    //Gather all the distances from the current world
    MPI_Gather(&dist[0], dist.size(), MPI_DOUBLE, &all_dist[0], dist.size(), MPI_DOUBLE, 0, curr_world);

    //The median of the distances, in the current world.
    double median;
    if(rank == 0)
    {
        median = findMedian(all_dist);
        //cout << "Median " << median << "\n";
        //median = kthSmallest(all_dist, 0, all_dist.size() - 1, all_dist.size() / 2);
    }
    //Broadcast the median to all processes of the current world.
    MPI_Bcast(&median, 1, MPI_DOUBLE, 0, curr_world);

    //Partition the distance vector of each process ONE time, so all points with distance equal or less
    //than the median will be on the left, and with distance larger than the median, on the right
    //int highStart = /quickSortPartition(dist, data, dim, 0, dist.size()-1, median);
    indices idx = threeWayPartition(dist, data, dim, median);
    int medianCount = idx.medianCount;
    int highStart = idx.highStart;

    //Variable that holds how many elements will be exchanged from the current process.
    int elementsToExchange = 0;

    //If the process is from the left half, it should hold points with distances <=median,
    //so it will exchange all points >median. 
    //IMPORTANT: It must be checked that the point at highStart is a point with distance greater than the median.
    //If it is not, it means that all the points in the process have distances <=median.
    if(rank < (world_size/2))
    {

        if(dist[highStart] > median)
            elementsToExchange = dist.size() - highStart;
    }
    //Do the same for processes in the right half.
    else
    {   
        if(dist[0] <= median)
            elementsToExchange = highStart;
    }

    //A vector which holds the amount of points to be exchanged in each process.
    //Each process has this vector.
    vector<int> elementVector(world_size);
    MPI_Allgather(&elementsToExchange, 1, MPI_INT, &elementVector[0], 1, MPI_INT, curr_world);

    vector<int> highStartVector(world_size);
    MPI_Gather(&highStart, 1, MPI_INT, &highStartVector[0], 1, MPI_INT, 0, curr_world);

    vector<int> medianCountVector(world_size);
    MPI_Gather(&medianCount, 1, MPI_INT, &medianCountVector[0], 1, MPI_INT, 0, curr_world);

    //The exchange vector holds information on which process should exchange how many elements with which vector.
    //It is structured as follows:
    //Eg. If there are 8 processes, its size will be 16.
    //The first 4 elements, will hold the number of exchanges to be made between rank 0 and ranks {4,5,6,7}
    //The next 4 elements for rank 1 with ranks {4,5,6,7}
    //and so forth.

    //Starting with with rank 0 and continuing with all ranks of the left side,
    //start by exchanging as many elements as possible with the first rank of the right side.
    //If the left side rank currently investigated has more elements, go to the next rank of the right side
    //and so forth, until all elements of the left side ranks have been exchanged.
    vector<int> exchangeVector(pow(world_size/2,2));
    if(rank == 0)
    {   
        for(int i = 0; i < world_size/2; i++)
        {
            int j = world_size/2;
            //int elems = elementVector[i];
            while(elementVector[i] > 0)
            {
                int subQuantity = 0;

                if(elementVector[i] >= elementVector[j])
                {
                    subQuantity = elementVector[j];
                }
                else
                {
                    subQuantity = elementVector[i];
                }
                exchangeVector[(world_size/2)*(i-1) + j] = subQuantity;
                elementVector[i] -= subQuantity;
                elementVector[j] -= subQuantity;

                j++;
                if (j == world_size)
                    break;
            }
        }
        //After the above loop is done, all points on the left side should have
        //all their points be exchanged.

        //Now, for all processes on the right side, find if any have leftover points
        //that need to be exchanged.
        //If so, find left side processes that have points with distances equal to the median
        for(int i = world_size/2; i < world_size; i++)
        {
            int j = 0;
            while(elementVector[i] > 0)
            {
                int subQuantity = 0;
                
                //Check if the left-side process has points with dist=median
                //If it hasn't, check the next process
                if(medianCountVector[j] < 0)
                {
                    j++;
                    continue;
                }

                //Exchange the most points possible between the left-side process points with dist=median,
                //and the right-side points that need to be traded
                if(elementVector[i] >= medianCountVector[j])
                {
                    subQuantity = medianCountVector[j];
                }
                else
                {
                    subQuantity = elementVector[i];
                }
                
                //Update the available points with dist=median of the left-side process
                medianCountVector[j] -= subQuantity;

                //Update the elements that need to be traded in the right-side process
                elementVector[i] -= subQuantity;
                
                //Update the exchange vector in the element holding the trades between
                //the right-side process i and the left-side process j
                exchangeVector[j*(world_size/2) + (i - world_size/2)] += subQuantity;

                //Update the starting point of exchanges for the left-side process,
                //by going subQuantity points to the left
                highStartVector[j] -= subQuantity;
                
                j++;
                if(j == world_size/2)
                    break;
            }
        }
    }

    //Broadcast the exchange vector to all processes
    MPI_Bcast(&exchangeVector[0], exchangeVector.size(), MPI_INT, 0, curr_world);

    //Broadcast the updated start points of higher than median points to all processes
    MPI_Bcast(&highStartVector[0], highStartVector.size(), MPI_INT, 0, curr_world);
    
    //And update it in each process
    highStart = highStartVector[rank];

    //If the rank is on the left side, exchange all points with distances >median
    if(rank < (world_size/2))
    {
        //After each exchange with the right side, a new batch of points is exchanged/
        //Each batch has continuous memory placement.
        //e.g. if rank 0 exchanges 4 points with rank 4, the next batch will begin from point 5.
        int offset = 0;
        for(int r = 0; r < world_size/2; r++)
        {
            //Determine the rank to exchange points with (one of the right side ranks)
            int share_rank = (world_size/2) + r;

            //Number of points to be exchanged is already determined by the exchangeVector
            int points_exchanged = exchangeVector[rank*(world_size/2) + r];

            //Exchange these points, along with their distances
            MPI_Sendrecv_replace(&dist[highStart+offset], points_exchanged, MPI_DOUBLE, share_rank, 0, share_rank, 0, curr_world, MPI_STATUS_IGNORE);
            MPI_Sendrecv_replace(&data[(highStart+offset)*dim], points_exchanged*dim, MPI_DOUBLE, share_rank, 1, share_rank, 1, curr_world, MPI_STATUS_IGNORE);

            //Update the offset
            offset += points_exchanged;
        }
    }
    //Else, if the rank is on the right side, exchange all points with distances <=median
    else
    {
        int offset = 0;
        for(int r = 0; r < world_size/2; r++)
        {
            //Determine the rank to exchange points with (one of the left side ranks)
            int share_rank = r;

            //Number of points to be exchanged is already determined by the exchangeVector
            int points_exchanged = exchangeVector[r*(world_size/2) + (rank - world_size/2)];

            //Exchange these points, along with their distances
            MPI_Sendrecv_replace(&dist[offset], points_exchanged, MPI_DOUBLE, share_rank, 0, share_rank, 0, curr_world, MPI_STATUS_IGNORE);
            MPI_Sendrecv_replace(&data[offset*dim], points_exchanged*dim, MPI_DOUBLE, share_rank, 1, share_rank, 1, curr_world, MPI_STATUS_IGNORE);

            //Update the offset
            offset += points_exchanged;

        }

    }
   
    //After the trades are done, the function has to be recursively called.

    //Split the current world into to 2, with the left part of the processes
    //becoming one world, and the right part, another world.

    //Determine the color of the current process (0 for left side processes, 1 for right side)
    int color = rank / (world_size/2);
    
    //Split the current world
    MPI_Comm new_world;
    MPI_Comm_split(curr_world, color, color, &new_world);

    //Recusrive call to the function
    distributeByMedian(new_world, data, dist, dim, p_per_p);
}

//The driver code for this project
int main(int argc, char** argv) {


    // Initialize MPI
    MPI_Init(&argc, &argv);

    // Get the number of processes in MPI_COMM_WORLD
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of this process in MPI_COMM_WORLD
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    //Start and end time of the distribute by median function
    double start, end;
    
    //Since we are emulating a system in which each node already has N/p points stored,
    //each process reads the appropriate part of the bin dataset.
    //If scatter was used, then the main process would have to read ALL the points, which would defeat the 
    //purpose of this project.

    //Open the bin file

    //ifstream f("data/CIFAR10.bin", ios::binary);
    ifstream f("data/test.bin", ios::binary);
    //ifstream f("data/fashionMnist.bin", ios::binary);
    //ifstream f("data/mnist.bin", ios::binary);

    //Read the dimension of the points, and the number of points
    vector<int64_t> dims(2);
    f.read((char*)&dims[0], sizeof(int64_t) * 2);
    int64_t dim = dims[0];
    int64_t p = dims[1];

    //If the number of points cannot be divided equally by the number of processes, throw an error and exit
    if((p % world_size) != 0)
    {
        if (rank == 0)
            cout << "Number of points is " << p << ".\nPlease enter a correct # of processes.";
        MPI_Finalize();
        exit(-1);
    }

    if(rank == 0)
        cout << "\nDistributing data...\n\n";

    //Calculate how many points each process holds
    int64_t p_per_p = p / world_size;
    
    //Read the appropriate chunk of data from the bin file
    vector<double> data(dim * p_per_p);

    f.ignore(rank * dim * p_per_p * sizeof(double));
    f.read((char*)&data[0], dim * p_per_p * sizeof(double));

    //Wait until all processes have read their chunk
    MPI_Barrier(MPI_COMM_WORLD);
    if( rank == 0)
        cout << p <<" points distributed to " << world_size << " processes!\nPoint dimensions: " << dim << "\n\n";

    //Start the timer
    start = MPI_Wtime();

    //Buffer to hold the pivot point
    vector<double> pivot(dim);

    //If the main process
    if (rank == 0)
    {
       //Choose a random pivot point index
        srand(time(NULL));
        int pivot_idx = rand() % p_per_p;

        //Get the point in that index
        pivot = vector<double>(data.cbegin() + dim*pivot_idx, data.cbegin() + dim * (pivot_idx+1));

    }
    //Broadcast the pivot to all processes
    MPI_Bcast(&pivot[0], pivot.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    

    //Calcucate the distance from the pivot for each point
    vector<double> dist(p_per_p);

    
    for (int i = 0; i < p_per_p; i++)
    {
        vector<double> point(data.begin() + i * dim, data.begin() + (i + 1) * dim);
        dist[i] = euclDistance(pivot, point);
    } 

    
    distributeByMedian(MPI_COMM_WORLD, data, dist, dim, p_per_p);
  
    //Wait until all processes are done with the recursion
    MPI_Barrier(MPI_COMM_WORLD);

    //Stop the timer
    end = MPI_Wtime();

    //Print the runtime of the program
    if (rank == 0)
    {   
        printf("Runtime = %f s\n\n", end-start);
    }
    
    //Check if the result is correct

    //Variables which hold the maximum and minimum distance of the points in the process
    double minD;
    double maxD;

    auto biggest = std::max_element(std::begin(dist), std::end(dist));
    auto smallest = std::min_element(std::begin(dist), std::end(dist));
    minD = *smallest;
    maxD = *biggest;
    

    //Vectors which hold bthe max and min values of all processes
    vector<double> maxVals;
    vector<double> minVals;

    //Gather the min and max values to the main process
    if(rank == 0)
    {
        maxVals.resize(world_size);
        minVals.resize(world_size);
    }
    MPI_Gather(&minD, 1, MPI_DOUBLE, &minVals[0], 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(&maxD, 1, MPI_DOUBLE, &maxVals[0], 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    //The main process determines whether the result is correct.
    //If the maximum distance in a process {i} is greater than the minimum distance in the process {i+1},
    //the algorithm has failed.
    if(rank == 0)
    {   
        int failedTest = 0;
        for(int i = 0; i < maxVals.size()-1; i++)
        {
            if(maxVals[i] > minVals[i+1])
            {
                failedTest = 1;
                cout << "Rank: " << i << ", value: " << maxVals[i] << "\n";
                cout << "Rank: " << (i+1) << ", value: " << maxVals[i+1] << "\n";
                break;
            }
        }

        if(failedTest)
        {
            cout << "\033[1;31mCheck failed!\033[0m\n\n";
        }
        else
        {
            cout << "\033[1;32mCheck passed!\033[0m\n\n";
        }
    }
    
    // Finalize MPI
    MPI_Finalize();

    return 0;
}
