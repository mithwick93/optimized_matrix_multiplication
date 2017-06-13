# Optimized Matrix Multiplication
  This repository contains c++ programs to calculate time taken to find matrix multiplication of two n x n matrixes. The parallel version utilizes the openMP library to parallelize the matrix multiplication.

## Instructions
### Compile
  Compile each file separately using following command
    
    g++ <file_name>.cpp -fopenmp
    
### Run the program
  The program requires the sample size( number of times to run to get average time) and the matrix size n (between 200 and 2000)
  
    <program_name>.exe <sample_size> <matrix_size>
  
