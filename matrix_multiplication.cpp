/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   main.cpp
 * Author: Shehan
 *
 * Created on 11 June 2017, 10:35
 */
#include <cstdlib>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

using namespace std;

double get_random_number();
double** initialize_matrix(bool random);
double** matrix_multiply(double** A, double** B, double** C);
double** matrix_multiply_parellel(double** A, double** B, double** C);

int n; // size of matrix 

/*
 * 
 */
int main(int argc, char** argv) {
    srand(static_cast<unsigned> (time(0)));

    for (int matrix_size = 200; matrix_size <= 2000; matrix_size += 200) {
        double start, finish, elapsed;
        n = matrix_size;
        printf("Matrix size : %d\n", matrix_size);
        fflush(stdout);

        double** A = initialize_matrix(true);
        double** B = initialize_matrix(true);
        double** C = initialize_matrix(false);
        double** D = initialize_matrix(false);

        // serial execution
        start = clock();
        C = matrix_multiply(A, B, C);
        finish = clock();

        elapsed = (finish - start) / CLOCKS_PER_SEC;
        printf("Normal multiplication time : %f s", elapsed);
        fflush(stdout);
        
        printf("\n");
        fflush(stdout);

        // parallel execution
        start = clock();
        D = matrix_multiply_parellel(A, B, D);
        finish = clock();

        elapsed = (finish - start) / CLOCKS_PER_SEC;
        printf("OpenMP multiplication time : %f s", elapsed);
        fflush(stdout);

        printf("\n\n");
        fflush(stdout);
    }

    return 0;
}

/**
 * Generate random floating point number
 * @return random float number
 */
double get_random_number() {
    return static_cast<float> (rand()) / (static_cast<float> (RAND_MAX / 10000.0));
}

/**
 * Initialise matrix 
 * @param random fill elements randomly
 * @return initialised matrix
 */
double** initialize_matrix(bool random) {
    // allocate memory for n*n matrix
    double** matrix;
    matrix = (double**) malloc(n * sizeof (double*));
    for (int i = 0; i < n; i++)
        matrix[i] = (double*) malloc(n * sizeof (double));

    // initialise matrix elements 
    for (int row = 0; row < n; row++) {
        for (int column = 0; column < n; column++) {
            matrix[row][column] = random ? get_random_number() : 0.0;
        }
    }

    return matrix;
}

/**
 * Multiply matrix A and B 
 * @param A matrix A
 * @param B matrix B
 * @param C matrix C
 * @return matrix C = A*B
 */
double** matrix_multiply(double** A, double** B, double** C) {
    for (int row = 0; row < n; row++) {
        for (int column = 0; column < n; column++) {
            for (int itr = 0; itr < n; itr++) {
                C[row][column] += A[row][itr] * B[itr][column];
            }
        }
    }

    return C;
}

double** matrix_multiply_parellel(double** A, double** B, double** C) {
    int row, column, itr;
    // declare shared and private variables for OpenMP threads
#pragma omp parallel shared(A,B,C) private(row,column,itr) 
    {
        // Static allocation of data to threads
#pragma omp for schedule(static)
        for (row = 0; row < n; row++) {
            for (column = 0; column < n; column++) {
                for (itr = 0; itr < n; itr++) {
                    C[row][column] += A[row][itr] * B[itr][column];
                }
            }
        }
    }

    return C;
}

