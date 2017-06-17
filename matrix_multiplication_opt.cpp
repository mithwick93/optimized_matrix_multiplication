/* 
 * File:   matrix_multiplication.cpp
 * Authors: Shehan, Yasas
 *
 * Created on 11 June 2017, 10:35
 */
#include <cstdlib>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <time.h>
#include <math.h>

using namespace std;

static void get_timings(char *msg);

static void free_matrix(double **matrix);

static double run_experiment();

static double get_random_number();

static double **initialize_matrix(bool random);

static double **matrix_multiply_parellel_optimized(double **A, double **B, double **C);

static int n; // size of matrix 
static int sample_size = 25; // test sample size

/*
 * Matrix multiplication program
 */
int main(int argc, char **argv) {
    printf("Testing for sample size: %d\n\n", sample_size);

    for (int matrix_size = 200; matrix_size <= 2000; matrix_size += 200) {
        n = matrix_size;
        printf("Matrix size : %d\n--------------------\n", matrix_size);
        fflush(stdout);

        // Optimized Parallel
        get_timings((char *) "Optimized Parallel");

        printf("\n");
        fflush(stdout);
    }
//    system("pause");
    return 0;
}

/**
 * Calculate time, standard deviation and sample size
 * @param msg message to display
 */
void get_timings(char *msg) {
    double total_time = 0.0;
    double execution_times[sample_size];

    // calculate average execution time
    for (int i = 0; i < sample_size; i++) {
        double elapsed_time = run_experiment();
        execution_times[i] = elapsed_time;
        total_time += elapsed_time;
    }

    double average_time = total_time / sample_size;
    printf("%s time : %.4f seconds\n", msg, average_time);
    fflush(stdout);

    if (sample_size > 1) {
        double variance = 0.0;

        // calculate standard deviation
        for (int i = 0; i < sample_size; i++) {
            variance += pow(execution_times[i] - average_time, 2);
        }

        double standard_deviation = sqrt(variance / (sample_size - 1));
        printf("%s deviation = %.4f seconds\n", msg, standard_deviation);
        fflush(stdout);

        // calculate sample size
        double samples =
                pow((100 * 1.96 * standard_deviation) / (5 * average_time), 2);
        printf("Samples required: %.4f\n\n", samples);
        fflush(stdout);
    }
}

/**
 * Run experiment using optimized parellel algo
 * and determine time
 * @return elapsed time
 */
double run_experiment() {
    srand(static_cast<unsigned> (time(0)));
    double start, finish, elapsed;

    // initialise matrixes
    double **A = initialize_matrix(true);
    double **B = initialize_matrix(true);
    double **C = initialize_matrix(false);

    start = clock();
    C = matrix_multiply_parellel_optimized(A, B, C);
    finish = clock();

    // calculate elapsed time
    elapsed = (finish - start) / CLOCKS_PER_SEC;

    // free matrix memory
    free_matrix(A);
    free_matrix(B);
    free_matrix(C);

    return elapsed;
}

/**
 * Clear matrix memory
 * @param matrix matrix to free
 */
void free_matrix(double **matrix) {
    for (int i = 0; i < n; i++) {
        free(matrix[i]);
    }
    free(matrix);
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
double **initialize_matrix(bool random) {
    // allocate memory for n*n matrix
    double **matrix;
    matrix = (double **) malloc(n * sizeof(double *));
    for (int i = 0; i < n; i++)
        matrix[i] = (double *) malloc(n * sizeof(double));

    // initialise matrix elements 
    for (int row = 0; row < n; row++) {
        for (int column = 0; column < n; column++) {
            matrix[row][column] = random ? get_random_number() : 0.0;
        }
    }

    return matrix;
}

/**
 * Optimized parallel multiply matrix A and B
 * @param A matrix A
 * @param B matrix B
 * @param C matrix C
 * @return matrix C = A*B
 */
double **matrix_multiply_parellel_optimized(double **A, double **B, double **C) {
    int row, column, itr;
    // declare shared and private variables for OpenMP threads
#pragma omp parallel shared(A, B, C) private(row, column, itr)
    {
        // Static allocation of data to threads
#pragma omp for schedule(static)
        for (row = 0; row < n; row++) {
            double *row_A = A[row];
            double *row_C = C[row];
            for (itr = 0; itr < n; itr++) {
                double *itr_B = B[itr];
                double itr_row_A = row_A[itr];
                for (column = 0; column < n; column++) {
                    row_C[column] += itr_row_A * itr_B[column];
                }
            }
        }
    }
    return C;
}
