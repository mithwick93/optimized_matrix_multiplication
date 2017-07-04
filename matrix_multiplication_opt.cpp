#include <cstdlib>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>


extern "C"
{
#include <immintrin.h>
}


using namespace std;

static void get_timings(char *msg);

static void free_matrix(double **matrix);

static double run_experiment();

static double **initialize_matrix(bool random);

static double **matrix_multiply_parallel_optimized(double **A, double **B, double **C);

static double **matrix_transpose(double **A);

static int n; // size of matrix 
static int sample_size = 200; // test sample size

/*
 * Matrix multiplication program
 */
int main(int argc, char **argv) {
    printf("Testing for sample size: %d\n\n", sample_size);

    for (int matrix_size = 200; matrix_size <= 2000; matrix_size += 200) {
        n = matrix_size;
        printf("Matrix size : %d\n--------------------\n", matrix_size);
        fflush(stdout);

        // optimised parallel
        get_timings((char *) "Optimised Parallel");

        printf("\n");
        fflush(stdout);
    }

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
 * Run experiment using optimised parallel algorithm
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
    C = matrix_multiply_parallel_optimized(A, B, C);
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
            matrix[row][column] = random ? (double)rand() : 0.0;
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
double **matrix_multiply_parallel_optimized(double **A, double **B, double **C) {
    int row, column, itr;
    double *row_A, *row_D, *ptr, *temp;

    double **D = matrix_transpose(B);

    double sums[8];

    __m256d ymm0, ymm1, ymm2, ymm3, ymm4,
            ymm8, ymm9, ymm10, ymm11, ymm12;
    // declare shared and private variables for OpenMP threads
#pragma omp parallel shared(A, B, C, D) private(row, column, itr, row_A, row_D, ptr, temp, sums, ymm0, ymm1, ymm2, ymm3, ymm4, ymm8, ymm9, ymm10, ymm11, ymm12)
    {
        // Static allocation of data to threads
#pragma omp for schedule(static)
        for (row = 0; row < n; row++) {
            row_A = &A[row][0];
            for (column = 0; column < n; column++) {
                row_D = &D[column][0];
                ptr = &C[row][column];
                for (itr = 0; itr < n; itr += 20) {
                    temp = row_A + itr;
                    ymm0 = _mm256_loadu_pd(temp);
                    ymm1 = _mm256_loadu_pd(temp + 4);
                    ymm2 = _mm256_loadu_pd(temp + 8);
                    ymm3 = _mm256_loadu_pd(temp + 12);
                    ymm4 = _mm256_loadu_pd(temp + 16);

                    temp = row_D + itr;
                    ymm8 = _mm256_loadu_pd(temp);
                    ymm9 = _mm256_loadu_pd(temp + 4);
                    ymm10 = _mm256_loadu_pd(temp + 8);
                    ymm11 = _mm256_loadu_pd(temp + 12);
                    ymm12 = _mm256_loadu_pd(temp + 16);

                    ymm0 = _mm256_mul_pd(ymm0, ymm8);
                    ymm1 = _mm256_mul_pd(ymm1, ymm9);
                    ymm2 = _mm256_mul_pd(ymm2, ymm10);
                    ymm3 = _mm256_mul_pd(ymm3, ymm11);
                    ymm4 = _mm256_mul_pd(ymm4, ymm12);

                    ymm0 = _mm256_add_pd(ymm0, ymm1);
                    ymm0 = _mm256_add_pd(ymm0, ymm2);
                    ymm0 = _mm256_add_pd(ymm0, ymm3);
                    ymm0 = _mm256_add_pd(ymm0, ymm4);

                    _mm256_storeu_pd(sums, ymm0);

                    for (int k = 0; k < 4; k++) {
                        *ptr += sums[k];
                    }
                }
            }
        }
    }
    return C;
}

/**
 * Gets transpose of A
 *
 * @param A
 * @return A^T
 */
double **matrix_transpose(double **A) {
    // allocate memory for n*n matrix
    double **D = initialize_matrix(false);
    int row, column;
    // declare shared and private variables for OpenMP threads
#pragma omp parallel shared(A, D) private(row, column)
    {
        // Static allocation of data to threads
#pragma omp for schedule(static)
        for (row = 0; row < n; row++) {
            for (column = 0; column < n; column++) {
                D[row][column] = A[column][row];
            }
        }
    }
    return D;
}