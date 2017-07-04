#include <cstdlib>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <iostream>

extern "C"
{
#include <immintrin.h>
}

using namespace std;

static void get_timings();

static void free_matrix(double **matrix);

static double get_random_number();

static double run_experiment();

static double **initialize_matrix(bool random);

static double **matrix_multiply_parellel_optimized(double **A, double **B, double **C);

static double **matrix_multiply_parellel_inst(double **A, double **B, double **C);

static double **matrix_multiply_parellel_inst2(double **A, double **B, double **C);

static double **matrix_transpose(double **A);

static bool matrix_equals(double **A, double **B);

static int n; // size of matrix
static int sample_size; // test sample size

/**
 * Program usage instructions
 * @param program_name
 */
static void program_help(char *program_name) {
    fprintf(stderr, "usage: %s <matrix_size> <sample_size>\n", program_name);
    exit(0);
}

/**
 * Initialise program variables using arguments received
 * @param argc
 * @param argv
 */
static void initialize(int argc, char *argv[]) {
    if (argc != 3) {
        program_help(argv[0]);
    }

    sscanf(argv[1], "%d", &n);
    sscanf(argv[2], "%d", &sample_size);

    if (sample_size <= 0 || n <= 0 || n > 2000) {
        program_help(argv[0]);
    }
}

/*
 * Matrix multiplication parallel program
 */
int main(int argc, char *argv[]) {
    initialize(argc, argv);
    printf(
            "Matrix size : %d | Sample size : %d\n----------------------------------------\n",
            n, sample_size
    );

    // parallel
    get_timings();
    printf("\n");

    return 0;
}

/**
 * Calculate time, standard deviation and sample size
 * @param algo_type algorithm to check
 * @param msg message to display
 */
void get_timings() {
    double total_time = 0.0;

    // calculate average execution time
    for (int i = 0; i < sample_size; i++) {
        total_time += run_experiment();
    }

    double average_time = total_time / sample_size;
    printf("Optimised Parallel run time : %.4f seconds\n", average_time);
}

/**
 * Run experiment using specified algorithm
 * @param algo_type type of algorithm to use
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

    // Validate the calculation
//    double **D = initialize_matrix(false);
//    D = matrix_multiply_parellel_optimized(A, B, D);
//
//    if (!matrix_equals(C, D)) {
//        cout << "Incorrect matrix multiplication!" << endl;
//    }
//
//    free_matrix(D);
    // Validation finalized

    // calculate elapsed time
    elapsed = (finish - start) / CLOCKS_PER_SEC;

//    cout << elapsed << endl;

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
double **matrix_multiply_parellel_inst2(double **A, double **B, double **C) {
    int row, column, itr, k;
    double *row_A, *row_C, *row_B;
    double val_A, arr_A[8];

    __m256d reg1, reg2, reg3;
    // declare shared and private variables for OpenMP threads
#pragma omp parallel shared(A, B, C) private(row, column, itr, row_A, row_C, row_B, val_A, arr_A, reg1, reg2, reg3, k)
    {
        // Static allocation of data to threads
#pragma omp for schedule(static)
        for (row = 0; row < n; row++) {
            row_A = A[row];
            row_C = C[row];
            for (itr = 0; itr < n; itr++) {
                row_B = B[itr];
                val_A = row_A[itr];
                for (k = 0; k < 4; k++)
                    arr_A[k] = val_A;

                reg1 = _mm256_loadu_pd(arr_A);
                // For each column of the selected row above
                //     Add the product of the values of corresponding row element of A
                //     with corresponding column element of B to corresponding row, column of C
                for (column = 0; column < n; column += 4) {
                    reg3 = _mm256_loadu_pd(&row_C[column]);
                    reg2 = _mm256_loadu_pd(&row_B[column]);
                    reg2 = _mm256_mul_pd(reg1, reg2);
                    reg3 = _mm256_add_pd(reg2, reg3);
                    _mm256_storeu_pd(&row_C[column], reg3);
                }
            }
        }
    }
    return C;
}

/**
 * Optimized parallel multiply matrix A and B
 *
 * @param A matrix A
 * @param B matrix B
 * @param C matrix C
 * @return matrix C = A*B
 */
double **matrix_multiply_parellel_inst(double **A, double **B, double **C) {
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
 * Optimized parallel multiply matrix A and B
 * @param A matrix A
 * @param B matrix B
 * @param C matrix C
 * @return matrix C = A*B
 */
double **matrix_multiply_parellel_optimized(double **A, double **B, double **C) {
    int row, column, itr;
    double *row_A, *row_C, *row_B;
    double val_A;
    // declare shared and private variables for OpenMP threads
#pragma omp parallel shared(A, B, C) private(row, column, itr, row_A, row_C, row_B, val_A)
    {
        // Static allocation of data to threads
#pragma omp for schedule(static)
        for (row = 0; row < n; row++) {
            row_A = A[row];
            row_C = C[row];
            for (itr = 0; itr < n; itr++) {
                row_B = B[itr];
                val_A = row_A[itr];
                // For each column of the selected row above
                //     Add the product of the values of corresponding row element of A
                //     with corresponding column element of B to corresponding row, column of C
                for (column = 0; column < n; column += 5) {
                    // Loop unrolling
                    row_C[column] += val_A * row_B[column];
                    row_C[column + 1] += val_A * row_B[column + 1];
                    row_C[column + 2] += val_A * row_B[column + 2];
                    row_C[column + 3] += val_A * row_B[column + 3];
                    row_C[column + 4] += val_A * row_B[column + 4];
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

/**
 * Checks whether A equals to B
 *
 * @param A
 * @param B
 * @return A==B
 */
bool matrix_equals(double **A, double **B) {
    if (A == B) return true;
//    cout << "Checking matrix elements for equality..." << endl;
    int i, j;
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            double round_a = roundf(A[i][j] * 100) / 100;
            double round_b = roundf(B[i][j] * 100) / 100;
            if (round_a != round_b) {
//                cout << round_a << " != " << round_b << endl;
                return false;
            }
        }
    }
    return true;
}