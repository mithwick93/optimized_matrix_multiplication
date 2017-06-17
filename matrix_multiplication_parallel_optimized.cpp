#include <cstdlib>
#include <stdio.h>
#include <iostream>
#include <time.h>
#include <math.h>

using namespace std;

static void get_timings();

static void free_matrix(double **matrix);

static double get_random_number();

static double run_experiment();

static double **initialize_matrix(bool random);

static double **matrix_multiply_parellel(double **A, double **B, double **C);

static double **matrix_multiply_parellel_optimized(double **A, double **B, double **C);

static double **matrix_multiply_parellel_optimized_block(double **A, double **B, double **C);

static double **matrix_transpose(double **A);

bool matrix_equals(double **A, double **B);

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

//    system("pause");
    return 0;
}

/**
 * Calculate time, standard deviation and sample size
 * @param algo_type algorithm to check
 * @param msg message to display
 */
void get_timings() {
    double total_time = 0.0;
    double execution_times[sample_size];

    // calculate average execution time
    for (int i = 0; i < sample_size; i++) {
        double elapsed_time = run_experiment();
        execution_times[i] = elapsed_time;
        total_time += elapsed_time;
    }

    double average_time = total_time / sample_size;
    printf("Optimized Parallel run time : %.4f seconds\n", average_time);
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
double **matrix_multiply_parellel_optimized_block(double **A, double **B, double **C) {
    int row, column, iter;
    int block_column = 0, block_iter = 0;
    int block_size = 25;
    for (block_iter = 0; block_iter < n; block_iter += block_size) {
        for (block_column = 0; block_column < n; block_column += block_size) {
            // declare shared and private variables for OpenMP threads
#pragma omp parallel shared(A, B, C, block_column, block_iter, block_size) private(row, column, iter)
            {
                // Static allocation of data to threads
#pragma omp for schedule(static)
                for (row = 0; row < n; ++row) {
                    double *row_A = A[row];
                    double *row_C = C[row];
                    for (iter = block_iter; iter < block_iter + block_size; ++iter) {
                        double *row_B = B[iter];
                        double val_A = row_A[iter];
                        for (column = block_column; column < block_column + block_size; ++column) {
//                            C[row][column] += A[row][iter] * B[iter][column];
                            row_C[column] += val_A * row_B[column];
                        }
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
                for (column = 0; column < n; column++) {
                    row_C[column] += val_A * row_B[column];
                }
            }
        }
    }
    return C;
}

/**
 * Parallel multiply matrix A and B
 * @param A matrix A
 * @param B matrix B
 * @param C matrix C
 * @return matrix C = A*B
 */
double **matrix_multiply_parellel(double **A, double **B, double **C) {
    int row, column, itr;
    // declare shared and private variables for OpenMP threads
#pragma omp parallel shared(A, B, C) private(row, column, itr)
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
                D[row][column] += A[column][row];
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