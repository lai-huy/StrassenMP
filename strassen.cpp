#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

using std::atoi;

#define DEBUG false

/**
 * @brief Print a matrix
 *
 * @param n The size of the matrix
 * @param matrix The matrix to be printed
 */
void print_matrix(size_t n, int** const& matrix) {
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j)
            printf("%d\t", matrix[i][j]);
        printf("\n");
    }
}

/**
 * @brief Allocate memory for a matrix
 *
 * @param n The size of the matrix
 * @return int** The allocated matrix
 */
int** allocate_matrix(size_t n) {
    int** matrix = new int* [n] {};
    for (size_t i = 0; i < n; ++i)
        matrix[i] = new int[n] {};
    return matrix;
}

/**
 * @brief Deallocate memory for a matrix
 *
 * @param n The size of the matrix
 * @param matrix The matrix to be deallocated
 */
void deallocate_matrix(size_t n, int** matrix) {
    for (size_t i = 0; i < n; ++i)
        delete[] matrix[i];
    delete[] matrix;
}

/**
 * @brief Generate a random matrix
 *
 * @param n The size of the matrix
 * @param matrix The matrix to be generated
 */
void generate_matrix(size_t n, int** matrix) {
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < n; j++)
            matrix[i][j] = rand() % 10;
}

/**
 * @brief Add two matrices. C = A + B
 *
 * @param n The size of the matrices
 * @param A The first matrix
 * @param B The second matrix
 *
 * @return int** The result matrix
 */
int** add_matrix(size_t n, int** A, int** B) {
    int** result = allocate_matrix(n);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
            result[i][j] = A[i][j] + B[i][j];

    return result;
}

/**
 * @brief Subtract two matrices. C = A - B
 *
 * @param n The size of the matrices
 * @param A The first matrix
 * @param B The second matrix
 *
 * @return int** The result matrix
 */
int** sub_matrix(size_t n, int** A, int** B) {
    int** result = allocate_matrix(n);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
            result[i][j] = A[i][j] - B[i][j];

    return result;
}

/**
 * @brief Extract a quadrant of the matrix
 *
 * @param n size of the matrix
 * @param matrix The matrix to be extracted
 * @param row row offset
 * @param col column offset
 * @return int** The extracted matrix
 */
int** seperate_matrix(size_t n, int** matrix, size_t row, size_t col) {
    size_t m = n >> 1;
    int** slice = allocate_matrix(m);
    for (size_t i = 0; i < m; ++i)
        for (size_t j = 0; j < m; ++j)
            slice[i][j] = matrix[i + row][j + col];

    return slice;
}

/**
 * @brief Combines 4 matrices into a single matrix
 *
 * @param m The size of the matrices
 * @param C11 The top-left matrix
 * @param C12 The top-right matrix
 * @param C21 The bottom-left matrix
 * @param C22 The bottom-right matrix
 *
 * @return int** The combined matrix
 */
int** combine_matrix(size_t m, int** C11, int** C12, int** C21, int** C22) {
    int n = m << 1;
    int** C = allocate_matrix(n);
    for (size_t i = 0; i < m; ++i)
        for (size_t j = 0; j < m; ++j) {
            C[i][j] = C11[i][j];
            C[i][j + m] = C12[i][j];
            C[i + m][j] = C21[i][j];
            C[i + m][j + m] = C22[i][j];
        }

    return C;
}

/**
 * @brief Multiplies two matrixes together with the traditional method.
 *
 * @param n The size of the matrices
 * @param A The first matrix
 * @param B The second matrix
 *
 * @return int** The result matrix
 */
int** naive(size_t n, int** A, int** B) {
    int** prod = allocate_matrix(n);

    size_t i{}, j{};

#pragma omp parallel for collapse(2)
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            prod[i][j] = 0;
            for (size_t k = 0; k < n; k++)
                prod[i][j] += A[i][k] * B[k][j];
        }
    }

    return prod;
}

/**
 * @brief Multiply two matrices using the Strassen algorithm. C = A * B
 *
 * @param n The size of the matrices
 * @param threshold The threshold value
 * @param A The first matrix
 * @param B The second matrix
 *
 * @return int** The result matrix
 */
int** strassen(size_t n, size_t threshold, int** A, int** B) {
    if (n <= threshold)
        return naive(n, A, B);

    size_t m = n >> 1;

    // Split matrices into 4 submatrices
    int** A11 = seperate_matrix(n, A, 0, 0);
    int** A12 = seperate_matrix(n, A, 0, m);
    int** A21 = seperate_matrix(n, A, m, 0);
    int** A22 = seperate_matrix(n, A, m, m);
    int** B11 = seperate_matrix(n, B, 0, 0);
    int** B12 = seperate_matrix(n, B, 0, m);
    int** B21 = seperate_matrix(n, B, m, 0);
    int** B22 = seperate_matrix(n, B, m, m);

    // M1 = (A11 + A22)(B11 + B22)
    int** M1{};
#pragma omp task shared(M1)
    {
        int** temp1 = add_matrix(m, A11, A22);
        int** temp2 = add_matrix(m, B11, B22);
        M1 = strassen(m, threshold, temp1, temp2);
        deallocate_matrix(m, temp1);
        deallocate_matrix(m, temp2);
    }

    // M2 = (A21 + A22) B11
    int** M2{};
#pragma omp task shared(M2)
    {
        int** temp = add_matrix(m, A21, A22);
        M2 = strassen(m, threshold, temp, B11);
        deallocate_matrix(m, temp);
    }

    // M3 = A11 (B12 - B22)
    int** M3{};
#pragma omp task shared(M3)
    {
        int** temp = sub_matrix(m, B12, B22);
        M3 = strassen(m, threshold, A11, temp);
        deallocate_matrix(m, temp);
    }

    // M4 = A22 (B21 - B11)
    int** M4{};
#pragma omp task shared(M4)
    {
        int** temp = sub_matrix(m, B21, B11);
        M4 = strassen(m, threshold, A22, temp);
        deallocate_matrix(m, temp);
    }

    // M5 = (A11 + A12) B22
    int** M5{};
#pragma omp task shared(M5)
    {
        int** temp = add_matrix(m, A11, A12);
        M5 = strassen(m, threshold, temp, B22);
        deallocate_matrix(m, temp);
    }

    // M6 = (A21 - A11) (B11 + B12)
    int** M6{};
#pragma omp task shared(M6)
    {
        int** temp1 = sub_matrix(m, A21, A11);
        int** temp2 = add_matrix(m, B11, B12);
        M6 = strassen(m, threshold, temp1, temp2);
        deallocate_matrix(m, temp1);
        deallocate_matrix(m, temp2);
    }

    // M7 = (A12 - A22) (B21 + B22)
    int** M7{};
#pragma omp task shared(M7)
    {
        int** temp1 = sub_matrix(m, A12, A22);
        int** temp2 = add_matrix(m, B21, B22);
        M7 = strassen(m, threshold, temp1, temp2);
        deallocate_matrix(m, temp1);
        deallocate_matrix(m, temp2);
    }

#pragma omp taskwait
    deallocate_matrix(m, A11);
    deallocate_matrix(m, A12);
    deallocate_matrix(m, A21);
    deallocate_matrix(m, A22);
    deallocate_matrix(m, B11);
    deallocate_matrix(m, B12);
    deallocate_matrix(m, B21);
    deallocate_matrix(m, B22);

    // C11 = M1 + M4 - M5 + M7
    int** c11{};
#pragma omp task shared(c11)
    {
        int** temp1 = add_matrix(m, M1, M4);
        int** temp2 = sub_matrix(m, M7, M5);
        c11 = add_matrix(m, temp1, temp2);
        deallocate_matrix(m, temp1);
        deallocate_matrix(m, temp2);
    }

    // C12 = M3 + M5
    int** c12{};
#pragma omp task shared(c12)
    {
        c12 = add_matrix(m, M4, M5);
    }

    // C21 = M2 + M4
    int** c21{};
#pragma omp task shared(c21)
    {
        c21 = add_matrix(m, M6, M7);
    }

    // C22 = M1 - M2 + M3 + M6
    int** c22{};
#pragma omp task shared(c22)
    {
        int** temp1 = sub_matrix(m, M1, M2);
        int** temp2 = add_matrix(m, M3, M6);
        c22 = add_matrix(m, temp1, temp2);
        deallocate_matrix(m, temp1);
        deallocate_matrix(m, temp2);
    }

#pragma omp taskwait
    deallocate_matrix(m, M1);
    deallocate_matrix(m, M2);
    deallocate_matrix(m, M3);
    deallocate_matrix(m, M4);
    deallocate_matrix(m, M5);
    deallocate_matrix(m, M6);
    deallocate_matrix(m, M7);

    int** prod = combine_matrix(m, c11, c12, c21, c22);

    deallocate_matrix(m, c11);
    deallocate_matrix(m, c12);
    deallocate_matrix(m, c21);
    deallocate_matrix(m, c22);

    return prod;
}

int main(int argc, char* argv[]) {
    struct timespec start, stop;
    double total_time;

    if (argc != 4) {
        printf("Usage: %s <k> <k'> <num_threads>\n", argv[0]);
        return EXIT_FAILURE;
    }

    int k = atoi(argv[1]);
    int k_prime = atoi(argv[2]);
    int num_threads = atoi(argv[3]);

    size_t n = 1 << k;
    int** A{};
    int** B{};
    int** C{};

    srand(0);

    A = allocate_matrix(n);
    B = allocate_matrix(n);

    generate_matrix(n, A);
    generate_matrix(n, B);
    
    if (DEBUG) {
        print_matrix(n, A);
        printf("-----------------------\n");
        print_matrix(n, B);
        printf("-----------------------\n");
    }

    clock_gettime(CLOCK_REALTIME, &start);

    omp_set_num_threads(num_threads);

#pragma omp parallel
    {
#pragma omp single
        {
            C = strassen(n, k_prime, A, B);
        }
    }

    clock_gettime(CLOCK_REALTIME, &stop);
    total_time = (stop.tv_sec - start.tv_sec) + 0.000000001 * (stop.tv_nsec - start.tv_nsec);

    if (DEBUG)
        print_matrix(n, C);
    printf("Time: %f\n", total_time);

    deallocate_matrix(n, A);
    deallocate_matrix(n, B);
    deallocate_matrix(n, C);

    return EXIT_SUCCESS;
}
