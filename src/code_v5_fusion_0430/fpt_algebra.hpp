/*
 *	This is light floating point algebra library
 *	Author: Yun Wu
 *	Created by: 2019-06-13
 *	Copyright @ Yun Wu
 *
 */

#ifndef SRC_FPT_ALGEBRA_HPP_
#define SRC_FPT_ALGEBRA_HPP_

#include "data.hpp"
#include "common.hpp"

class Float_Point_Algebra {

public:

	// Generate all zero matrix
	template<class T, int M, int N>
	void ZEROS_MAT(T A[M][N]) {
		for (int i = 0; i < M; i++) {
			for (int j = 0; j < N; j++) {
				A[i][j] = 0;
			}
		}
	}

	// Generate all one matrix
	template<class T, int M, int N>
	void ONES_MAT(T A[M][N]) {
		for (int i = 0; i < M; i++) {
			for (int j = 0; j < N; j++) {
				A[i][j] = 1;
			}
		}
	}

	// Generate identity matrix
	template<class T, int M, int N>
	void IDENDTITY_MAT(T A[M][N]) {
		for (int i = 0; i < M; i++) {
			for (int j = 0; j < N; j++) {
				if (i == j)
					A[i][j] = 1;
				else
					A[i][j] = 0;
			}
		}
	}
	template<class T, int M, int N>
	void IDENDTITY_MAT(T **A) {
		for (int i = 0; i < M; i++) {
			for (int j = 0; j < N; j++) {
				if (i == j)
					A[i][j] = 1;
				else
					A[i][j] = 0;
			}
		}
	}

	// Generate random matrix
	template<class T, int M, int N>
	void RND_MAT(T A[M][N]) {
		srand(time(NULL));
		for (int i = 0; i < M; i++) {
			for (int j = 0; j < N; j++) {
				double rnd = 2 * (std::rand() % FLOAT_SIZE) / FLOAT_SIZE - 1;
				double rndnum = (double) INTEGER_SCALE * rnd;
				A[i][j] = (T) rndnum;
			}
		}
	}

	// Generate random matrix
	template<class T, int M, int N>
	void RND_DIAGMAT(T A[M][N]) {
		srand(time(NULL));
		int sparse_num = floor( DIAG_RATIO * M);
		for (int i = 0; i < M; i++) {
			for (int j = 0; j < N; j++) {
				if (i < sparse_num && i == j) {
					double rnd = 2 * (std::rand() % FLOAT_SIZE) / FLOAT_SIZE
							- 1;
					double rndnum = (double) INTEGER_SCALE * rnd;
					A[i][j] = (T) rndnum;
				} else
					A[i][j] = 0;
			}
		}
	}

	// Generate all zero vector
	template<class T, int M>
	void ZEROS_VEC(T V[M]) {
		for (int i = 0; i < M; i++) {
			V[i] = 0;
		}
	}

	// Generate all one vector
	template<class T, int M>
	void ONES_VEC(T V[M]) {
		for (int i = 0; i < M; i++) {
			V[i] = 1;
		}
	}

	// Generate random vector
	template<class T, int M>
	void RND_VEC(T V[M]) {
		srand(time(NULL));
		for (int i = 0; i < M; i++) {
			double rnd = 2 * (std::rand() % FLOAT_SIZE) / FLOAT_SIZE - 1;
			double rndnum = (double) INTEGER_SCALE * rnd;
			V[i] = (T) rndnum;
		}
	}
	template<class T, int M>
	void RND_VEC_SCALE(T V[M], T scale) {
		srand(time(NULL));
		for (int i = 0; i < M; i++) {
			double rnd = 2 * (std::rand() % FLOAT_SIZE) / FLOAT_SIZE - 1;
			double rndnum = (double) scale * INTEGER_SCALE * rnd;
			V[i] = (T) rndnum;
		}
	}

	// Transfer vector values
	template<class T, int M>
	void VEC_EQ(T V1[M], T V2[M]) {
		for (int i = 0; i < M; i++) {
			V2[i] = V1[i];
		}
	}
	template<class T1, class T2, int M>
	void VEC_EQ(T1 V1[M], T2 V2[M]) {
		for (int i = 0; i < M; i++) {
			V2[i] = (T2) V1[i];
		}
	}

	// reverse the sign of vector elements
	template<class T, int M>
	void VEC_MINUS(T V1[M], T V2[M]) {
		for (int i = 0; i < M; i++) {
			V2[i] = -V1[i];
		}
	}

	// Vector addition
	template<class T, int M>
	void VEC_ADD(T V1[M], T V2[M], T V3[M]) {
		for (int i = 0; i < M; i++) {
			V3[i] = V1[i] + V2[i];
		}
	}
	template<class T1, class T2, class T3, int M>
	void VEC_ADD(T1 V1[M], T2 V2[M], T3 V3[M]) {
		for (int i = 0; i < M; i++) {
			V3[i] = V1[i] + V2[i];
		}
	}

	// Vector subtraction
	template<class T, int M>
	void VEC_SUB(T V1[M], T V2[M], T V3[M]) {
		for (int i = 0; i < M; i++) {
			V3[i] = V1[i] - V2[i];
		}
	}

	// Vector multiplication
	template<class T, int M>
	void VEC_MUL_2SCALAR(T V1[M], T V2[M], T S) {
		S = 0;
		for (int i = 0; i < M; i++) {
			S += V1[i] * V2[i];
		}
	}
	template<class T1, class T2, class T3, int M>
	void VEC_MUL_2SCALAR(T1 V1[M], T2 V2[M], T3 S) {
		S = 0;
		for (int i = 0; i < M; i++) {
			S += V1[i] * V2[i];
		}
	}
	template<class T, int M, int N>
	void VEC_MUL_2MATRIX(T V1[M], T V2[N], T M3[M][N]) {
		for (int i = 0; i < M; i++) {
			for (int j = 0; j < N; j++) {
				M3[i][j] = V1[i] * V2[j];
			}
		}
	}
	template<class T1, class T2, class T3, int M, int N>
	void VEC_MUL_2MATRIX(T1 V1[M], T2 V2[N], T3 M3[M][N]) {
		for (int i = 0; i < M; i++) {
			for (int j = 0; j < N; j++) {
				M3[i][j] = V1[i] * V2[j];
			}
		}
	}

	// Vector division
	template<class T, int M>
	void VEC_DIV(T V1[M], T V2[M], T V3[M]) {
		for (int i = 0; i < M; i++) {
			V3[i] = V1[i] / V2[i];
		}
	}
	template<class T, int M>
	void VEC_DIV(T V1[M], T v, T V3[M]) {
		for (int i = 0; i < M; i++) {
			V3[i] = V1[i] / v;
		}
	}

	// Vector norm
	template<class T, int M>
	void VEC_NORM(T V1[M], T &S) {
		S = 0;
		for (int i = 0; i < M; i++) {
			S += V1[i] * V1[i];
		}
		double tmp = (double) S;
		S = (T) std::sqrt(tmp);
	}
	template<class T1, class T2, int M>
	void VEC_NORM(T1 V1[M], T2 &S) {
		S = 0;
		for (int i = 0; i < M; i++) {
			S += V1[i] * V1[i];
		}
		double tmp = (double) S;
		S = (T2) std::sqrt(tmp);
	}

	// Vector norm .^2
	template<class T, int M>
	void VEC_NORM2(T V1[M], T &S) {
		S = 0;
		for (int i = 0; i < M; i++) {
			S += V1[i] * V1[i];
		}
	}
	template<class T1, class T2, int M>
	void VEC_NORM2(T1 V1[M], T2 &S) {
		S = 0;
		for (int i = 0; i < M; i++) {
			S += V1[i] * V1[i];
		}
	}

	// Vector square norm
	template<class T, int M>
	void VEC_SQUARE_NORM(T V1[M], T &S) {
		S = 0;
		for (int i = 0; i < M; i++) {
			S += V1[i] * V1[i];
		}
	}
	template<class T1, class T2, int M>
	void VEC_SQUARE_NORM(T1 V1[M], T2 &S) {
		S = 0;
		for (int i = 0; i < M; i++) {
			S += V1[i] * V1[i];
		}
	}

	// Vector scalar addition
	template<class T, int M>
	void VEC_SCALAR_ADD(T V1[M], T S, T V3[M]) {
		for (int i = 0; i < M; i++) {
			V3[i] = V1[i] + S;
		}
	}
	template<class T1, class T2, class T3, int M>
	void VEC_SCALAR_ADD(T1 V1[M], T2 S, T3 V3[M]) {
		for (int i = 0; i < M; i++) {
			V3[i] = V1[i] + S;
		}
	}

	// Vector scalar subtraction
	template<class T, int M>
	void VEC_SCALAR_SUB(T V1[M], T S, T V3[M]) {
		for (int i = 0; i < M; i++) {
			V3[i] = V1[i] - S;
		}
	}
	template<class T1, class T2, class T3, int M>
	void VEC_SCALAR_SUB(T1 V1[M], T2 S, T3 V3[M]) {
		for (int i = 0; i < M; i++) {
			V3[i] = V1[i] - S;
		}
	}

	// Vector scalar multiplication
	template<class T, int M>
	void VEC_SCALAR_MUL(T V1[M], T S, T V3[M]) {
		for (int i = 0; i < M; i++) {
			V3[i] = V1[i] * S;
		}
	}
	template<class T, int M>
	void VEC_SCALAR_MUL(T S, T V1[M], T V3[M]) {
		for (int i = 0; i < M; i++) {
			V3[i] = V1[i] * S;
		}
	}

	// Vector scalar compare
	template<class T, int M>
	void VEC_SCALAR_MIN(T V1[M], T S, T V3[M]) {
		for (int i = 0; i < M; i++) {
			V3[i] = V1[i] < S ? V1[i] : S;
		}
	}
	template<class T1, class T2, class T3, int M>
	void VEC_SCALAR_MIN(T1 V1[M], T2 S, T3 V3[M]) {
		for (int i = 0; i < M; i++) {
			V3[i] = V1[i] < S ? V1[i] : S;
		}
	}
	template<class T, int M>
	void VEC_SCALAR_MAX(T V1[M], T S, T V3[M]) {
		for (int i = 0; i < M; i++) {
			V3[i] = V1[i] > S ? V1[i] : S;
		}
	}
	template<class T1, class T2, class T3, int M>
	void VEC_SCALAR_MAX(T1 V1[M], T2 S, T3 V3[M]) {
		for (int i = 0; i < M; i++) {
			V3[i] = V1[i] > S ? V1[i] : S;
		}
	}

	// The basic matrix addition with complexity of O(NN)
	template<class T, int M, int N>
	void MAT_ADD(T A[M][N], T B[M][N], T C[M][N]) {
		for (int i = 0; i < M; i++) {
			for (int j = 0; j < N; j++) {
				C[i][j] = A[i][j] + B[i][j];
			}
		}
	}
	template<class T, int M, int N>
	void MAT_ADD(T **A, T **B, T **C) {
		for (int i = 0; i < M; i++) {
			for (int j = 0; j < N; j++) {
				C[i][j] = A[i][j] + B[i][j];
			}
		}
	}
	template<class T1, class T2, class T3, int M, int N>
	void MAT_ADD(T1 A[M][N], T2 B[M][N], T3 C[M][N]) {
		for (int i = 0; i < M; i++) {
			for (int j = 0; j < N; j++) {
				C[i][j] = A[i][j] + B[i][j];
			}
		}
	}

	

	// Matrix Differential
	template<class T, int M, int N, int dim, int order>
	void MAT_DIFF(T MatA[M][N], T MatB[M][N], int dir) {
		T MatTmp1[M][N];
		T MatTmp2[M][N];
		MAT_EQ<T, M, N>(MatA, MatTmp1);
		if (dim == 1) {
			for (int i = 0; i < order; i++) {
				for (int j = 0; j < N; j++) {
					for (int k = 0; k < M - i - 1; k++) {
						MatTmp2[k][j] = MatTmp1[k + 1][j] - MatTmp1[k][j];
						MatTmp2[k][j] *= dir;
					}
				}
				MAT_EQ<T, M, N>(MatTmp2, MatTmp1);
			}
			MAT_EQ<T, M, N>(MatTmp1, MatB);
		} else if (dim == 2) {
			for (int i = 0; i < order; i++) {
				for (int j = 0; j < M; j++) {
					for (int k = 0; k < N - i - 1; k++) {
						MatTmp2[j][k] = MatTmp1[j][k + 1] - MatTmp1[j][k];
						MatTmp2[j][k] *= dir;
					}
				}
				MAT_EQ<T, M, N>(MatTmp2, MatTmp1);
			}
			MAT_EQ<T, M, N>(MatTmp1, MatB);
		} else {
			std::cout << "only support 2 dimension !!" << std::endl;
			std::exit(0);
		}
	}

	// From real matrix to complex matrix
	template<class T, int M, int N>
	void MAT_REAL2COMPLEX(T MatA[M][N], Complex<T> MatB[M][N]) {
		for (int i = 0; i < M; i++) {
			for (int j = 0; j < N; j++) {
				MatB[i][j].real = MatA[i][j];
				MatB[i][j].imag = 0;
			}
		}
	}

	// Extract real matrix from complex matrix
	template<class T, int M, int N>
	void MAT_COMPLEX_GETREAL(Complex<T> MatA[M][N], T MatB[M][N]) {
		for (int i = 0; i < M; i++) {
			for (int j = 0; j < N; j++) {
				MatB[i][j] = MatA[i][j].real;
			}
		}
	}

	// Extract real submatrix from complex matrix
	template<class T, int M, int N, int P, int Q>
	void MAT_COMPLEX_GETREALSUBMAT(Complex<T> MatA[M][N], T MatB[P][Q], int row,
			int col) {
		for (int i = row; i < P + row; i++) {
			for (int j = col; j < Q + col; j++) {
				MatB[i][j] = MatA[i][j].real;
			}
		}
	}

	// Extract imag matrix from complex matrix
	template<class T, int M, int N>
	void MAT_COMPLEX_GETIMAG(Complex<T> MatA[M][N], T MatB[M][N]) {
		for (int i = 0; i < M; i++) {
			for (int j = 0; j < N; j++) {
				MatB[i][j] = MatA[i][j].imag;
			}
		}
	}

	// Extract imag submatrix from complex matrix
	template<class T, int M, int N, int P, int Q>
	void MAT_COMPLEX_GETIMAGSUBMAT(Complex<T> MatA[M][N], T MatB[P][Q], int row,
			int col) {
		for (int i = row; i < P + row; i++) {
			for (int j = col; j < Q + col; j++) {
				MatB[i][j] = MatA[i][j].imag;
			}
		}
	}

	// Matrix compare scalar
	template<class T1, class T2, int M, int N>
	void MAT_MAXCMP(T1 MatA[M][N], T2 B, T1 MatC[M][N]) {
		for (int i = 0; i < M; i++)
			for (int j = 0; j < N; j++)
				MatC[i][j] = MatA[i][j] >= B ? MatA[i][j] : B;
	}
	template<class T1, class T2, int M, int N>
	void MAT_MAXCMP(T2 B, T1 MatA[M][N], T1 MatC[M][N]) {
		for (int i = 0; i < M; i++)
			for (int j = 0; j < N; j++)
				MatC[i][j] = MatA[i][j] >= B ? MatA[i][j] : B;
	}
	template<class T1, class T2, int M, int N>
	void MAT_MINCMP(T1 MatA[M][N], T2 B, T1 MatC[M][N]) {
		for (int i = 0; i < M; i++)
			for (int j = 0; j < N; j++)
				MatC[i][j] = MatA[i][j] >= B ? B : MatA[i][j];
	}
	template<class T1, class T2, int M, int N>
	void MAT_MINCMP(T2 B, T1 MatA[M][N], T1 MatC[M][N]) {
		for (int i = 0; i < M; i++)
			for (int j = 0; j < N; j++)
				MatC[i][j] = MatA[i][j] >= B ? B : MatA[i][j];
	}

	// Matrix sign
	template<class T, int M, int N>
	void MAT_SIGN(T MatA[M][N], T MatB[M][N]) {
		for (int i = 0; i < M; i++)
			for (int j = 0; j < N; j++)
				MatB[i][j] = MatA[i][j] >= 0 ? 1 : -1;
	}

	// Matrix absolute value
	template<class T, int M, int N>
	void MAT_ABS(T MatA[M][N], T MatB[M][N]) {
		for (int i = 0; i < M; i++)
			for (int j = 0; j < N; j++)
				MatB[i][j] = MatA[i][j] >= 0 ? MatA[i][j] : -MatA[i][j];
	}

	// Matrix dot multiplication
	template<class T, int M, int N>
	void MAT_DOTSQUARE(T MatA[M][N], T MatB[M][N]) {
		for (int i = 0; i < M; i++)
			for (int j = 0; j < N; j++)
				MatB[i][j] = MatA[i][j] * MatA[i][j];
	}

	// Matrix dot multiplication
	template<class T, int M, int N>
	void MAT_DOTMUL(T MatA[M][N], T MatB[M][N], T MatC[M][N]) {
		for (int i = 0; i < M; i++)
			for (int j = 0; j < N; j++)
				MatC[i][j] = MatA[i][j] * MatB[i][j];
	}

	// Matrix dot division
	template<class T, int M, int N>
	void MAT_DOTDIV(T MatA[M][N], T MatB[M][N], T MatC[M][N]) {
		for (int i = 0; i < M; i++)
			for (int j = 0; j < N; j++)
				MatC[i][j] = MatA[i][j] / MatB[i][j];
	}

	// Matrix dot division
	template<class T, int M, int N>
	void MAT_COMPLEX_DOTDIV_REAL(Complex<T> MatA[M][N], T MatB[M][N],
			Complex<T> MatC[M][N]) {
		for (int i = 0; i < M; i++) {
			for (int j = 0; j < N; j++) {
				MatC[i][j].real = MatA[i][j].real / MatB[i][j];
				MatC[i][j].imag = MatA[i][j].imag / MatB[i][j];
			}
		}
	}
	template<class T, int M, int N, int P, int Q>
	void MAT_COMPLEX_DOTDIV_REAL(Complex<T> MatA[M][N], T MatB[P][Q],
			Complex<T> MatC[M][N]) {
		for (int i = 0; i < P; i++) {
			for (int j = 0; j < Q; j++) {
				MatC[i][j].real = MatA[i][j].real / MatB[i][j];
				MatC[i][j].imag = MatA[i][j].imag / MatB[i][j];
			}
		}
	}

	// Matrix dot inverse
	template<class T, int M, int N>
	void MAT_DOTINV(T MatA[M][N], T MatB[M][N]) {
		for (int i = 0; i < M; i++)
			for (int j = 0; j < N; j++)
				MatB[i][j] = (T) 1 / MatA[i][j];
	}

	// Matrix dot multiply scalar
	template<class T1, class T2, int M, int N>
	void MAT_SCALAR_DOTADD(T1 MatA[M][N], T2 B, T1 MatC[M][N]) {
		for (int i = 0; i < M; i++)
			for (int j = 0; j < N; j++)
				MatC[i][j] = MatA[i][j] + B;
	}
	template<class T1, class T2, int M, int N>
	void MAT_SCALAR_DOTADD(T2 B, T1 MatA[M][N], T1 MatC[M][N]) {
		for (int i = 0; i < M; i++)
			for (int j = 0; j < N; j++)
				MatC[i][j] = MatA[i][j] + B;
	}

	// Matrix dot multiply scalar
	template<class T1, class T2, int M, int N>
	void MAT_SCALAR_DOTSUB(T1 MatA[M][N], T2 B, T1 MatC[M][N]) {
		for (int i = 0; i < M; i++)
			for (int j = 0; j < N; j++)
				MatC[i][j] = MatA[i][j] - B;
	}
	template<class T1, class T2, int M, int N>
	void MAT_SCALAR_DOTSUB(T2 B, T1 MatA[M][N], T1 MatC[M][N]) {
		for (int i = 0; i < M; i++)
			for (int j = 0; j < N; j++)
				MatC[i][j] = B - MatA[i][j];
	}

	// Matrix dot multiply scalar
	template<class T, int M, int N>
	void MAT_SCALAR_DOTMUL(T MatA[M][N], T B, T MatC[M][N]) {
		for (int i = 0; i < M; i++)
			for (int j = 0; j < N; j++)
				MatC[i][j] = MatA[i][j] * B;
	}
	template<class T, int M, int N>
	void MAT_SCALAR_DOTMUL(T **MatA, T B, T **MatC) {
		for (int i = 0; i < M; i++)
			for (int j = 0; j < N; j++)
				MatC[i][j] = MatA[i][j] * B;
	}
	template<class T1, class T2, int M, int N>
	void MAT_SCALAR_DOTMUL(T1 MatA[M][N], T2 B, T1 MatC[M][N]) {
		for (int i = 0; i < M; i++)
			for (int j = 0; j < N; j++)
				MatC[i][j] = MatA[i][j] * B;
	}
	template<class T1, class T2, int M, int N>
	void MAT_SCALAR_DOTMUL(T2 B, T1 MatA[M][N], T1 MatC[M][N]) {
		for (int i = 0; i < M; i++)
			for (int j = 0; j < N; j++)
				MatC[i][j] = MatA[i][j] * B;
	}

	// Matrix dot division scalar
	template<class T1, class T2, int M, int N>
	void MAT_SCALAR_DOTDIV(T1 MatA[M][N], T2 B, T1 MatC[M][N]) {
		for (int i = 0; i < M; i++)
			for (int j = 0; j < N; j++)
				MatC[i][j] = MatA[i][j] / B;
	}
	template<class T1, class T2, int M, int N>
	void MAT_SCALAR_DOTDIV(T2 B, T1 MatA[M][N], T1 MatC[M][N]) {
		for (int i = 0; i < M; i++)
			for (int j = 0; j < N; j++)
				MatC[i][j] = B / MatA[i][j];
	}

	// Extract subvector
	template<class T, int M, int N>
	void VEC_SUBVEC(T VecA[M], T VecB[N], int start) {
		for (int i = start, k = 0; i < M && k < N; i++, k++)
			VecB[k] = VecA[i];
	}
	template<class T, int M, int N>
	void VEC_SUBVEC(T VecA[M], T VecB[N], int start, int step) {
		for (int i = start, k = 0; i < M && k < N; i += step, k++)
			VecB[k] = VecA[i];
	}

	// Vector merge
	template<class T, int M, int N>
	void VEC_MERGE2VEC(T VecA[M], T VecB[N], T VecC[M + N]) {
		for (int i = 0; i < M; i++)
			VecC[i] = VecA[i];
		for (int i = M, k = 0; i < N + M && k < N; i++, k++)
			VecC[i] = VecB[i];

	}
	template<class T, int M>
	void VEC_MERGE2MAT(T VecA[M], T VecB[M], T MatC[2][M]) {
		for (int i = 0; i < M; i++) {
			MatC[0][i] = VecA[i];
			MatC[1][i] = VecB[i];
		}
	}

	// Vector differential
	template<class T, int M>
	void VEC_DIFF(T VecA, T VecB, int order) {
		T VecTmp = VecA;
		for (int i = 0; i < order; i++) {
			for (int j = 0; j < M - i - 1; j++)
				VecB[i] = VecTmp[i + 1] - VecTmp[i];
			VecTmp = VecB;
		}
	}

	// Vector absolute value
	template<class T, int M>
	void VEC_ABS(T VecA[M], T VecB[M]) {
		for (int i = 0; i < M; i++)
			VecB[i] = VecA[i] >= 0 ? VecA[i] : -VecA[i];
	}

	// Vector dot division
	template<class T, int M>
	void VEC_DOTDIV(T MatA[M], T MatB[M], T MatC[M]) {
		for (int i = 0; i < M; i++)
			MatC[i] = MatA[i] / MatB[i];
	}

	// Vector dot division scalar
	template<class EigenT, class T, int M>
	void VEC_SCALAR_DOTDIV(T MatA[M], T B, T MatC[M]) {
		for (int i = 0; i < M; i++)
			MatC[i] = MatA[i] / B;
	}
	template<class EigenT, class T, int M>
	void VEC_SCALAR_DOTDIV(T A, T MatB[M], T MatC[M]) {
		for (int i = 0; i < M; i++)
			MatC[i] = A / MatB[i];
	}
	// 0427 ADD Matrix conv

	template<class T, int In_rows, int In_cols, int Stride, int Kernel_size,
			int Out_rows, int Out_cols>
	void CONV_2D(T MatA[In_rows][In_cols], T Kernel[Kernel_size][Kernel_size],
			T MatB[Out_rows][Out_cols], bool padding = false) {
		constexpr int Padding = padding ? (Kernel_size - 1) / 2 : 0;
		static_assert(Out_rows == (In_rows + 2 * Padding - Kernel_size)/Stride +1,"Conv output row dismatch!");
		static_assert(Out_cols == (In_cols + 2 * Padding - Kernel_size)/Stride +1,"Conv output col dismatch!");
		for (int out_i = 0; out_i < Out_rows; out_i += Stride) {
			for (int out_j = 0; out_j < Out_cols; out_j += Stride) {
				T sum = 0;
				for (int k_i = 0; k_i < Kernel_size; k_i += Stride) {
					for (int k_j = 0; k_j < Kernel_size; k_j += Stride) {
						const int in_i = out_i + k_i;
						const int in_j = out_j + k_j;
						if (in_i < 0 || in_i > In_rows || in_j < 0
								|| in_j > In_cols) {
							sum = sum;
						} else {
							sum = sum + MatA[in_i][in_j] * Kernel[k_i][k_j];
						}

					}
				}
				MatB[out_i][out_j] = sum;
			}
		}
	}

	// 0427 ADD bilinear_interpolation

	template<class T, int In_rows, int In_cols, int Out_rows, int Out_cols>
	void B_INTERPOL(T MatA[In_rows][In_cols], T MatB[Out_rows][Out_cols])

	{
		if constexpr (Out_rows == 1 && Out_cols == 1) {
			MatB[0][0] = MatA[0][0];
		}

		const float x_scale =
				(In_rows > 1) ? (In_rows - 1.0f) / (Out_rows - 1.0f) : 0;
		const float y_scale =
				(In_cols > 1) ? (In_cols - 1.0f) / (Out_cols - 1.0f) : 0;

		for (int x = 0; x < Out_rows; ++x) {
			float src_x = (In_rows > 1) ? x * x_scale : 0.5f;
			int x0 = static_cast<int>(src_x);
			int x1 = std::min(x0 + 1, In_rows - 1);
			float dx = src_x - x0;
		}
		for (int y = 0; y < Out_cols; ++y) {
			float src_y = (In_cols > 1) ? y * y_scale : 0.5f;
			int y0 = static_cast<int>(src_y);
			int y1 = std::min(y0 + 1, In_cols - 1);
			float dy = src_y - y0;

			T val00 = MatA[x0][y0];
			T val01 = MatA[x0][y1];
			T val10 = MatA[x1][y0];
			T val11 = MatA[x1][y1];

			T n_val0 = val00 + dx * (val01 - val00);
			T n_val1 = val10 + dy * (val11 - val10);

			MatB[x][y] = n_val0 + dy * (n_val1 - n_val0);

		}

	}

	//0429 add some operate without template

	//add random vec without template
	//using & to change value in vector

	void RND_VEC_NT(Vec &V) {
		srand(time(NULL));
		int M = V.size();
		for (int i = 0; i < M; i++) {
			double rnd = 2 * (std::rand() % FLOAT_SIZE) / FLOAT_SIZE - 1;
			double rndnum = (double) INTEGER_SCALE * rnd;
			V[i] = (CF) rndnum;
		}
	}

	//add random mat without template
	//using & to change value in vector
	void RND_MAT_NT(Mat &A) {
		srand(time(NULL));
		int M = A.size();
		int N = A[0].size();
		for (int i = 0; i < M; i++) {
			for (int j = 0; j < N; j++) {
				double rnd = 2 * (std::rand() % FLOAT_SIZE) / FLOAT_SIZE - 1;
				double rndnum = (double) INTEGER_SCALE * rnd;
				A[i][j] = (CF) rndnum;
			}
		}
	}

	//add 2dconv without template
	//using & to change value in vector
	void CONV_2D_NT(const Tri &MatA, const Convk &Kernel, const Vec &Bias,
			Tri &MatB, const bool padding = true, const int Stride = 1) {
		int Padding = padding ? (Kernel[0][0].size() - 1) / 2 : 0;
		int Out_rows = (MatA[0].size() + 2 * Padding - Kernel[0][0].size())
				/ Stride + 1;
		int Out_cols = (MatA[0][0].size() + 2 * Padding - Kernel[0][0].size())
				/ Stride + 1;
//		static_assert(Out_rows == (In_rows + 2 * Padding - Kernel_size)/Stride +1,"Conv output row dismatch!");
//		static_assert(Out_cols == (In_cols + 2 * Padding - Kernel_size)/Stride +1,"Conv output col dismatch!");
		assert(Kernel.size() > 0 && Kernel[0].size() > 0);
		assert(bias.size() == Kernel.size());

		//resize MatB
		MatB.resize(Kernal.size());
		for (int i = 0; i < Kernal.size(); i++) {
			MatB[i].resize(Out_rows);
			for (int j = 0; j < Out_rows; j++) {
				MatB[i][j].resize(Out_cols);
			}
		}

		//conv operation
		for (int c = 0; c < Kernal.size(); c++) {
			for (int out_i = 0; out_i < Out_rows; out_i += Stride) {
				for (int out_j = 0; out_j < Out_cols; out_j += Stride) {
					T sum = 0;
					for (int in_ch = 0; in_ch < MatA.size(); in_ch++) {
						for (int k_i = 0; k_i < Kernel[0][0].size(); k_i +=
								Stride) {
							for (int k_j = 0; k_j < Kernel[0][0].size(); k_j +=
									Stride) {
								int in_i = out_i + k_i;
								int in_j = out_j + k_j;
								if (in_i < 0 || in_i > In_rows || in_j < 0
										|| in_j > In_cols) {
									sum = sum;
								} else {
									sum =
											sum
													+ MatA[in_ch][in_i][in_j]
															* Kernel[c][in_ch][k_i][k_j];
								}
							}
						}
					}
					MatB[c][out_i][out_j] = sum + Bias[c];
				}
			}
		}
	}

	//add RELU and it's Vec,Mat version
	//using & to change value in vector

	CF RELU(CF val) {
		return (val > CF(0)) ? val : CF(0);
	}

	//RELU in Vec
	void RELU_VEC(Vec &vec) {
		for (auto &val : vec) {
			val = RELU(val);
		}
	}

	//RELU in Mat
	void RELU_MAT(Mat &mat) {
		for (auto &row : mat) {
			row = RELU_VEC(row);
		}
	}

	//MAT bilinear_interpolation without template
	//use large ap type to do with outrange data

	Mat B_INTERPLO_NT(const Mat &input, float scale = 2.0f)
	{

		int out_rows = static_cast<int>(input.size() * scale);
		int out_cols = static_cast<int>(input[0].size() * scale);
		const int in_rows = input.size();
		const int in_cols = input[0].size();

		Mat output(out_rows, Vec(out_cols, 0));

		for (int x = 0; x < out_rows; ++x) {
			const float src_x = (in_rows > 1) ? x * x_scale : 0.5f;
			const int x0 = static_cast<int>(src_x);
			const int x1 = std::min(x0 + 1, in_rows - 1);
			const float dx = src_x - x0;
		}
		for (int y = 0; y < out_cols; ++y) {
			const float src_y = (in_cols > 1) ? y * y_scale : 0.5f;
			const int y0 = static_cast<int>(src_y);
			const int y1 = std::min(y0 + 1, in_cols - 1);
			const float dy = src_y - y0;

			const CF val00 = (x0 < in_rows && y0 < in_cols ? input[x0][y0] : 0);
			const CF val01 = (x0 < in_rows && y1 < in_cols ? input[x0][y1] : 0);
			const CF val10 = (x1 < in_rows && y0 < in_cols ? input[x1][y0] : 0);
			const CF val11 = (x1 < in_rows && y1 < in_cols ? input[x1][y1] : 0);

			const CFF dx_fix = dx;
			const CFF dy_fix = dy;

			const CFF interp_x0 = val00 + dx_fix * (val01 - val00);
			const CFF interp_x1 = val10 + dx_fix * (val11 - val10);
			const CFF final_val = interp_x0 + dy_fix * (interp_x1 - interp_x0);

			output[x][y] = final_val.range(3, 0);

		}
		return output;

	}

	//0430 TRI B_INTERPLO
	Tri TRI_B_INTERPLO_NT(const Tri& input ,float scale = 2.0f)
	{
		Tri output(input.size());
		for(int c = 0;c < input.size();++c)
		{
			Mat chan_in = input[c];
			Mat chan_out;
			fpa.B_INTERPLO_NT(chan_in,chan_out,scale);
			output[c] = chan_out;
		}
		return output;
	}




	//0430
	//ADD matrix add without template
	// The basic matrix addition with complexity of O(NN)
	Mat MAT_ADD_NT(Mat& A, Mat& B)
	{
		Mat C = A;
		for (int i = 0; i < A.size(); i++) {
			for (int j = 0; j < A[0].size(); j++) {
				C[i][j] = A[i][j] + B[i][j];
			}
		}
		return C;
	}

	Tri TRI_ADD_NT(Tri& A,Tri& B)
	{
		Tri C =A;
		for(int i = 0;i < A.size();i++)
		{
			for(int j = 0;j < A[0].size();j++)
			{
				for(int k = 0;k < A[0][0].size;k++)
				{
					C[i][j][k] = A[i][j][k] + B[i][j][k];
				}
			}
		}
		return C;
	}

private:

protected:

};

#endif /* SRC_FPT_ALGEBRA_HPP_ */
