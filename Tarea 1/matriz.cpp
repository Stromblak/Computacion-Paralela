#include <bits/stdc++.h>
#include <omp.h>
using namespace std;
using namespace chrono;

typedef vector<vector<int>> vii;
// matriz.exe <n>


void imprimir(vii &M){
	cout << "Matriz" << endl;
	for(int i=0; i<M.size(); i++){
		for(int j=0; j<M.size(); j++){
			cout << M[i][j] << " ";
		}
		cout << endl;
	}
}


// Matrices tradicionales
void mat_trad_sec(vii &A, vii &B, vii &C, int n){
	for(int i=0; i<n; i++){
		for(int j=0; j<n; j++){
			for(int k=0; k<n; k++) C[i][j] += A[i][k] * B[k][j];
		}
	}
}

void mat_trad_par(vii &A, vii &B, vii &C, int n){
	for(int i=0; i<n; i++){
		#pragma omp parallel for
			for(int j=0; j<n; j++){
				for(int k=0; k<n; k++) 
					C[i][j] += A[i][k] * B[k][j];
			}
	}
}


// Matrices amigables con el cache
void mat_amigable_sec(vii &A, vii &B, vii &C, int n){
	vector<vector<int>> BT(n, vector<int>(n));
	for(int i=0; i<n; i++){
		for(int j=0; j<n; j++) BT[i][j] = B[j][i];
	}

	for(int i=0; i<n; i++){
		for(int j=0; j<n; j++){
			int suma = 0;
			for(int k=0; k<n; k++){
				suma += A[i][k] * BT[j][k];
			}
			C[i][j] = suma;
		}
	}
}

void mat_amigable_par(vii &A, vii &B, vii &C, int n){
	vector<vector<int>> BT(n, vector<int>(n));
	for(int i=0; i<n; i++){
		for(int j=0; j<n; j++) BT[i][j] = B[j][i];
	}

	for(int i=0; i<n; i++){
		#pragma omp parallel for
			for(int j=0; j<n; j++){
				int suma = 0;
				for(int k=0; k<n; k++){
					suma += A[i][k] * BT[j][k];
				}
				C[i][j] = suma;
			}
	}
}


// Strassen
// ?????
// https://www.youtube.com/watch?v=OSelhO6Qnlc
vii suma(vii M1, vii M2){
	int tam = M1.size();
	vii M(tam, vector<int>(tam));

	for(int i=0; i<tam; i++){
		for(int j=0; j<tam; j++){
			M[i][j] = M1[i][j] + M2[i][j];
		}
	}

	return M;
}

vii resta(vii M1, vii M2){
	int tam = M1.size();
	vii M(tam, vector<int>(tam));

	for(int i=0; i<tam; i++){
		for(int j=0; j<tam; j++){
			M[i][j] = M1[i][j] - M2[i][j];
		}
	}

	return M;
}

vii strassen(vii A, vii B){
	if(A.size() == 2){
		int A11 = A[0][0], A12 = A[0][1]; 
		int A21 = A[1][0], A22 = A[1][1];
		int B11 = B[0][0], B12 = B[0][1]; 
		int B21 = B[1][0], B22 = B[1][1];

		int M1 = (A11 + A22) * (B11 + B22);
		int M2 = (A21 + A22) * B11;
		int M3 = A11 * (B12 - B22);
		int M4 = A22 * (B21 - B11);
		int M5 = (A11 + A12) * B22;
		int M6 = (A21 - A11) * (B11 + B12);
		int M7 = (A12 - A22) * (B21 + B22);	
				
		vii C(2, vector<int>(2));
		C[0][0] = M1 + M4 - M5 + M7;
		C[0][1] = M3 + M5;
		C[1][0] = M2 + M4;
		C[1][1] = M1 - M2 + M3 + M6;
		
		return C;
	}

	int tam = A.size()/2;

	vii A11(tam, vector<int>(tam)), A12(tam, vector<int>(tam));
	vii A21(tam, vector<int>(tam)), A22(tam, vector<int>(tam));
	vii B11(tam, vector<int>(tam)), B12(tam, vector<int>(tam));
	vii B21(tam, vector<int>(tam)), B22(tam, vector<int>(tam));

	for(int i=0; i<tam; i++){
		for(int j=0; j<tam; j++){
			A11[i][j] = A[i][j];
			A12[i][j] = A[i][j+tam];
			A21[i][j] = A[i+tam][j];
			A22[i][j] = A[i+tam][j+tam];
			B11[i][j] = A[i][j];
			B12[i][j] = A[i][j+tam];
			B21[i][j] = A[i+tam][j];
			B22[i][j] = A[i+tam][j+tam];
		}
	}

	vii M1 = strassen( suma(A11, A22), suma(B11, B22) );
	vii M2 = strassen( suma(A21, A22), B11 );
	vii M3 = strassen( A11, resta(B12, B22) );
	vii M4 = strassen( A22, resta(B21, B11) );
	vii M5 = strassen( suma(A11, A12), B22 );
	vii M6 = strassen( resta(A21, A11), suma(B11, B12) );
	vii M7 = strassen( resta(A12, A22), suma(B21, B22) );


	vii C11 = suma( suma(M1, M4), resta(M7, M5));
	vii C12 = suma(M3, M5);
	vii C21 = suma(M2, M4);
	vii C22 = suma( resta(M1, M2), suma(M3, M6));

	vii C(A.size(), vector<int>(A.size()));
	for(int i=0; i<tam; i++) for(int j=0; j<tam; j++) C[i][j] = C11[i][j];
	for(int i=0; i<tam; i++) for(int j=0; j<tam; j++) C[i][j+tam] = C12[i][j];
	for(int i=0; i<tam; i++) for(int j=0; j<tam; j++) C[i+tam][j] = C21[i][j];
	for(int i=0; i<tam; i++) for(int j=0; j<tam; j++) C[i+tam][j+tam] = C22[i][j];

	return C;
}

int main(int argc, char *argv[]){
	minstd_rand rng;
	rng.seed(10);

	int n = 2 << (atoi(argv[1]) - 1);
	// int m = atoi(argv[2]);

	vector<vector<int>> A(n, vector<int>(n)), B(n, vector<int>(n));
	for(int i=0; i<n; i++){
		for(int j=0; j<n; j++){
			A[i][j] = rng()%10;
			B[j][i] = rng()%10;
		}
	}
	
	// imprimir(A);
	// imprimir(B);


	int i = 0;
	while(1){
		vector<vector<int>> C(n, vector<int>(n));

		auto start = high_resolution_clock::now();

		if(i == 0) mat_trad_sec(A, B, C, n);
		else if(i == 1) mat_trad_par(A, B, C, n);
		else if(i == 2) mat_amigable_sec(A, B, C, n);
		else if(i == 3) mat_amigable_par(A, B, C, n);
		else if(i == 4) C = strassen(A, B);
		else break;

		auto finish = high_resolution_clock::now();
		auto d = duration_cast<nanoseconds> (finish - start).count();
		cout <<"total time "<< d << " [ns]" << " \n";

		if(i == 0 || i == 4) imprimir(C);

		i++;
	}

	return 0;
}