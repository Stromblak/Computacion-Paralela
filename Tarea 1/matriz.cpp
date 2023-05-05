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
// https://www.youtube.com/watch?v=OSelhO6Qnlc
vii suma(vii &M1, vii &M2){
	int tam = M1.size();
	vii M(tam, vector<int>(tam));

	for(int i=0; i<tam; i++){
		for(int j=0; j<tam; j++){
			M[i][j] = M1[i][j] + M2[i][j];
		}
	}

	return M;
}

vii resta(vii &M1, vii &M2){
	int tam = M1.size();
	vii M(tam, vector<int>(tam));

	for(int i=0; i<tam; i++){
		for(int j=0; j<tam; j++){
			M[i][j] = M1[i][j] - M2[i][j];
		}
	}

	return M;
}

/*
void strassen(vii A, vii B, vii C){
	if(A.size() == 1){
		C[0][0] = A[0][0] * B[0][0];
		return;
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
			B11[i][j] = B[i][j];
			B12[i][j] = B[i][j+tam];
			B21[i][j] = B[i+tam][j];
			B22[i][j] = B[i+tam][j+tam];
		}
	}
	vii M1(tam, vector<int>(tam));
	vii M2(tam, vector<int>(tam));
	vii M3(tam, vector<int>(tam));
	vii M4(tam, vector<int>(tam));
	vii M5(tam, vector<int>(tam));
	vii M6(tam, vector<int>(tam));
	vii M7(tam, vector<int>(tam));

	strassen( suma(A11, A22), suma(B11, B22), M1 );
	strassen( suma(A21, A22), B11, M2 );
	strassen( A11, resta(B12, B22), M3 );
	strassen( A22, resta(B21, B11), M4 );
	strassen( suma(A11, A12), B22, M5 );
	strassen( resta(A21, A11), suma(B11, B12), M6 );
	strassen( resta(A12, A22), suma(B21, B22), M7 );


	vii C11 = suma( suma(M1, M4), resta(M7, M5));
	vii C12 = suma(M3, M5);
	vii C21 = suma(M2, M4);
	vii C22 = suma( resta(M1, M2), suma(M3, M6));

	for(int i=0; i<tam; i++){
		for(int j=0; j<tam; j++){
			C[i][j] = C11[i][j];
			C[i][j+tam] = C12[i][j];
			C[i+tam][j] = C21[i][j];
			C[i+tam][j+tam] = C22[i][j];
		}
	}
}
*/

vii strassen(vii A, vii B){
	if(A.size() == 1){
		vii C(1, vector<int>(1));
		C[0][0] = A[0][0] * B[0][0];
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
			B11[i][j] = B[i][j];
			B12[i][j] = B[i][j+tam];
			B21[i][j] = B[i+tam][j];
			B22[i][j] = B[i+tam][j+tam];
		}
	}

	vii M1 = strassen( suma(A11, A22), suma(B11, B22) );
	vii M2 = strassen( suma(A21, A22), B11 );
	vii M3 = strassen( A11, resta(B12, B22) );
	vii M4 = strassen( A22, resta(B21, B11) );
	vii M5 = strassen( suma(A11, A12), B22 );
	vii M6 = strassen( resta(A21, A11), suma(B11, B12) );
	vii M7 = strassen( resta(A12, A22), suma(B21, B22) );

	vii C(tam*2, vector<int>(tam*2));
	for(int i=0; i<tam; i++){
		for(int j=0; j<tam; j++){
			C[i][j] = M1[i][j] + M4[i][j] - M5[i][j] + M7[i][j];
			C[i][j+tam] = M3[i][j] + M5[i][j];
			C[i+tam][j] = M2[i][j] + M4[i][j];
			C[i+tam][j+tam] = M1[i][j] - M2[i][j] + M3[i][j] + M6[i][j];
		}
	}


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
	
	//imprimir(A);
	//imprimir(B);


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

		if(n == 2 && (i == 0 || i == 4) ) imprimir(C);

		i++;
	}

	return 0;
}