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
vii strasenRec(vii &A, vii &B, vii &C, int fi, int ff, int ci, int cf){

	if(ff - fi > 2){
		int delta = (ff - fi)/2;
		vii C1 = strasenRec(A, B, C, fi, fi + delta, ci, ci + delta);
		vii C2 = strasenRec(A, B, C, fi, fi + delta, ci + delta, cf);
		vii C3 = strasenRec(A, B, C, fi + delta, ff, ci + delta, cf);
		vii C4 = strasenRec(A, B, C, fi + delta, ff, ci, ci + delta);

	}else{
		int A11 = A[fi][ci], A12 = A[fi][ci+1], A21 = A[fi+1][ci], A22 = A[fi+1][ci+1];
		int B11 = B[fi][ci], B12 = B[fi][ci+1], B21 = B[fi+1][ci], B22 = B[fi+1][ci+1];

		int M1 = (A11 + A22) * (B11 + B22);
		int M2 = (A21 + A22) * B11;
		int M3 = A11 * (B12 - B22);
		int M4 = A22 * (B21 - B11);
		int M5 = (A11 + A12) * B22;
		int M6 = (A21 - A11) * (B11 + B12);
		int M7 = (A12 - A22) * (B21 + B22);

		vii Caux(2, vector<int>(2));

		Caux[0][0] = M1 + M4 - M5 + M7;
		Caux[0][1] = M3 + M5;
		Caux[1][0] = M2 + M4;
		Caux[1][1] = M1 - M2 + M3 + M6;

		return Caux;
	}

	// ?????
	// https://www.youtube.com/watch?v=OSelhO6Qnlc
}

void strassen(vii &A, vii &B, vii &C, int n){
	strasenRec(A, B, C, 0, n, 0, n);
}


int main(int argc, char *argv[]){
	minstd_rand rng;
	rng.seed(10);

	int n = atoi(argv[1]);
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
		else if(i == 4) strassen(A, B, C, n);
		else break;

		auto finish = high_resolution_clock::now();
		auto d = duration_cast<nanoseconds> (finish - start).count();
		cout <<"total time "<< d << " [ns]" << " \n";

		// imprimir(C);

		i++;
	}

	return 0;
}