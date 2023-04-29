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
void strassen(){

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


	int i = 0;
	while(1){
		vector<vector<int>> C(n, vector<int>(n));

		auto start = high_resolution_clock::now();

		if(i == 0) mat_trad_sec(A, B, C, n);
		else if(i == 1) mat_trad_par(A, B, C, n);
		else if(i == 2) mat_amigable_sec(A, B, C, n);
		else if(i == 3) mat_amigable_par(A, B, C, n);
		else break;

		auto finish = high_resolution_clock::now();
		auto d = duration_cast<nanoseconds> (finish - start).count();
		cout <<"total time "<< d << " [ns]" << " \n";

		/*
		imprimir(A);
		imprimir(B);
		imprimir(C);
		*/

		i++;
	}

	return 0;
}