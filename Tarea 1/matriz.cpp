#include <bits/stdc++.h>
#include <omp.h>
using namespace std;
using namespace chrono;

typedef vector<vector<int>> vii;

void imprimir(vii &M){
	cout << "Matriz" << endl;
	for(int i=0; i<M.size(); i++){
		for(int j=0; j<M.size(); j++){
			cout << M[i][j] << " ";
		}
		cout << endl;
	}
}


void mat_trad_sec(vii &A, vii &B, vii &C, int n){
	for(int i=0; i<n; i++){
		for(int j=0; j<n; j++){
			for(int k=0; k<n; k++) C[i][j] += A[i][k] * B[k][j];
		}
	}

	// imprimir(C);
}


void mat_trad_par(vii &A, vii &B, vii &C, int n){
	for(int i=0; i<n; i++){
		#pragma omp parallel for
			for(int j=0; j<n; j++){
				for(int k=0; k<n; k++) 
					C[i][j] += A[i][k] * B[k][j];
			}
	}

	// imprimir(C);
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


	vector<vector<int>> C(n, vector<int>(n));


	// Multiplicacion tradicional secuencial
	auto start = high_resolution_clock::now();

	mat_trad_sec(A, B, C, n);

	auto finish = high_resolution_clock::now();
	auto d = duration_cast<nanoseconds> (finish - start).count();
	cout <<"total time "<< d << " [ns]" << " \n";



	for(int i=0; i<n; i++) for(int j=0; j<n; j++) C[i][j] = 0;

	// Multiplicacion tradicional paralela
	start = high_resolution_clock::now();

	mat_trad_par(A, B, C, n);

	finish = high_resolution_clock::now();
	d = duration_cast<nanoseconds> (finish - start).count();
	cout <<"total time "<< d << " [ns]" << " \n";



	/*/ --------------- prints --------------------------
	cout << "A" << endl;
	for(int i=0; i<n; i++){
		for(int j=0; j<n; j++){
			cout << A[i][j] << " ";
		}
		cout << endl;
	}

	cout << "B" << endl;
	for(int i=0; i<n; i++){
		for(int j=0; j<n; j++){
			cout << B[i][j] << " ";
		}
		cout << endl;
	}		

	cout << "C" << endl;
	for(int i=0; i<n; i++){
		for(int j=0; j<n; j++){
			cout << C[i][j] << " ";
		}
		cout << endl;
	}	*/


	return 0;
}