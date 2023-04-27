#include <bits/stdc++.h>
#include <omp.h>
using namespace std;
using namespace chrono;



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


	auto start = high_resolution_clock::now();

	// Multiplicacion tradicional secuencial
	vector<vector<int>> C(n, vector<int>(n));
	for(int i=0; i<n; i++){
		for(int j=0; j<n; j++){
			for(int k=0; k<n; k++){
				C[i][j] += A[i][k]*B[k][j];
			}
		}
	}
	auto finish = high_resolution_clock::now();
	auto d = duration_cast<nanoseconds> (finish - start).count();
	cout <<"total time "<< d << " [ns]" << " \n";

	cout << "C" << endl;
	for(int i=0; i<n; i++){
		for(int j=0; j<n; j++){
			cout << C[i][j] << " ";
		}
		cout << endl;
	}



	// Multiplicacion tradicional paralela
	for(auto fila: C)
		for(int i: fila) i = 0;

	start = high_resolution_clock::now();

	for(int i=0; i<n; i++){
		#pragma omp parallel
		{
			for(int j=omp_get_thread_num(); j<n; j += omp_get_num_threads()){
				for(int k=0; k<n; k++){
					C[i][j] += A[i][k]*B[k][j];
				}
			}
		}
		#pragma omp barrier
	}

	

	finish = high_resolution_clock::now();
	d = duration_cast<nanoseconds> (finish - start).count();
	cout <<"total time "<< d << " [ns]" << " \n";

	cout << "C" << endl;
	for(int i=0; i<n; i++){
		for(int j=0; j<n; j++){
			cout << C[i][j] << " ";
		}
		cout << endl;
	}


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