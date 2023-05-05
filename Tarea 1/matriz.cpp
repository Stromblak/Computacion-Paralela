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
	cout << endl;
}


// 1. Matrices tradicionales
vii mat_trad_sec(vii &A, vii &B){
	int n = A.size();
	vector<vector<int>> C(n, vector<int>(n));

	for(int i=0; i<n; i++){
		for(int j=0; j<n; j++){
			for(int k=0; k<n; k++) C[i][j] += A[i][k] * B[k][j];
		}
	}

	return C;
}

vii mat_trad_par(vii &A, vii &B){
	int n = A.size();
	vector<vector<int>> C(n, vector<int>(n));

	for(int i=0; i<n; i++){
		#pragma omp parallel for
			for(int j=0; j<n; j++){
				for(int k=0; k<n; k++) 
					C[i][j] += A[i][k] * B[k][j];
			}
	}

	return C;
}


// 2. Matrices amigables con el cache
vii mat_amigable_sec(vii &A, vii &B){
	int n = A.size();
	vector<vector<int>> C(n, vector<int>(n));

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

	return C;
}

vii mat_amigable_par(vii &A, vii &B){
	int n = A.size();
	vector<vector<int>> C(n, vector<int>(n));

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

	return C;
}


// 4. Strassen - https://www.youtube.com/watch?v=OSelhO6Qnlc
vii suma(vii &M1, vii &M2){
	int n = M1.size();
	vii M(n, vector<int>(n));

	for(int i=0; i<n; i++){
		for(int j=0; j<n; j++){
			M[i][j] = M1[i][j] + M2[i][j];
		}
	}

	return M;
}

vii resta(vii &M1, vii &M2){
	int n = M1.size();
	vii M(n, vector<int>(n));

	for(int i=0; i<n; i++){
		for(int j=0; j<n; j++){
			M[i][j] = M1[i][j] - M2[i][j];
		}
	}

	return M;
}
 
vii multiplicacion_3a_sec(vii A, vii B){
	if(A.size() <= 2 << 5) return mat_amigable_sec(A, B);

	int n = A.size()/2;

	vii A11(n, vector<int>(n)), A12(n, vector<int>(n));
	vii A21(n, vector<int>(n)), A22(n, vector<int>(n));
	vii B11(n, vector<int>(n)), B12(n, vector<int>(n));
	vii B21(n, vector<int>(n)), B22(n, vector<int>(n));

	for(int i=0; i<n; i++){
		for(int j=0; j<n; j++){
			A11[i][j] = A[i][j];
			A12[i][j] = A[i][j+n];
			A21[i][j] = A[i+n][j];
			A22[i][j] = A[i+n][j+n];
			B11[i][j] = B[i][j];
			B12[i][j] = B[i][j+n];
			B21[i][j] = B[i+n][j];
			B22[i][j] = B[i+n][j+n];
		}
	}

	vii aux1 = multiplicacion_3a_sec(A11, B11);
	vii aux2 = multiplicacion_3a_sec(A12, B21);
	vii C1 = suma(aux1, aux2);
	
	aux1 = multiplicacion_3a_sec(A11, B12);
	aux2 = multiplicacion_3a_sec(A12, B22);	
	vii C2 = suma(aux1, aux2);

	aux1 = multiplicacion_3a_sec(A21, B11);
	aux2 = multiplicacion_3a_sec(A22, B21);	
	vii C3 = suma(aux1, aux2);

	aux1 = multiplicacion_3a_sec(A22, B12);
	aux2 = multiplicacion_3a_sec(A22, B22);	
	vii C4 = suma(aux1, aux2);


	vii C(n*2, vector<int>(n*2));
	for(int i=0; i<n; i++){
		for(int j=0; j<n; j++){
			C[i][j] = C1[i][j];
			C[i][j+n] = C2[i][j];
			C[i+n][j] = C3[i][j];
			C[i+n][j+n] = C4[i][j];
		}
	}

	return C;
}

vii strassen_sec(vii A, vii B){
	// limite recursion
	if(A.size() <= 2 << 5) return mat_amigable_sec(A, B);

	int n = A.size()/2;

	// Submatrices de A y B
	vii A11(n, vector<int>(n)), A12(n, vector<int>(n));
	vii A21(n, vector<int>(n)), A22(n, vector<int>(n));
	vii B11(n, vector<int>(n)), B12(n, vector<int>(n));
	vii B21(n, vector<int>(n)), B22(n, vector<int>(n));

	// Particionar las matrices A y B en 4 submatrices cada una
	for(int i=0; i<n; i++){
		for(int j=0; j<n; j++){
			A11[i][j] = A[i][j];
			A12[i][j] = A[i][j+n];
			A21[i][j] = A[i+n][j];
			A22[i][j] = A[i+n][j+n];
			B11[i][j] = B[i][j];
			B12[i][j] = B[i][j+n];
			B21[i][j] = B[i+n][j];
			B22[i][j] = B[i+n][j+n];
		}
	}

	vii M1 = strassen_sec( suma(A11, A22), suma(B11, B22) );
	vii M2 = strassen_sec( suma(A21, A22), B11 );
	vii M3 = strassen_sec( A11, resta(B12, B22) );
	vii M4 = strassen_sec( A22, resta(B21, B11) );
	vii M5 = strassen_sec( suma(A11, A12), B22 );
	vii M6 = strassen_sec( resta(A21, A11), suma(B11, B12) );
	vii M7 = strassen_sec( resta(A12, A22), suma(B21, B22) );

	vii C(n*2, vector<int>(n*2));
	for(int i=0; i<n; i++){
		for(int j=0; j<n; j++){
			C[i][j] = M1[i][j] + M4[i][j] - M5[i][j] + M7[i][j];
			C[i][j+n] = M3[i][j] + M5[i][j];
			C[i+n][j] = M2[i][j] + M4[i][j];
			C[i+n][j+n] = M1[i][j] - M2[i][j] + M3[i][j] + M6[i][j];
		}
	}

	return C;
}

// 3a paralelizado
vii sumap(vii &M1, vii &M2){
    int n = M1.size();
    vii M(n, vector<int>(n));

    #pragma omp parallel for
    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
            M[i][j] = M1[i][j] + M2[i][j];
        }
    }

    return M;
}

vii para3a(vii A, vii B){
	if(A.size() <= 2 << 5) return mat_amigable_sec(A, B);

	int n = A.size()/2;

	vii A11(n, vector<int>(n)), A12(n, vector<int>(n));
	vii A21(n, vector<int>(n)), A22(n, vector<int>(n));
	vii B11(n, vector<int>(n)), B12(n, vector<int>(n));
	vii B21(n, vector<int>(n)), B22(n, vector<int>(n));
	
	#pragma omp parallel for
	for(int i=0; i<n; i++){
		for(int j=0; j<n; j++){
			A11[i][j] = A[i][j];
			A12[i][j] = A[i][j+n];
			A21[i][j] = A[i+n][j];
			A22[i][j] = A[i+n][j+n];
			B11[i][j] = B[i][j];
			B12[i][j] = B[i][j+n];
			B21[i][j] = B[i+n][j];
			B22[i][j] = B[i+n][j+n];
		}
	}

	vii aux1, aux2, C1, C2, C3, C4;

	#pragma omp parallel sections 
	{
    #pragma omp section
    {
        aux1 = para3a(A11, B11);
        aux2 = para3a(A12, B21);
        C1 = sumap(aux1, aux2);
    }
    #pragma omp section
    {
        aux1 = para3a(A11, B12);
        aux2 = para3a(A12, B22);    
        C2 = sumap(aux1, aux2);
    }
    #pragma omp section
    {
        aux1 = para3a(A21, B11);
        aux2 = para3a(A22, B21);    
        C3 = sumap(aux1, aux2);
    }
    #pragma omp section
    {
        aux1 = para3a(A22, B12);
        aux2 = para3a(A22, B22);    
        C4 = sumap(aux1, aux2);
    }
	}

	vii C(n*2, vector<int>(n*2));
	#pragma omp parallel for
	for(int i=0; i<n; i++){
		for(int j=0; j<n; j++){
			C[i][j] = C1[i][j];
			C[i][j+n] = C2[i][j];
			C[i+n][j] = C3[i][j];
			C[i+n][j+n] = C4[i][j];
		}
	}

	return C;
}
// strassen paralelizado

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
		vii C;
		auto start = high_resolution_clock::now();

		if(i == 0) C = mat_trad_sec(A, B);
		else if(i == 1) C = mat_trad_par(A, B);
		else if(i == 2) C = mat_amigable_sec(A, B);
		else if(i == 3) C = mat_amigable_par(A, B);
		else if(i == 4) C = multiplicacion_3a_sec(A, B);
		else if(i == 5) C = strassen_sec(A, B);
		else if(i == 6) C = para3a(A, B);
		else break;

		auto finish = high_resolution_clock::now();
		auto d = duration_cast<nanoseconds> (finish - start).count();


		if(i == 0) 		cout << "Trad sec     "<< d << " [ns]" << endl;
		else if(i == 1) cout << "Trad par     "<< d << " [ns]" << endl;
		else if(i == 2) cout << "Amig sec     "<< d << " [ns]" << endl;
		else if(i == 3) cout << "Amig par     "<< d << " [ns]" << endl;
		else if(i == 4) cout << "3a par       "<< d << " [ns]" << endl;
		else if(i == 5) cout << "Strassen sec "<< d << " [ns]" << endl;
		else if(i == 6) cout << "3a par para  "<< d << " [ns]" << endl;

		if(n <= 4) imprimir(C);
		i++;
	}

	return 0;
}