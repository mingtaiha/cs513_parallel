#include <iostream>
#include <cmath>
#include <cstdlib>

using namespace std;




int * make_array_2_to_n(int n) {

	int * array = (int *) malloc((n-1) * sizeof(int));
	for (int i = 0; i < (n-1); i++) {
		array[i] = 1;
	}
	return array;
}

void print_array(int * arr, int n) {

	for (int i = 0; i < (n-1); i++) {
		cout << (i+2) << " " << arr[i] << endl;
	}

}

void print_prime(int * arr, int n) {

	for (int i = 0; i < (n-1); i++) {
		if (arr[i] == 1) {
			cout << (i+2) << endl;
		}
	}
}

void seq_sieve(int * arr, int n) {

	int sqrt_n = int(ceil(sqrt(float(n))));
	int i_sqr;	

	for (int i = 2; i <= sqrt_n; i++) {
		if (arr[i-2] == 1) {
			i_sqr = i * i;
			for (int j = i_sqr; j <= n; j+=i) {
				arr[j - 2] = 0;
			}
		}
	}
}



int main(int argc, char** argv) {

	if (argc != 2) {
		cout << "Takes one argument - n, positive integer - to calculate the number of primes at most n\n";
	}

	int n = atoi(argv[1]);

	int * seq_array = make_array_2_to_n(n);
	print_array(seq_array, n);
	seq_sieve(seq_array, n);
	print_prime(seq_array, n);

	return 0;

}
