#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define RANGE 10000

float * make_array(int size) {


	// Makes the array
	float * array = (float *) malloc(size * sizeof(float));
	
	srand(time(NULL));

	int i;
	for (i = 0; i < size; i++) {
		array[i] = rand() % RANGE;
	}

	return array;

}

float * prefix_sum_twice(float * array, int size) {

	int i;

	// Calculating first prefix sum
	for (i = 1; i < size; i++) {
		array[i] = array[i] + array[i - 1];
	}

	// Calculating second prefix sum
	for (i = 1; i < size; i++) {
		array[i] = array[i] + array[i - 1];
	}

	return array;
}

int main(int argc, char** argv) {

	int num_elem;
	if (argc != 2) {
		printf("Takes one argument - the number of elements in an array\n");
		return 0;
	}

	num_elem = atoi(argv[1]);	
	printf("num_elem: %d\n", num_elem);

	float * array = make_array(num_elem);
	printf("last element: %f\n", array[num_elem - 1]);

	
	clock_t start, end;
	int diff;
	start = clock();
	float * transformed_array = prefix_sum_twice(array, num_elem);
	end = clock();
	
	diff = (end - start) * 1000 / CLOCKS_PER_SEC;
	printf("Time taken to run sequential algorithm: %d msec\n", diff);

	return 0;

}
