include <stdio.h>
include <stdlib.h>

// you may add more methods here

int greatestSortedSubmatrix(int matrixSize, int matrix[100][100]) 
{
	int ret=0;
	vector<int> dummy[matrixSize*matrixSize][100];
	for(int i=0; i<matrixSize; i++)
	{
		for(int j=0; j<matrixSize; j++)
		{
			dummy[0].push_back[matrix
		}
	}
	return 0;
}

int main() {
	int N;
	int A[100][100];
	int i, j;
	scanf("%d",&N);
	for(i=0;i<N;i++)
		for(j=0;j<N;j++)
			scanf("%d",&A[i][j]);
	printf("%d\n",greatestSortedSubmatrix(N,A));
	return 0;
}
