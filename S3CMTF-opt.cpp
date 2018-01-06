/*
* @file        S3CMTF-opt.cpp
* @author      Dongjin Choi (skywalker5@snu.ac.kr), Seoul National University
* @author      Jun-Gi Jang (elnino4@snu.ac.kr), Seoul National University
* @author      U Kang (ukang@snu.ac.kr), Seoul National University
* @version     1.0
* @date        2017-06-22
*
* S3CMTF: Fast, Accurate, and Scalable Method for Coupled Matrix-Tensor Factorization
*
* This software is free of charge under research purposes.
* For commercial purposes, please contact the author.
*
* Usage:
*   - make opt
*/

/////    Header files     /////

#include <stdio.h>
#include <algorithm>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <vector>
#include <armadillo>
#include <omp.h>
#define ARMA_USE_LAPACK
#define ARMA_USE_BLAS

using namespace std;
using namespace arma;

///////////////////////////////


/////////      Pre-defined values      ///////////

#define MAX_ORDER 4							//The max order/way of input tensor
#define MAX_INPUT_DIMENSIONALITY 15000     //The max dimensionality/mode length of input tensor
#define MAX_CORE_TENSOR_DIMENSIONALITY 30	//The max dimensionality/mode length of core tensor
#define MAX_ENTRY 250000000						//The max number of entries in input tensor
#define MAX_CORE_SIZE 27000					//The max number of entries in core tensor
#define MAX_ITER 2000						//The maximum iteration number

/////////////////////////////////////////////////


/////////      Variables           ///////////

int threadsNum, order, dimensionality[MAX_ORDER], coreSize[MAX_ORDER], trainIndex[MAX_ENTRY][MAX_ORDER], testIndex[MAX_ENTRY][MAX_ORDER], trainEntryNum, testEntryNum, coreNum = 1, coreIndex[MAX_CORE_SIZE][MAX_ORDER], coupleDim[MAX_ORDER], isGraph[MAX_ORDER], iterNum, nanFlag = 0, nanCount = 0;
int i, j, k, l, aa, bb, ee, ff, gg, hh, ii, jj, kk, ll;
int indexPermute[MAX_ORDER*MAX_ENTRY];
double trainEntries[MAX_ENTRY], testEntries[MAX_ENTRY], sTime, trainRMSE, testRMSE, prevTrainRMSE = -1, minv = 2147483647, maxv = -2147483647, cminv = 2147483647, cmaxv = -2147483647;
double facMat[MAX_ORDER][MAX_INPUT_DIMENSIONALITY][MAX_CORE_TENSOR_DIMENSIONALITY], coreEntries[MAX_CORE_SIZE];

int numCoupledMat;
int coupleEntryNum[MAX_ORDER+1];
int entryNumCum[MAX_ORDER];
int totalN = 0;
int coupleDimensionality[MAX_ORDER];
int coupleMatIndex[MAX_ORDER][MAX_ENTRY][3];
double lambdaCouple, lambdaGraph;
double coupledEntries[MAX_ORDER][MAX_ENTRY];
double coupleFacMat[MAX_ORDER][MAX_INPUT_DIMENSIONALITY][MAX_CORE_TENSOR_DIMENSIONALITY];
int coupleWhere[MAX_ORDER][2][MAX_INPUT_DIMENSIONALITY];

double errorForTrain[MAX_ENTRY], errorForTest[MAX_ENTRY], trainNorm, error;
vector<int> trainWhere[MAX_ORDER][MAX_INPUT_DIMENSIONALITY], testWhere[MAX_ORDER][MAX_CORE_TENSOR_DIMENSIONALITY];
double lambdaReg;
double initialLearnRate;
double learnRate;
double tempCore[MAX_CORE_SIZE];
int Mul[MAX_ORDER], tempPermu[MAX_ORDER], rowcount;
double timeHistory[MAX_ITER], trainRmseHistory[MAX_ITER], testRmseHistory[MAX_ITER];
int normalization;
int nonnegativity, nonnegFlag, orthogonality, isInputPath;
double alpha;
int iter = 0;

/////////////////////////////////////////////////

char* ConfigPath;
char* TrainPath;
char* TestPath;
char CoupledPath[MAX_ORDER][100];
char* ResultPath;
char InputPath[100];

/////////////////////////////////////////////////

double frand(double x, double y) {//return the random value in (x,y) interval
	return ((y - x)*((double)rand() / RAND_MAX)) + x;
}

void Getting_Input() {
	FILE *fin = fopen(TrainPath, "r");
	FILE *ftest = fopen(TestPath, "r");
	FILE *fcouple;
	FILE *config = fopen(ConfigPath, "r");
	//INPUT
	double Timee = clock();
	printf("Reading input\n");
	fscanf(config, "%d", &order);
	for (i = 1; i <= order; i++) {
		fscanf(config, "%d", &dimensionality[i]);
	}
	for (i = 1; i <= order; i++) {
		fscanf(config, "%d", &coreSize[i]);
		coreNum *= coreSize[i];
	}
	fscanf(config, "%d", &threadsNum);
	omp_set_num_threads(threadsNum);

	fscanf(config, "%d", &trainEntryNum);
	fscanf(config, "%d", &testEntryNum);
	totalN += trainEntryNum;

	for (i = 1; i <= numCoupledMat; i++) {
		char a[100];
		fscanf(config, "%d", &coupleDim[i]);
		fscanf(config, "%s", &CoupledPath[i]);
		fscanf(config, "%d", &coupleDimensionality[i]);
		fscanf(config, "%d", &coupleEntryNum[i]);
		fscanf(config, "%d", &isGraph[i]);
		entryNumCum[i] = totalN;
		totalN += coupleEntryNum[i];
	}

	if (isInputPath) {
		fscanf(config, "%s", &InputPath);
	}

	entryNumCum[numCoupledMat + 1] = totalN;

	for (i = 1; i <= numCoupledMat; i++) {
		for (k = 1; k <= coupleDimensionality[i]; k++) {
			coupleWhere[i][1][k] = 0;
		}
	}

	for (i = 1; i <= numCoupledMat; i++) {
		fcouple = fopen(CoupledPath[i], "r");
		for (j = 1; j <= coupleEntryNum[i]; j++) {
			fscanf(fcouple, "%d", &k);
			coupleMatIndex[i][j][1] = k;

			fscanf(fcouple, "%d", &k);
			coupleMatIndex[i][j][2] = k;
			coupleWhere[i][1][k]++;
			fscanf(fcouple, "%lf", &coupledEntries[i][j]);

			if (cminv > coupledEntries[i][j]) cminv = coupledEntries[i][j];
			if (cmaxv < coupledEntries[i][j]) cmaxv = coupledEntries[i][j];
		}
	}

	for (i = 1; i <= trainEntryNum; i++) {
		for (j = 1; j <= order; j++) {
			fscanf(fin, "%d", &k);
			trainIndex[i][j] = k;
			trainWhere[j][k].push_back(i);
		}
		fscanf(fin, "%lf", &trainEntries[i]);
		trainNorm += trainEntries[i] * trainEntries[i];
		if (minv > trainEntries[i]) minv = trainEntries[i];
		if (maxv < trainEntries[i]) maxv = trainEntries[i];
	}
	trainNorm = sqrt(trainNorm);

	for (i = 1; i <= testEntryNum; i++) {
		for (j = 1; j <= order; j++) {
			fscanf(ftest, "%d", &k);
			testIndex[i][j] = k;
		}
		fscanf(ftest, "%lf", &testEntries[i]);
	}
	printf("Elapsed Time:\t%lf\n", (clock() - Timee) / CLOCKS_PER_SEC);
	printf("Reading Done.\nNorm : %lf\nInitialize\n", trainNorm);
}
void Initialize() {	//INITIALIZE
	if (isInputPath == 0) {
		double Timee = clock();
		iter = 0;
		double initVal = pow((maxv / coreNum), (1 / double(order + 1)));
		if (nonnegativity) nonnegFlag = 1;

		for (i = 1; i <= numCoupledMat; i++) {
			if (!isGraph[i]) {
				int cplDim = coupleDim[i];
				double cinitVal = cmaxv / (coreSize[cplDim] * initVal);
				for (j = 1; j <= coupleDimensionality[i]; j++) {
					for (k = 1; k <= coreSize[cplDim]; k++) {
						coupleFacMat[i][j][k] = frand(cinitVal, cinitVal * 2);
					}
				}
			}
		}
		for (i = 1; i <= order; i++) {
			for (j = 1; j <= dimensionality[i]; j++) {
				for (k = 1; k <= coreSize[i]; k++) {
					facMat[i][j][k] = frand(initVal / 2, initVal);
				}
			}
		}
		for (i = 1; i <= coreNum; i++) {
			coreEntries[i] = frand(initVal / 2, initVal);

			for (j = 1; j <= order; j++) {
				coreIndex[i][j] = coreIndex[i - 1][j];
			}
			coreIndex[i][order]++;  k = order;
			while (coreIndex[i][k] > coreSize[k]) {
				coreIndex[i][k] -= coreSize[k];
				coreIndex[i][k - 1]++; k--;
			}
			if (i == 1) {
				for (j = 1; j <= order; j++) coreIndex[i][j] = 1;
			}

			for (j = 1; j <= order; j++) {
				if (nanCount == 1) {
					testWhere[j][coreIndex[i][j]].push_back(i);
				}
			}
		}
		printf("Elapsed Time:\t%lf\n", (clock() - Timee) / CLOCKS_PER_SEC);
	}
	else {
		double Timee = clock();
		iter = 0;
		if (nonnegativity) nonnegFlag = 1;

		char temp[50];
		for (int iii = 1; iii <= order; iii++) {
			sprintf(temp, "%s/FACTOR%d", InputPath, iii);
			FILE *fin = fopen(temp, "r");
			for (int jjj = 1; jjj <= dimensionality[iii]; jjj++) {
				for (int kkk = 1; kkk <= coreSize[iii]; kkk++) {
					fscanf(fin, "%lf", &facMat[iii][jjj][kkk]);
				}
			}
		}


		for (i = 1; i <= numCoupledMat; i++) {
			if (!isGraph[i]) {
				sprintf(temp, "%s/CFACTOR%d", InputPath, i);
				FILE *fcin = fopen(temp, "r");
				for (j = 1; j <= coupleDimensionality[i]; j++) {
					for (k = 1; k <= coreSize[coupleDim[i]]; k++) {
						fscanf(fcin, "%lf", &coupleFacMat[i][j][k]);
					}
				}
			}
		}

		sprintf(temp, "%s/CORETENSOR", InputPath);
		FILE *fcore = fopen(temp, "r");
		for (i = 1; i <= coreNum; i++) {
			for (j = 1; j <= order; j++) {
				fscanf(fcore, "%d", &coreIndex[i][j]);
			}
			fscanf(fcore, "%lf", &coreEntries[i]);
		}
		for (j = 1; j <= order; j++) {
			if (nanCount == 1) {
				testWhere[j][coreIndex[i][j]].push_back(i);
			}
		}
		printf("Elapsed Time:\t%lf\n", (clock() - Timee) / CLOCKS_PER_SEC);
	}
}
double abss(double x) {
	return x > 0 ? x : -x;
}

//[Input] Input tensor X, initialized core tensor G, and factor matrices A^{(n)} (n=1...N)  
//[Output] Updated factor matrices A^{(n)} (n=1...N)
//[Function] Update all factor matrices according to the differential equation.
void Update_Factor_Matrices() {
	int i, temp;
	//Generate random permutation
	for (i = totalN; i >= 1; --i) {
		indexPermute[i] = i;
	}
	for (i = totalN; i >= 1; --i) {
		j = (rand() % i) + 1;
		temp = indexPermute[i];
		indexPermute[i] = indexPermute[j];
		indexPermute[j] = temp;
	}
#pragma omp parallel for schedule(static)
	for (i = 1; i <= totalN; i++)
	{
		if (1 <= indexPermute[i] && indexPermute[i] <= trainEntryNum) {
			int current_input_entry = indexPermute[i];
			double currentVal = trainEntries[current_input_entry];

			double current_estimation = 0;
			double CoreProducts[MAX_CORE_SIZE];
			int ii;
			for (ii = 1; ii <= coreNum; ii++) {
				double temp = coreEntries[ii];
				int jj;
				for (jj = 1; jj <= order; jj++) {
					temp *= facMat[jj][trainIndex[current_input_entry][jj]][coreIndex[ii][jj]];
				}
				CoreProducts[ii] = temp;
				current_estimation += temp;
			}
			double Sigma[MAX_CORE_TENSOR_DIMENSIONALITY];
			//Updating Factor matrices
			int jjj;
			for (jjj = 1; jjj <= order; jjj++) {//ith Factor Matrix
				int l;
				int column_size = coreSize[jjj];
				double temp2;
				for (l = 1; l <= column_size; l++) {
					int core_nonzeros = testWhere[jjj][l].size();
					int k;
					Sigma[l] = 0;
					if (abss(facMat[jjj][trainIndex[current_input_entry][jjj]][l]) < 0.00000001) {
						facMat[jjj][trainIndex[current_input_entry][jjj]][l] = 0.0000001;
						continue;
					}
					for (k = 0; k < core_nonzeros; k++) {
						int current_core_entry = testWhere[jjj][l][k];
						Sigma[l] += CoreProducts[current_core_entry];
					}
					Sigma[l] /= facMat[jjj][trainIndex[current_input_entry][jjj]][l];
				}

				if (normalization == 2) {
					for (k = 1; k <= column_size; k++) {
						int II = trainIndex[current_input_entry][jjj];
						facMat[jjj][II][k] = facMat[jjj][II][k]
							- learnRate*(lambdaReg / (double)(trainWhere[jjj][II].size())*facMat[jjj][II][k]
								- (currentVal - current_estimation)*Sigma[k]);
						if (nonnegativity) {
							if (facMat[jjj][II][k] <= 0 && (iter>10 || nonnegFlag == 0)) {
								facMat[jjj][II][k] /= 2;
							}
						}
					}
				}
				else {
					for (k = 1; k <= column_size; k++) {
						int II = trainIndex[current_input_entry][jjj];
						facMat[jjj][II][k] = facMat[jjj][II][k]
							- learnRate*(lambdaReg / (double)(trainWhere[jjj][II].size())*facMat[jjj][II][k] / abss(facMat[jjj][II][k])
								- (currentVal - current_estimation)*Sigma[k]);
						if (nonnegativity) {
							if (facMat[jjj][II][k] <= 0 && (iter>10 || nonnegFlag == 0)) {
								facMat[jjj][II][k] /= 2;
							}
						}
					}
				}

			}
			//Update_Core_Tensor

			if (i%threadsNum == 0) {
				int kk;
				if (normalization == 2) {
					for (kk = 1; kk <= coreNum; kk++) {
						double temp2;
						if (abss(coreEntries[kk]) < 0.00000001) {
							coreEntries[kk] = 0.0000001;
						}
						temp2 = CoreProducts[kk] / coreEntries[kk];
						coreEntries[kk] = coreEntries[kk] + learnRate*(currentVal - current_estimation)*temp2 - learnRate*lambdaReg*coreEntries[kk] / trainEntryNum;

						if (nonnegativity) {
							if (coreEntries[kk] <= 0 && (iter>10 || nonnegFlag == 0)) {
								coreEntries[kk] /= 2;
							}
						}
					}
				}
				else {
					for (kk = 1; kk <= coreNum; kk++) {
						double temp2;
						if (abss(coreEntries[kk]) < 0.00000001) {
							coreEntries[kk] = 0.0000001;
						}
						temp2 = CoreProducts[kk] / coreEntries[kk];
						coreEntries[kk] = coreEntries[kk] + learnRate*(currentVal - current_estimation)*temp2 - learnRate*lambdaReg*coreEntries[kk] / (trainEntryNum*abss(coreEntries[kk]));
						if (nonnegativity) {
							if (coreEntries[kk] <= 0 && (iter>10 || nonnegFlag == 0)) {
								coreEntries[kk] /= 2;
							}
						}
					}
				}
			}

		}
		else {
			int coupleMode;
			int current_input_entry;
			int ii;
			int cplNum;
			for (ii = 1; ii <= numCoupledMat; ii++) {
				if (indexPermute[i]>entryNumCum[ii + 1]) {
					continue;
				}
				coupleMode = coupleDim[ii];
				cplNum = ii;
				current_input_entry = indexPermute[i] - entryNumCum[ii];
				break;
			}
			double currentCoupledVal = coupledEntries[cplNum][current_input_entry];
			if (!isGraph[cplNum]) {
				double coupledM_estimation = 0;
				int column_size = coreSize[coupleMode];
				for (ii = 1; ii <= column_size; ii++) {
					coupledM_estimation += facMat[coupleMode][coupleMatIndex[cplNum][current_input_entry][1]][ii]
						* coupleFacMat[cplNum][coupleMatIndex[cplNum][current_input_entry][2]][ii];
				}

				//Updating Factor matrix vector

				double Sigma[MAX_CORE_TENSOR_DIMENSIONALITY];
				int l;
				for (l = 1; l <= column_size; l++) {
					Sigma[l] = -lambdaCouple*(currentCoupledVal - coupledM_estimation)
						*coupleFacMat[cplNum][coupleMatIndex[cplNum][current_input_entry][2]][l];
				}
				for (l = 1; l <= column_size; l++) {
					facMat[coupleMode][coupleMatIndex[cplNum][current_input_entry][1]][l] -=
						learnRate*Sigma[l];
					if (nonnegativity) {
						if (facMat[coupleMode][coupleMatIndex[cplNum][current_input_entry][1]][l] <= 0 && (iter > 10 || nonnegFlag == 0)) {
							facMat[coupleMode][coupleMatIndex[cplNum][current_input_entry][1]][l] /= 2;
						}
					}
				}
				//Updating Coupled Factor matrix vector
				if (normalization == 2) {
					for (l = 1; l <= column_size; l++) {
						Sigma[l] = -lambdaCouple*(currentCoupledVal - coupledM_estimation)
							*facMat[coupleMode][coupleMatIndex[cplNum][current_input_entry][1]][l]
							+ lambdaReg*lambdaCouple / (double)coupleWhere[cplNum][1][coupleMatIndex[cplNum][current_input_entry][2]] * coupleFacMat[cplNum][coupleMatIndex[cplNum][current_input_entry][2]][l];
					}
				}
				else {
					for (l = 1; l <= column_size; l++) {
						Sigma[l] = -lambdaCouple*(currentCoupledVal - coupledM_estimation)
							*facMat[coupleMode][coupleMatIndex[cplNum][current_input_entry][1]][l]
							+ lambdaReg*lambdaCouple / (double)coupleWhere[cplNum][1][coupleMatIndex[cplNum][current_input_entry][2]] * coupleFacMat[cplNum][coupleMatIndex[cplNum][current_input_entry][2]][l] / abss(coupleFacMat[cplNum][coupleMatIndex[cplNum][current_input_entry][2]][l]);
					}
				}
				for (l = 1; l <= column_size; l++) {
					coupleFacMat[cplNum][coupleMatIndex[cplNum][current_input_entry][2]][l] -=
						learnRate*Sigma[l];
					if (nonnegativity) {
						if (coupleFacMat[cplNum][coupleMatIndex[cplNum][current_input_entry][2]][l] <= 0 && (iter > 10 || nonnegFlag == 0)) {
							coupleFacMat[cplNum][coupleMatIndex[cplNum][current_input_entry][2]][l] /= 2;
						}
					}
				}
			}
			else {
				int column_size = coreSize[coupleMode];

				//Updating Factor matrix vector

				double Sigma[MAX_CORE_TENSOR_DIMENSIONALITY];
				int l;
				for (l = 1; l <= column_size; l++) {
					Sigma[l] = lambdaGraph* currentCoupledVal*(
						facMat[coupleMode][coupleMatIndex[cplNum][current_input_entry][1]][l]
						- facMat[coupleMode][coupleMatIndex[cplNum][current_input_entry][2]][l]);
				}
				for (l = 1; l <= column_size; l++) {
					facMat[coupleMode][coupleMatIndex[cplNum][current_input_entry][1]][l] -=
						learnRate*Sigma[l];
					if (nonnegativity) {
						if (facMat[coupleMode][coupleMatIndex[cplNum][current_input_entry][1]][l] <= 0 && (iter > 10 || nonnegFlag == 0)) {
							facMat[coupleMode][coupleMatIndex[cplNum][current_input_entry][1]][l] /= 2;
						}
					}
				}
				for (l = 1; l <= column_size; l++) {
					facMat[coupleMode][coupleMatIndex[cplNum][current_input_entry][2]][l] +=
						learnRate*Sigma[l];
					if (nonnegativity) {
						if (facMat[coupleMode][coupleMatIndex[cplNum][current_input_entry][2]][l] <= 0 && (iter > 10 || nonnegFlag == 0)) {
							facMat[coupleMode][coupleMatIndex[cplNum][current_input_entry][2]][l] /= 2;
						}
					}
				}
			}
		}
	}

}

//[Input] The index of observable entry X
//[Output] Reconstructed value of observable entry X
//[Function] Getting reconstructed value by multiplying core tensor and factor matrices

//[Input] Input tensor X, core tensor G, and factor matrices A^{(n)} (n=1...N)
//[Output] trainRMSE = 1-||X-X'||/||X|| (Reconstruction error = ||X-X'||)
//[Function] Calculating fit and reconstruction error in parallel.
void Reconstruction() {
	error = 0;
#pragma omp parallel for 
	for (i = 1; i <= trainEntryNum; i++) {
		errorForTrain[i] = trainEntries[i];
	}

#pragma omp parallel for 
	for (i = 1; i <= trainEntryNum; i++) {
		int j;
		for (j = 1; j <= coreNum; j++) {
			double temp = coreEntries[j];
			int k;
			for (k = 1; k <= order; k++) {
				temp *= facMat[k][trainIndex[i][k]][coreIndex[j][k]];
			}
			errorForTrain[i] -= temp;
		}
	}

#pragma omp parallel for reduction(+:error)

	for (i = 1; i <= trainEntryNum; i++) {
		error += errorForTrain[i] * errorForTrain[i];
	}
	if (trainNorm == 0) trainRMSE = 1;
	else trainRMSE = sqrt(error) / sqrt(trainEntryNum);

	error = 0;

#pragma omp parallel for 
	for (i = 1; i <= testEntryNum; i++) {
		errorForTest[i] = testEntries[i];
	}

#pragma omp parallel for 
	for (i = 1; i <= testEntryNum; i++) {
		int j;
		double temp2 = 0;
		for (j = 1; j <= coreNum; j++) {
			double temp = coreEntries[j];
			int k;
			for (k = 1; k <= order; k++) {
				temp *= facMat[k][testIndex[i][k]][coreIndex[j][k]];
			}
			temp2 += temp;
		}
		if (temp2 > maxv) temp2 = maxv;
		else if (temp2 < minv) temp2 = minv;
		errorForTest[i] -= temp2;
	}

#pragma omp parallel for reduction(+:error)
	for (i = 1; i <= testEntryNum; i++) {
		error += errorForTest[i] * errorForTest[i];
	}
	testRMSE = sqrt(error) / sqrt(testEntryNum);

}

//[Input] Updated factor matrices A^{(n)} (n=1...N)
//[Output] Orthonormal factor matrices A^{(n)} (n=1...N) and updated core tensor G
//[Function] Orthogonalize all factor matrices and update core tensor simultaneously.
void Orthogonalize() {
	Mul[order] = 1;
	for (i = order - 1; i >= 1; i--) {
		Mul[i] = Mul[i + 1] * coreSize[i + 1];
	}
	for (i = 1; i <= order; i++) {
		mat Q, R;
		mat X = mat(dimensionality[i], coreSize[i]);
		for (k = 1; k <= dimensionality[i]; k++) {
			for (l = 1; l <= coreSize[i]; l++) {
				X(k - 1, l - 1) = facMat[i][k][l];
			}
		}
		qr_econ(Q, R, X);
		double coeff = 1;
		for (k = 1; k <= dimensionality[i]; k++) {
			for (l = 1; l <= coreSize[i]; l++) {
				facMat[i][k][l] = Q(k - 1, l - 1)*coeff;
			}
		}
		for (j = 1; j <= coreNum; j++) {
			tempCore[j] = 0;
		}
		for (j = 1; j <= coreNum; j++) {
			for (k = 1; k <= i - 1; k++) {
				tempPermu[k] = coreIndex[j][k];
			}
			for (k = i + 1; k <= order; k++) {
				tempPermu[k] = coreIndex[j][k];
			}
			for (k = 1; k <= coreSize[i]; k++) {
				tempPermu[i] = k;
				int cur = j + (k - coreIndex[j][i])*Mul[i];
				tempCore[cur] += coreEntries[j] * (R(k - 1, coreIndex[j][i] - 1) / coeff);
			}
		}
		for (j = 1; j <= coreNum; j++) {
			coreEntries[j] = tempCore[j];
		}

		for (j = 1; j <= numCoupledMat; j++) {
			if (i == coupleDim[j]) {
				mat Y = mat(coupleDimensionality[j], coreSize[i]);
				for (k = 1; k <= coupleDimensionality[j]; k++) {
					for (l = 1; l <= coreSize[i]; l++) {
						Y(k - 1, l - 1) = coupleFacMat[j][k][l];
					}
				}
				Y = Y*(R.t());
				for (k = 1; k <= coupleDimensionality[j]; k++) {
					for (l = 1; l <= coreSize[i]; l++) {
						coupleFacMat[j][k][l] = Y(k - 1, l - 1);
					}
				}
			}
		}
	}
}

//[Input] Input tensor and initialized core tensor and factor matrices
//[Output] Updated core tensor and factor matrices
//[Function] Performing main algorithm which updates core tensor and factor matrices iteratively
void CMTF() {
	printf("CMTF START\n");

	double sTime = omp_get_wtime();
	double avertime = 0;
	learnRate = initialLearnRate;
	while (1) {

		double itertime = omp_get_wtime(), steptime;
		steptime = itertime;

		Update_Factor_Matrices();
		printf("Factor Time : %lf\n", omp_get_wtime() - steptime);
		steptime = omp_get_wtime();

		Reconstruction();
		printf("Recon Time : %lf\n", omp_get_wtime() - steptime);
		steptime = omp_get_wtime();

		avertime += omp_get_wtime() - itertime;
		printf("iter%d :      Train Rmse : %lf\tTest Rmse : %lf\tElapsed Time : %lf\n", ++iter, trainRMSE, testRMSE, omp_get_wtime() - itertime);
		if (iter == 12 && nonnegativity == 1 && nonnegFlag == 1) {
			iter = 1; nonnegFlag = 0;
		}
		learnRate = initialLearnRate / (1+alpha*iter);//(1 + initialLearnRate * 100 * iter);
		timeHistory[iter - 1] = omp_get_wtime() - itertime;
		trainRmseHistory[iter - 1] = trainRMSE;
		testRmseHistory[iter - 1] = testRMSE;
		if (trainRMSE != trainRMSE) {
			nanFlag = 1;
			break;
		}
		if (iter == iterNum) break;
		prevTrainRMSE = trainRMSE;
	}

	avertime /= iter;

	printf("\niterNum ended.\ttrainRMSE : %lf\tAverage iteration time : %lf\n", trainRMSE, avertime);

	printf("\nOrthogonalize and updating core tensor...\n\n");

	if (normalization == 2 && nonnegativity == 0 && orthogonality) {
		Orthogonalize();
	}

	Reconstruction();

	printf("\nTotal update ended.\tFinal Rmse : %lf\tTotal Elapsed time: %lf\n", trainRMSE, omp_get_wtime() - sTime);
}

//[Input] Updated core tensor G and factor matrices A^{(n)} (n=1...N)
//[Output] core tensor G in sparse tensor format and factor matrices A^{(n)} (n=1...N) in full-dense matrix format
//[Function] Writing all factor matrices and core tensor in result path
void Print() {
	printf("\nWriting factor matrices and core tensor to file...\n");
	char temp[50];
	sprintf(temp, "mkdir %s", ResultPath);
	system(temp);
	for (i = 1; i <= order; i++) {
		sprintf(temp, "%s/FACTOR%d", ResultPath, i);
		FILE *fin = fopen(temp, "w");
		for (j = 1; j <= dimensionality[i]; j++) {
			for (k = 1; k <= coreSize[i]; k++) {
				fprintf(fin, "%f\t", facMat[i][j][k]);
			}
			fprintf(fin, "\n");
		}
	}
	for (i = 1; i <= numCoupledMat; i++) {
		if (!isGraph[i]) {
			sprintf(temp, "%s/CFACTOR%d", ResultPath, i);
			FILE *fcin = fopen(temp, "w");
			for (j = 1; j <= coupleDimensionality[i]; j++) {
				for (k = 1; k <= coreSize[coupleDim[i]]; k++) {
					fprintf(fcin, "%f\t", coupleFacMat[i][j][k]);
				}
				fprintf(fcin, "\n");
			}
		}
	}
	sprintf(temp, "%s/CORETENSOR", ResultPath);
	FILE *fcore = fopen(temp, "w");
	for (i = 1; i <= coreNum; i++) {
		for (j = 1; j <= order; j++) {
			fprintf(fcore, "%d\t", coreIndex[i][j]);
		}
		fprintf(fcore, "%f\n", coreEntries[i]);
	}
}

void PrintTime() {
	printf("\nWriting Time and error to file...\n");
	char temp[50];
	sprintf(temp, "mkdir %s", ResultPath);
	system(temp);
	sprintf(temp, "%s/TIMEERROR", ResultPath);
	FILE *ftime = fopen(temp, "w");
	for (i = 0; i < iter; i++) {
		fprintf(ftime, "%f\t%f\t%f\n", timeHistory[i], trainRmseHistory[i], testRmseHistory[i]);
	}
}

//[Input] Path of configuration file, input tensor file, and result directory
//[Output] Core tensor G and factor matrices A^{(n)} (n=1...N)
//[Function] Performing P-Tucker to factorize partially observable tensor
int main(int argc, char* argv[]) {
	if (argc == 16) {
		initialLearnRate = atof(argv[6]);
		lambdaReg = atof(argv[7]);
		lambdaCouple = atof(argv[8]);
		iterNum = atoi(argv[9]);
		alpha = atof(argv[10]);
		normalization = atoi(argv[11]);
		if (normalization != 1 && normalization != 2) {
			printf("please input proper arguments\n");
			return 0;
		}
		nonnegativity = atoi(argv[12]);
		if (nonnegativity != 1 && nonnegativity != 0) {
			printf("please input proper arguments\n");
			return 0;
		}
		lambdaGraph = atof(argv[13]);
		orthogonality = atoi(argv[14]);
		isInputPath = atoi(argv[15]);
	}
	else if (argc == 15) {
		initialLearnRate = atof(argv[6]);
		lambdaReg = atof(argv[7]);
		lambdaCouple = atof(argv[8]);
		iterNum = atoi(argv[9]);
		alpha = atof(argv[10]);
		normalization = atoi(argv[11]);
		if (normalization != 1 && normalization != 2) {
			printf("please input proper arguments\n");
			return 0;
		}
		nonnegativity = atoi(argv[12]);
		if (nonnegativity != 1 && nonnegativity != 0) {
			printf("please input proper arguments\n");
			return 0;
		}
		lambdaGraph = atof(argv[13]);
		orthogonality = atoi(argv[14]);
		isInputPath = 0;
	}
	else if (argc == 14) {
		initialLearnRate = atof(argv[6]);
		lambdaReg = atof(argv[7]);
		lambdaCouple = atof(argv[8]);
		iterNum = atoi(argv[9]);
		alpha = atof(argv[10]);
		normalization = atoi(argv[11]);
		if (normalization != 1 && normalization != 2) {
			printf("please input proper arguments\n");
			return 0;
		}
		nonnegativity = atoi(argv[12]);
		if (nonnegativity != 1 && nonnegativity != 0) {
			printf("please input proper arguments\n");
			return 0;
		}
		lambdaGraph = atof(argv[13]);
		orthogonality = 0;
		isInputPath = 0;
	}
	else if (argc == 13) {
		initialLearnRate = atof(argv[6]);
		lambdaReg = atof(argv[7]);
		lambdaCouple = atof(argv[8]);
		iterNum = atoi(argv[9]);
		alpha = atof(argv[10]);
		normalization = atoi(argv[11]);
		if (normalization != 1 && normalization != 2) {
			printf("please input proper arguments\n");
			return 0;
		}
		nonnegativity = atoi(argv[12]);
		if (nonnegativity != 1 && nonnegativity != 0) {
			printf("please input proper arguments\n");
			return 0;
		}
		lambdaGraph = 0.1;
		orthogonality = 0;
		isInputPath = 0;
	}
	else if (argc == 12) {
		initialLearnRate = atof(argv[6]);
		lambdaReg = atof(argv[7]);
		lambdaCouple = atof(argv[8]);
		iterNum = atoi(argv[9]);
		alpha = atof(argv[10]);
		normalization = atoi(argv[11]);
		if (normalization != 1 && normalization != 2) {
			printf("please input proper arguments\n");
			return 0;
		}
		nonnegativity = 0;
		lambdaGraph = 0.1;
		orthogonality = 0;
		isInputPath = 0;
	}
	else if (argc == 11) {
		initialLearnRate = atof(argv[6]);
		lambdaReg = atof(argv[7]);
		lambdaCouple = atof(argv[8]);
		iterNum = atoi(argv[9]);
		alpha = atof(argv[10]);
		normalization = 2;
		nonnegativity = 0;
		lambdaGraph = 0.1;
		orthogonality = 0;
		isInputPath = 0;
	}
	else if (argc == 10) {
		initialLearnRate = atof(argv[6]);
		lambdaReg = atof(argv[7]);
		lambdaCouple = atof(argv[8]);
		iterNum = atoi(argv[9]);
		alpha = 0.1;
		normalization = 2;
		nonnegativity = 0;
		lambdaGraph = 0.1;
		orthogonality = 0;
		isInputPath = 0;
	}
	else if (argc == 9) {
		initialLearnRate = atof(argv[6]);
		lambdaReg = atof(argv[7]);
		lambdaCouple = atof(argv[8]);
		iterNum = 100;
		alpha = 0.1;
		normalization = 2;
		nonnegativity = 0;
		lambdaGraph = 0.1;
		orthogonality = 0;
		isInputPath = 0;
	}
	else if (argc == 8) {
		initialLearnRate = atof(argv[6]);
		lambdaReg = atof(argv[7]);
		lambdaCouple = 0.1;
		iterNum = 100;
		alpha = 0.1;
		normalization = 2;
		nonnegativity = 0;
		lambdaGraph = 0.1;
		orthogonality = 0;
		isInputPath = 0;
	}
	else if (argc == 7) {
		initialLearnRate = atof(argv[6]);
		lambdaReg = 0.1;
		lambdaCouple = 0.1;
		iterNum = 100;
		alpha = 0.1;
		normalization = 2;
		nonnegativity = 0;
		lambdaGraph = 0.1;
		orthogonality = 0;
		isInputPath = 0;
	}
	else if (argc == 6) {
		initialLearnRate = 0.001;
		lambdaReg = 0.1;
		lambdaCouple = 0.1;
		iterNum = 100;
		alpha = 0.1;
		normalization = 2;
		nonnegativity = 0;
		lambdaGraph = 0.1;
		orthogonality = 0;
		isInputPath = 0;
	}

	else {
		printf("please input proper arguments\n");
		return 0;
	}

	ConfigPath = argv[1];
	TrainPath = argv[2];
	TestPath = argv[3];
	ResultPath = argv[4];
	numCoupledMat = atoi(argv[5]);

	srand((unsigned)time(NULL));

	sTime = clock();

	Getting_Input();

	do {
		nanFlag = 0;
		nanCount++;

		Initialize();

		CMTF();
	} while (nanFlag && nanCount<10);

	Print();

	PrintTime();

	return 0;
}
