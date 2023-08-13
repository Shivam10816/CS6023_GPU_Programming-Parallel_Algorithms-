
#include <bits/stdc++.h>
#include <cstdio> // Added for printf() function
#include <cuda.h>
#include <fstream>
#include <sys/time.h> // Added to get time of day
    using namespace std;

ofstream outfile; // The handle for printing the output

__global__ void per_row_AB_kernel(long int *A, long int *B, long int *C,
                                  long int m, long int n) {
                                
  long int ia = blockIdx.x;
  long int ib = threadIdx.x;
  
  for (long int ja = 0; ja < n; ja++) {
    for (long int jb = 0; jb < n; jb++) {
      long int i = ia * n + jb;
      long int j = ja * m + ib;
      C[i * m * n + j] = A[ia * n + ja] * B[ib * n + jb];
    }
  }
}

__global__ void per_column_AB_kernel(long int *A, long int *B, long int *C,long
int m, long int n){
    // --> Complete the kernel ....

    long int jb = threadIdx.x*blockDim.y + threadIdx.y;
    long int ja = blockIdx.x;
    if(jb<n){
        for(long int ia=0;ia<m;ia++){
          for(long int ib =0;ib<m;ib++){
            long int i = ia * n + jb;
            long int j = ja * m + ib;
            C[i * m * n + j] = A[ia * n + ja] * B[ib * n + jb];
        }
      }
    }
    
}



__global__ void per_element_kernel(long int *A, long int *B, long int *C,long
int m, long int n){
    // --> Complete the kernel ....

    long int id = (blockIdx.x *gridDim.y +blockIdx.y)*1024 + threadIdx.x * blockDim.y +threadIdx.y;
        if(id<m*m*n*n){
            int i = id/(m*n);
            int j = id%(m*n);
            int ia = i/n,jb =i%n ,ja =j/m,ib =j%m; 

             //int ni = ia * n + jb;
             //int nj = ja * m + ib;
           C[id]=A[ia*n +ja]*B[ib*n +jb];
            //printf("%d %d %d %d %d %d %d %d \n",i,j,ia,ja,ib,jb,ni,nj);
            
        }

}


/**
 * Prints any 1D array in the form of a matrix
 **/
void printMatrix(long int *arr, long int rows, long int cols, char *filename) {
  outfile.open(filename);
  for (long int i = 0; i < rows; i++) {
    for (long int j = 0; j < cols; j++) {
      outfile << arr[i * cols + j] << " ";
    }
    outfile << "\n";
  }
  outfile.close();
}

//--------------------------------------------------------------------------------------------

// * Timing functions taken from the matrix multiplication source code
//* rtclock - Returns the time of the day
//* printtime - Prints the time taken for computation

double rtclock() {
  struct timezone Tzp;
  struct timeval Tp;
  int stat;
  stat = gettimeofday(&Tp, &Tzp);
  if (stat != 0)
    printf("Error return from gettimeofday: %d", stat);
  return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}
//--------------------------------------------------------------------------------------------

//-------------------------------------------------------------------------------------------
void printtime(const char *str, double starttime, double endtime) {
  printf("%s%3f seconds\n", str, endtime - starttime);
}

//-------------------------------------------------------------------------------------------
int main(int argc, char **argv) {
  // Variable declarations
  long int m=20, n=20;
  cin >> m >> n;
  //cout<<m<<" "<<n<<endl;

   // Host_arrays
   long int *h_a,*h_b,*h_c;

   // Device arrays
   long int *d_a,*d_b,*d_c;

   // Allocating space for the host_arrays
   h_a = (long int *) malloc(m * n * sizeof(long int));
   h_b = (long int *) malloc(m * n * sizeof(long int));
   h_c = (long int *) malloc(m * m * n * n * sizeof(long int));

   // Allocating memory for the device arrays
   // --> Allocate memory for A on device xs
      cudaMalloc(&d_a,m*n*sizeof(long int));

   // --> Allocate memory for B on device
   cudaMalloc(&d_b,m*n*sizeof(long int));

   // --> Allocate memory for C on device
     cudaMalloc(&d_c,m*n*m*n*sizeof(long int));

   // Read the input matrix A
   for(long int i = 0; i < m * n; i++) {
       cin>>h_a[i];
      
   }
   //cout<<endl;

   //Read the input matrix B
   for(long int i = 0; i < m * n; i++) {
       cin>>h_b[i];
       
   }
   //cout<<endl;

   // Transfer the input host arrays to the device

   // --> Copy A from Host to Device
     cudaMemcpy(d_a,h_a,m*n*sizeof(long int),cudaMemcpyHostToDevice);
     cudaMemcpy(d_b,h_b,m*n*sizeof(long int),cudaMemcpyHostToDevice);

   long int gridDimx, gridDimy;

   // Launch the kernels

   // * Kernel 1 - per_row_AB_kernel
   // * To be launched with 1D grid, 1D block
   // * Each thread should process a complete row of A, B


   // --> Set the launch configuration

   
  dim3 grid1(m,1,1);
  dim3 block1(m,1,1);



   double starttime = rtclock();

   // --> Launch the kernel
   per_row_AB_kernel<<<grid1,block1>>>(d_a, d_b, d_c, m, n);
   cudaDeviceSynchronize();

   double endtime = rtclock();
       printtime("GPU Kernel-1 time: ", starttime, endtime);

   // --> Copy C from Device to Host
   cudaMemcpy(h_c,d_c,m*m*n*n*sizeof(long int),cudaMemcpyDeviceToHost);

   printMatrix(h_c, m * n, m * n,"kernel1.txt");
   cudaMemset(d_c, 0, m * n * m * n * sizeof(long int));




    //* Kernel 2 - per_column_AB_kernel
    //* To be launched with 1D grid, 2D block
   // * Each thread should process a complete column of  A, B
 

   // --> Set the launch configuration

      dim3 grid2(n,1,1);
     dim3 block2((n/2)+1,2,1);
     


   starttime = rtclock();

   // --> Launch the kernel
    per_column_AB_kernel<<<grid2,block2>>>(d_a, d_b,d_c,m,  n);

   cudaDeviceSynchronize();

   endtime = rtclock();
       printtime("GPU Kernel-2 time: ", starttime, endtime);


   // --> Copy C from Device to Host

   printMatrix(h_c, m * n, m * n,"kernel2.txt");
   cudaMemset(d_c, 0, m * n * m * n * sizeof(long int));


   // * Kernel 3 - per_element_kernel
   // * To be launched with 2D grid, 2D block
  //  * Each thread should process one element of the output

   gridDimx = ceil(float(n * n) / 16);
   gridDimy = ceil(float(m * m) / 64);
   dim3 grid3(gridDimx,gridDimy,1);
   dim3 block3(64,16,1);

   starttime = rtclock();

   // --> Launch the kernel

   per_element_kernel<<<grid3,block3>>>(d_a, d_b,d_c, m,  n);
   cudaDeviceSynchronize();

   endtime = rtclock();
       printtime("GPU Kernel-3 time: ", starttime, endtime);

   // --> Copy C from Device to Host

   printMatrix(h_c, m * n, m * n,"kernel3.txt");

   

  return 0;
}

