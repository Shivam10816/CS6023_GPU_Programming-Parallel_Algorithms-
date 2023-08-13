/*
 * Title: CS6023, GPU Programming, Jan-May 2023, Assignment-3
 * Description: Activation Game 
 */

#include <cstdio>        // Added for printf() function 
#include <sys/time.h>    // Added to get time of day
#include <cuda.h>
#include <bits/stdc++.h>
#include <fstream>
#include "graph.hpp"
 
using namespace std;


ofstream outfile; // The handle for printing the output

/******************************Write your kerenels here ************************************/



//--------------compute start and end for next level--------------------

__global__ void next_level(int *d_InDegree,int *d_start,int *d_end,int V){

    int i=d_start[0];

    while(i<V && d_InDegree[i]==0){
      i++;
    }
    d_end[0] =i-1;

    
}

//-----------------Compute Indegree of each vertex---------------

__global__ void compute(int *d_InDegree, int *d_csrList,int E){
    
    int id=blockIdx.x*blockDim.x + threadIdx.x;

    if(id<E){
        int vertex = d_csrList[id];               //get Destination vertex from csrList
        int prev = atomicAdd(&d_InDegree[vertex],1);  // Atomic add 1 to Indegree of vertex
        
    }
}
    
    
__global__ void activationNode(int *adi, int *apr ,int *csr,int *offset,int *indegree,int * active,int start , int end,int level){

    //-------Id is vertex in Level------------

    int id = start+blockIdx.x*blockDim.x + threadIdx.x;

    //Check if vertex is of 1st or last vertex of level
    if(id==start || id==end){

        // If condition for activation meets then fire that vertex
        if(adi[id]>=apr[id]){
            
            // increase count of active vertex by one in active
            // atomic add is used to avoid data inconsistency

            int prev = atomicAdd(&active[level],1);

            // Fire the vertex ( Make adr of all desination increase by one)
            for(int i=offset[id];i<offset[id+1];i++){
                int node = csr[i];
                prev = atomicAdd(&adi[node],1);

                // decrease Indegree of destination by one (for getting level)
                prev = atomicSub(&indegree[node],1);
            }


        }
        else{
             
            for(int i=offset[id];i<offset[id+1];i++){
                int node = csr[i];
                // decrease Indegree of destination by one (for getting level)
                int prev = atomicSub(&indegree[node],1);
                
            }
        }
    }

    //Check if vertex is between 1st or last vertex of level
    else if(id>start && id<end){

        //---check condition for activation function---

        if( adi[id]>=apr[id] && (adi[id-1]>=apr[id-1] || adi[id+1]>=apr[id+1]) ){
            
            // increase count of active vertex by one in active
            // atomic add is used to avoid data inconsistency

            int prev = atomicAdd(&active[level],1);

            
            // Fire the vertex ( Make adr of all desination increase by one)
            for(int i=offset[id];i<offset[id+1];i++){
                int node = csr[i];
                prev = atomicAdd(&adi[node],1);

                // decrease Indegree of destination by one (for getting level)
                prev = atomicSub(&indegree[node],1);
            }


        }
        else{
            
            // If vertex is deactivate
            for(int i=offset[id];i<offset[id+1];i++){
                int node = csr[i];

                // decrease Indegree of destination by one (for getting level)
                int prev = atomicSub(&indegree[node],1);
                
            }
        }
    }
}
  
    
/**************************************END*************************************************/



//Function to write result in output file
void printResult(int *arr, int V,  char* filename){
    outfile.open(filename);
    for(long int i = 0; i < V; i++){
        outfile<<arr[i]<<" ";   
    }
    outfile.close();
}

/**
 * Timing functions taken from the matrix multiplication source code
 * rtclock - Returns the time of the day 
 * printtime - Prints the time taken for computation 
 **/
double rtclock(){
    struct timezone Tzp;
    struct timeval Tp;
    int stat;
    stat = gettimeofday(&Tp, &Tzp);
    if (stat != 0) printf("Error return from gettimeofday: %d", stat);
    return(Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

void printtime(const char *str, double starttime, double endtime){
    printf("%s%3f seconds\n", str, endtime - starttime);
}

int main(int argc,char **argv){
    // Variable declarations
    int V ; // Number of vertices in the graph
    int E; // Number of edges in the graph
    int L; // number of levels in the graph

    //Reading input graph
    char *inputFilePath = argv[1];
    graph g(inputFilePath);

    //Parsing the graph to create csr list
    g.parseGraph();

    //Reading graph info 
    V = g.num_nodes();
    E = g.num_edges();
    L = g.get_level();


    //Variable for CSR format on host
    int *h_offset; // for csr offset
    int *h_csrList; // for csr
    int *h_apr; // active point requirement

    //reading csr
    h_offset = g.get_offset();
    h_csrList = g.get_csr();   
    h_apr = g.get_aprArray();
    
    // Variables for CSR on device
    int *d_offset;
    int *d_csrList;
    int *d_apr; //activation point requirement array
    int *d_aid; // acive in-degree array
    //Allocating memory on device 
    cudaMalloc(&d_offset, (V+1)*sizeof(int));
    cudaMalloc(&d_csrList, E*sizeof(int)); 
    cudaMalloc(&d_apr, V*sizeof(int)); 
    cudaMalloc(&d_aid, V*sizeof(int));

    //copy the csr offset, csrlist and apr array to device
    cudaMemcpy(d_offset, h_offset, (V+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrList, h_csrList, E*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_apr, h_apr, V*sizeof(int), cudaMemcpyHostToDevice);

    // variable for result, storing number of active vertices at each level, on host
    int *h_activeVertex;
    h_activeVertex = (int*)malloc(L*sizeof(int));
    // setting initially all to zero
    memset(h_activeVertex, 0, L*sizeof(int));

    // variable for result, storing number of active vertices at each level, on device
    int *d_activeVertex;
	cudaMalloc(&d_activeVertex, L*sizeof(int));
    cudaMemset(d_activeVertex,0,L*sizeof(int));
    //printD(d_activeVertex,L);


/***Important***/

// Initialize d_aid array to zero for each vertex
cudaMemset(d_aid,0,V*sizeof(int));
// Make sure to use comments

/***END***/
double starttime = rtclock(); 

/*********************************CODE AREA*****************************************/

//--Define Indegree Array on Host--
int *h_InDegree;
h_InDegree =(int*)malloc(V*sizeof(int));
memset(h_InDegree, 0, V*sizeof(int));

//--Define Indegree Array on Device---

int *d_InDegree;
cudaMalloc(&d_InDegree,V*sizeof(int));
cudaMemset(d_InDegree,0,V*sizeof(int));

//------Compute in degree of each vertices------
int blocks =E/1024 +1;
compute<<<blocks,1024>>>(d_InDegree,d_csrList,E);

//copy Indegree from Device to Host

cudaMemcpy(h_InDegree, d_InDegree, V*sizeof(int), cudaMemcpyDeviceToHost);


//---Launch kernel for each level----
int h_start[1],h_end[1] ,*d_end,*d_start;

h_start[0]=0;
h_end[0]=0;



int level=0;

cudaMalloc(&d_start,V*sizeof(int));
cudaMalloc(&d_end,V*sizeof(int));

cudaMemset(&d_start,0,sizeof(int));
cudaMemset(&d_end,0,sizeof(int));



while(level<L){

    // Calculate Nodes In Level (With Indegree =0 from start vertex)
    

    //set dimentions for blocks and grid
   
    
    next_level<<<1,1>>>(d_InDegree,d_start,d_end,V);

    //copy start and end of level from Device to Host

    cudaMemcpy(h_start, d_start, 1*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_end, d_end, 1*sizeof(int), cudaMemcpyDeviceToHost);

    blocks = (h_end[0]-h_start[0]+1)/1024 +1;
    int start =h_start[0],end =h_end[0];
   
    //Launch Kernel to calculate Active Node in Level
    activationNode<<<blocks,1024>>>(d_aid, d_apr ,d_csrList,d_offset,d_InDegree,d_activeVertex,start ,end,level);

    

   
    
    level++;

    
    

    //----copy end point to start from host to device

    h_end[0]+=1;
    cudaMemcpy(d_start,h_end,sizeof(int),cudaMemcpyHostToDevice);
    

}

//printD(d_activeVertex,L);
cudaMemcpy(h_activeVertex,d_activeVertex,L*sizeof(int),cudaMemcpyDeviceToHost) ;   

/********************************END OF CODE AREA**********************************/
double endtime = rtclock();  
printtime("GPU Kernel time: ", starttime, endtime);  

// --> Copy C from Device to Host
char outFIle[30] = "./output.txt" ;
printResult(h_activeVertex, L, outFIle);
if(argc>2)
{
    for(int i=0; i<L; i++)
    {
        printf("level = %d , active nodes = %d\n",i,h_activeVertex[i]);
    }
}

    return 0;
}
