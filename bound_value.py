import random 
import sys
from random import randint
import numpy
import math 
import numpy as np
import time
from numba import cuda,jit,int32,float32,uint16

'''
The code outputs the total value and weight of a packing. If the set of 
items in the final packing needs to be calculated, the code needs to be 
modified to include appropriate pointers in DP tables. 


variable notations in the code are as below
l : number of categories 
n : number of items in each category
epsilon : Input parameter indicating approximation factor and fairness violation
v_i : Upper bound of value in each category
v_i_l : Lower bound of value in each category
B : The total capacity of knapsack
'''


v_min=1;
v_max=100;
n=int(sys.argv[1]);
l=int(sys.argv[2]);
epsilon=float(sys.argv[3]);
v_i=float(sys.argv[4]);
v_i_l=float(sys.argv[5]);
B=int(sys.argv[6]);



depsilon=pow(1+3*epsilon/8,1/(math.ceil(math.log(l,2))+1))-1;

print("n: ");
print(n);
print("l: ");
print(l);
print("epsilon :")
print(epsilon);


print("depsilon ")
print(depsilon);

MAX_WEIGHT=(n*l*v_max)+1;
MAX_WEIGHT_GROUP=(n*v_max)+1;

seed=25;

random.seed(seed);

values = numpy.zeros((n,l),dtype=numpy.uint8);
weights=numpy.zeros((n,l),dtype=numpy.uint8);

TWEIGHT=0;
TVALUES=0;

for j in range(l):
	for i in range(n):
		values[i][j]=randint(v_min,v_max);
		TVALUES=TVALUES+values[i][j];

	

random.seed(2*seed);

for j in range(l):
	for i in range(n):
		weights[i][j]=randint(v_min,v_max);
		TWEIGHT=TWEIGHT+weights[i][j];

print(" Total Value:");
print(TVALUES);
print(" Total Weight:");
print(TWEIGHT);
print("\n\n");



time0=time.time();

@cuda.jit
def initialize_X(W,X,W_len,X_len,l,eps,deps,v_i,v_min):
	tx=cuda.threadIdx.x;
	ty=cuda.blockIdx.x;
	bw=cuda.blockDim.x;
	
	pos = tx + ty * bw;
	
	if pos >=l: 
		return;

	j_prime=W_len-1;
	for j in range(X_len-1,-1,-1):
		while pow(1+eps,j_prime) >= pow(1+deps,j):
			if v_min*pow((1+eps),j_prime-1)<= v_i :
				if v_min*pow((1+eps),j_prime+1)>= v_i_l :
					X[pos][j]=min(X[pos][j],W[pos][j_prime]);
					if j<X_len-1:
						X[pos][j]=min(X[pos][j],X[pos][j+1]);
				
			j_prime=j_prime-1;
			

@cuda.jit
def combine_bundles_parallel(X_prev,X_cur,deps,X_len,X_prev_height):
	tx=cuda.threadIdx.x;
	ty=cuda.blockIdx.x;
	bw=cuda.blockDim.x;
	
	pos = tx + ty * bw;

	if pos>=(int(len(X_prev[:,0])/2)*X_len):
		return;
	
	
	X_0=X_prev[int(pos/(X_len))*2,:];
	X_1=X_prev[int(pos/(X_len))*2+1,:];

	X_cur_thread=X_cur[int(pos/(X_len)),:];
	X_cur_thread_index=pos%(X_len);
	
	for j_prime in range(X_len):
			for j_dprime in range(X_len): 
				if pow(1+deps,X_cur_thread_index) <= pow(1+deps,j_prime)+pow(1+deps,j_dprime):
					X_cur_thread[X_cur_thread_index]=min(X_0[j_dprime]+X_1[j_prime], X_cur_thread[X_cur_thread_index]);

			
						
						

@cuda.jit
def min_weight_cuda(values,weights,epsilon,output_thread,v_min,v_max,MATHLOG,MAX_WEIGHT,value_1,no_of_threads,v,MAX_WEIGHT_GROUP):
	tx=cuda.threadIdx.x;
	ty=cuda.blockIdx.x;
	bw=cuda.blockDim.x;
	
	pos = tx + ty * bw;
	
	
	
	if pos > no_of_threads-1 :
		return;
	
	i_p=int(pos/int(MATHLOG));
	j_p=pos%int(MATHLOG);
	
	weight=weights[:,i_p];
	value=values[:,i_p];

	v=v_min*pow(v,j_p);
	
	max_v_h=int(len(value)*(1+2*epsilon)/(epsilon))+1;	
	
	# Use DP table with only two rows because of the limitation of shared '
	# memory
	
	d_h=cuda.shared.array(shape=(1,2,12284), dtype=uint16);
	
	
	for i in range(len(value)):
		value_1[pos][i]=int(len(value)*value[i]/(epsilon*v));
	
	
	
	for i in range(2):
		for j in range(max_v_h):
			d_h[0][i][j]=MAX_WEIGHT_GROUP;

	
	if value_1[pos][0] < max_v_h :
		d_h[0][0][int(value_1[pos][0])]=weight[0];
	
	
	
	for i in range(1,len(value)):
		for j in range(max_v_h):
			d_h[0][i%2][j]=d_h[0][(i-1)%2][j];
			if (j-int(value_1[pos][i])) >= 0 :
				if d_h[0][i%2][j] > d_h[0][(i-1)%2][j-int(value_1[pos][i])]+weight[i]:
					d_h[0][i%2][j]=d_h[0][(i-1)%2][j-int(value_1[pos][i])]+weight[i];
			
	
	
	minSet=MAX_WEIGHT;
			
	for i in range(int((1-2*epsilon)*n/epsilon), int((1+2*epsilon)*n/epsilon)+1):
		if i < max_v_h :	
			if i>= 0 :
				if minSet > d_h[0][(len(value)-1)%2][i]:
					minSet=d_h[0][(len(value)-1)%2][i];
	
	if minSet==MAX_WEIGHT_GROUP:
		minSet=MAX_WEIGHT;
		
	output_thread[i_p][j_p]=minSet;
	
	
	

	
	

W_host=numpy.zeros((l,int(math.ceil(math.log(n*v_max/v_min,1+epsilon/8)))),dtype=numpy.uint32);

for i in range(l):
	for j in range(int(math.ceil(math.log(n*v_max/v_min,1+epsilon/8)))):
		W_host[i][j]=MAX_WEIGHT;

MAX_THREADS_PER_BLOCK=1;
	
noOfblocks=int(l*(int(math.ceil(math.log(n*v_max/v_min,1+epsilon/8))))/MAX_THREADS_PER_BLOCK)+1;
noOfthreads=l*(int(math.ceil(math.log(n*v_max/v_min,1+epsilon/8))));

d_values=cuda.to_device(values);
d_weights=cuda.to_device(weights);
d_value_1=cuda.to_device(np.zeros((noOfthreads+1,n),dtype=np.uint16));
W=cuda.to_device(W_host);


print("Before min_weight_call")
print("max weight");
print(MAX_WEIGHT);
print("max weight in group 3");
print(MAX_WEIGHT_GROUP);

print("bundling started");

# Computing bundles parallelly
min_weight_cuda[noOfblocks,MAX_THREADS_PER_BLOCK](d_values,d_weights,epsilon/6,W,
v_min,v_max,math.ceil(math.log(n*v_max/v_min,1+epsilon/8)),MAX_WEIGHT,
d_value_1,noOfthreads,(1+epsilon/8),MAX_WEIGHT_GROUP);


	
time1=time.time();
print("bundling completed" );
		

X=numpy.zeros((l,int(math.ceil(math.log(n*l*v_max/v_min,1+depsilon)))));

X_dims1=len(X[0,:]);

for i in range(l):
	for j in range(int(math.ceil(math.log(n*l*v_max/v_min,1+depsilon)))):
		X[i][j]=MAX_WEIGHT;
		
d_W=cuda.to_device(W);
d_X=cuda.to_device(X);
initialize_X[1,l](d_W,d_X,len(W[0,:]),len(X[0,:]),l,epsilon/8,depsilon,v_i,v_min);


for i in range(l):
	for j in range(int(math.ceil(math.log(n*l*v_max/v_min,1+depsilon)))):
		X[i][j]=d_X[i][j];

		
max_in_group=0;		
for i in range(l):
	max_in_group=0;
	for j in range(int(math.ceil(math.log(n*l*v_max/v_min,1+depsilon)))):
		if X[i][j]<MAX_WEIGHT:
			max_in_group=max(max_in_group,j);
		

X_prev=X;
X_cur=numpy.full((int(l/2),int(math.ceil(math.log(n*l*v_max/v_min,1+depsilon)))),MAX_WEIGHT);


d_X_prev=cuda.to_device(X_prev);
d_X_cur=cuda.to_device(X_cur);

r=l;
k=0;

print("Combining bundles now...");

while r>=1:
	MAX_THREADS_PER_BLOCK=512;
	noOfblocks=int(int(r/2)*len(X[0,:])/MAX_THREADS_PER_BLOCK)+1;
	
	# Combining bundles in devide and conquere fashion, parallelly at each
	# stage
	
	combine_bundles_parallel[noOfblocks,MAX_THREADS_PER_BLOCK](d_X_prev,d_X_cur,depsilon,len(X_prev[0,:]),len(X_prev[:,0]));
	k=int(r/2)+r%2;
	
	if k<=1:
		break;
	
	X_prev=numpy.zeros((k,int(math.ceil(math.log(n*l*v_max/v_min,1+depsilon)))));
	
	
	for j in range(int(r/2)):
		for s in range(int(math.ceil(math.log(n*l*v_max/v_min,1+depsilon)))):
			X_prev[j][s]=d_X_cur[j][s];
	
	if (r%2)>0:
		for s in range(int(math.ceil(math.log(n*l*v_max/v_min,1+depsilon)))):
			X_prev[k-1][s]=d_X_prev[r-1][s];
			
	X_cur=numpy.full((int(k/2),int(math.ceil(math.log(n*l*v_max/v_min,1+depsilon)))),MAX_WEIGHT);
	

	d_X_prev=cuda.to_device(X_prev);
	d_X_cur=cuda.to_device(X_cur);
	
	r=k;
	


print("\n\n");

X_prev=numpy.zeros((int(math.ceil(math.log(n*l*v_max/v_min,1+depsilon)))))

for i in range(int(math.ceil(math.log(n*l*v_max/v_min,1+depsilon)))):
	X_prev[i]=d_X_cur[0][i];

maxJ=-12;
for i in range(int(math.ceil(math.log(n*l*v_max/v_min,1+depsilon)))):
	if X_prev[i] <= B :
		if  v_min*pow((1+depsilon),i-1) <= (l*v_i):
			maxJ=max(maxJ,i);


print(" Value of packing :");
print(v_min*pow(1+depsilon,maxJ));
if maxJ >=0 :
	print("Weight of packing : ")
	print(X_prev[maxJ]);		
print("Knapsack Weight: ");
print(B);
print(" v_i: ");
print(v_i);
print(" v_i_l: ");
print(v_i_l);
print(" n: ");
print(n);
print(" l: ");
print(l);
print("epsilon");
print(epsilon);
print("depsilon");
print(depsilon);

print("\n\n\n");

print("total running time : ");
print(time.time()-time0);
print("seed: ");
print(seed);

