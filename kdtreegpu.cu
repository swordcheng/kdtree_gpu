/*
 *  kdtreegpu.cu
 *
 *  Created on: Sep 26, 2018
 *  Author: swordcheng
 */

#include "kdtreegpu.h"

kdtreegpu::kdtreegpu(int max_node_num, int max_neighbor_num, int query_num, int dim) 
{

	kdtree_dim = dim;
	kdtree_max_neighbor_num = max_neighbor_num + 1;
	kdtree_query_num = query_num;
	kdtree_node_num = max_node_num;
	// 主机端分配内存
	cudaMallocHost((Node**) &n, sizeof(Node) * kdtree_node_num);
	cudaMallocHost((int**) &split, sizeof(int) * kdtree_node_num);
	cudaMallocHost((float**) &query_node, sizeof(float) * kdtree_dim * kdtree_query_num);
	cudaMallocHost((void**) &query_result, sizeof(Pair_my) * kdtree_max_neighbor_num * kdtree_query_num);
	
	for (int i = 0; i < kdtree_node_num; i++)
	{	

		for (int j = 0; j < kdtree_dim; j++)
		{
			n[i].point[j] = float(rand())/RAND_MAX;
		}
	}
	for (int i = 0; i < kdtree_query_num; i++)
	{	

		for (int j = 0; j < kdtree_dim; j++)
		{
			query_node[i * kdtree_dim + j] = float(rand())/RAND_MAX;
		}
	}
}

//建树，采用非递归的形式，比递归的形式快
void kdtreegpu::build()
{

	pair<int,int> temp;

	temp.first = 0;
	temp.second = kdtree_node_num - 1;
	s.push(temp); // s为栈，用来存L, R, 相当于递归的栈

	while(!s.empty()) {

		temp = s.top();
		s.pop();

		int le = temp.first;
		int ri = temp.second;

		if (le > ri) {
			continue;
		}

		float var, maxs = -1;

		for (int i = 0; i < kdtree_dim; i ++) {
			float ave = 0;
			for (int j = le; j <= ri; j ++) {
				ave += n[j].point[i];
			}
			ave /= (ri - le + 1);
			var = 0;
			for (int j = le; j <= ri; j ++) {
				var += (n[j].point[i] - ave) * (n[j].point[i] - ave);
			}
			var /= (ri - le + 1);
			if (var > maxs) {
				maxs = var;
				split_node = i;
			}
		}
		int mid = (le + ri) >> 1;
		split[mid] = split_node;
		nth_element(n + le, n + mid, n + ri + 1);
		s.push(std::make_pair(le, mid - 1));
		s.push(std::make_pair(mid + 1, ri));
	}
}

// 由于cuda c不支持stl，因此在查询kd的时候，实现了堆。push_gpu(), pop_gpu(), swap_gpu()
//这三个函数是gpu上堆的操作

//交换堆中的节点
__host__ __device__ void swap_gpu(int id, int id1, int id2, 
									Pair_my *query_result_gpu, 
									int kdtree_max_neighbor_num)
{
	Pair_my temp;

	temp.id = query_result_gpu[id * kdtree_max_neighbor_num + id1].id;
	temp.dis = query_result_gpu[id * kdtree_max_neighbor_num + id1].dis;

	query_result_gpu[id * kdtree_max_neighbor_num + id1].id = query_result_gpu[id * kdtree_max_neighbor_num + id2].id;
	query_result_gpu[id * kdtree_max_neighbor_num + id1].dis = query_result_gpu[id * kdtree_max_neighbor_num + id2].dis;
	query_result_gpu[id * kdtree_max_neighbor_num + id2].id = temp.id;
	query_result_gpu[id * kdtree_max_neighbor_num + id2].dis = temp.dis;

}
// 在堆中插入节点
__host__ __device__ void push_gpu(float dis_, int id_, int id, 
									int * q_cur_id_gpu, Pair_my * query_result_gpu, 
									int kdtree_max_neighbor_num)
{
	query_result_gpu[id * kdtree_max_neighbor_num + q_cur_id_gpu[id]].id = id_;
	query_result_gpu[id * kdtree_max_neighbor_num + q_cur_id_gpu[id]].dis = dis_;

	int re = q_cur_id_gpu[id];

	while (re > 1)
	{

		if (query_result_gpu[id * kdtree_max_neighbor_num + re].dis > query_result_gpu[id * kdtree_max_neighbor_num + (re >> 1)].dis)
		{
			swap_gpu(id, re, re >> 1, query_result_gpu, kdtree_max_neighbor_num);
			re >>= 1;
		}
		else
		{
			break;
		}
	}
	q_cur_id_gpu[id]++;
} 
// 弹出节点
__host__ __device__ void pop_gpu(int id, int * q_cur_id_gpu, 
									Pair_my * query_result_gpu, 
									int kdtree_max_neighbor_num)
{

	query_result_gpu[id * kdtree_max_neighbor_num + 1] = query_result_gpu[id * kdtree_max_neighbor_num + q_cur_id_gpu[id] - 1];
	q_cur_id_gpu[id]--;

	int re = 1;
	while (re < q_cur_id_gpu[id])
	{
		if ((re << 1) < q_cur_id_gpu[id])
		{

			if ((re << 1 | 1) < q_cur_id_gpu[id])
			{
				if (query_result_gpu[id * kdtree_max_neighbor_num + (re << 1)].dis >= query_result_gpu[id * kdtree_max_neighbor_num + (re << 1 | 1)].dis)
				{

					if (query_result_gpu[id * kdtree_max_neighbor_num + (re << 1)].dis > query_result_gpu[id * kdtree_max_neighbor_num + re].dis)
					{
							swap_gpu(id, re << 1, re, query_result_gpu, kdtree_max_neighbor_num);
							re <<= 1;
					}
					else
					{
						break;
					}
				}
				else
				{
					if (query_result_gpu[id * kdtree_max_neighbor_num + (re << 1 | 1)].dis > query_result_gpu[id * kdtree_max_neighbor_num + re].dis)
					{
						swap_gpu(id, re << 1 | 1, re, query_result_gpu, kdtree_max_neighbor_num);
						re <<= 1;
						re |= 1;
					}
					else
					{
						break;
					}
				}
			}
			else
			{
				if (query_result_gpu[id * kdtree_max_neighbor_num + (re << 1)].dis > query_result_gpu[id * kdtree_max_neighbor_num + re].dis)
				{
					swap_gpu(id, re << 1, re, query_result_gpu, kdtree_max_neighbor_num);
					re <<= 1;
				}
				else
				{
					break;
				}
			}
		}
		else
		{
			break;
		}
	}
}

// gpu上查询一个节点，非递归查询
__host__ __device__ void query_one_gpu(int left, int right, int idx, 
									int *split_gpu, 
									float *query_node_gpu, 
									Pair_my *query_result_gpu, 
									Node *n_gpu, 
									int *q_cur_id_gpu, 
									int kdtree_max_neighbor_num, int kdtree_dim)
{
	Stack_my sta[20], temp;
	int cou = 0;
	sta[0].first = left;
	sta[0].second = right;
	cou++;
	while (cou){

		temp = sta[cou - 1];
		cou--;
		if (temp.first > temp.second){
			continue;
		}
		if (q_cur_id_gpu[idx] == kdtree_max_neighbor_num && temp.val > 
			query_result_gpu[idx * kdtree_max_neighbor_num + 1].dis){
			continue;
		}

		int mid = (temp.first + temp.second) >> 1;
		int cur = split_gpu[mid];
		float ans = 0;
		for (int i = 0; i < kdtree_dim; i++) 
		{
			ans += (query_node_gpu[idx * kdtree_dim + i] - 
				n_gpu[mid].point[i]) * 
					(query_node_gpu[idx * kdtree_dim + i] - 
				n_gpu[mid].point[i]);
		}
		if (q_cur_id_gpu[idx] < kdtree_max_neighbor_num) 
		{
			push_gpu(ans, mid, idx, q_cur_id_gpu, query_result_gpu, kdtree_max_neighbor_num);
		}
		else if (ans < query_result_gpu[idx * kdtree_max_neighbor_num + 1].dis) 
		{
			pop_gpu(idx, q_cur_id_gpu, query_result_gpu, kdtree_max_neighbor_num);
			push_gpu(ans, mid, idx, q_cur_id_gpu, query_result_gpu, kdtree_max_neighbor_num);
		}

		float radiu = abs(query_node_gpu[idx * kdtree_dim + cur] 
			- n_gpu[mid].point[cur]);

		if (query_node_gpu[idx * kdtree_dim + cur] 
			< n_gpu[mid].point[cur]){
			
			sta[cou].first = mid + 1;
			sta[cou].second = temp.second;
			sta[cou].val = radiu;
			cou++;
			
			sta[cou].first = temp.first;
			sta[cou].second = mid - 1;
			sta[cou].val = 0;
			cou++;
		}
		else{
			
			sta[cou].first = temp.first;
			sta[cou].second = mid - 1;
			sta[cou].val = radiu;
			cou++;

			sta[cou].first = mid + 1;
			sta[cou].second = temp.second;
			sta[cou].val = 0;
			cou++;
		}
	}
}

//gpu上并行查询kernel函数
__global__ void query_all_gpu(int query_num, int *split_gpu, 
								int *q_cur_id_gpu, float *query_node_gpu, 
								Pair_my *query_result_gpu, 
								Node *n_gpu, int kdtree_max_neighbor_num, 
								int kdtree_dim, int kdtree_node_num)
{
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < query_num; i += blockDim.x * gridDim.x)
	{
		q_cur_id_gpu[i] = 1;
		query_one_gpu(0, kdtree_node_num - 1, i, 
						split_gpu, query_node_gpu, 
						query_result_gpu, n_gpu, q_cur_id_gpu, 
						kdtree_max_neighbor_num, kdtree_dim);
	}
}

// gpu上并行查询
void kdtreegpu::query_gpu()
{
	Node *n_gpu = NULL;
	Pair_my *query_result_gpu;
	float *query_node_gpu = NULL;
	int *q_cur_id_gpu = NULL;
	int *split_gpu = NULL;

	//gpu上分配内存
	cudaMalloc((Node**) &n_gpu, sizeof(Node) * (kdtree_node_num));
	cudaMalloc((Pair_my**) &query_result_gpu, sizeof(Pair_my) * (kdtree_max_neighbor_num) * kdtree_query_num);
	cudaMalloc((float**) &query_node_gpu, sizeof(float) * kdtree_dim * kdtree_query_num);
	cudaMalloc((int**) &q_cur_id_gpu, sizeof(int) * kdtree_query_num);
	cudaMalloc((int**) &split_gpu, sizeof(int) * kdtree_node_num);
	// 把数据从主机端拷贝到设备端
	cudaMemcpy((Node*) n_gpu, (Node*) n, sizeof(Node) * (kdtree_node_num), cudaMemcpyHostToDevice);
	cudaMemcpy((int*) split_gpu, (int*) split, sizeof(int) * (kdtree_node_num), cudaMemcpyHostToDevice);
	cudaMemcpy((float*) query_node_gpu, (float*) query_node, sizeof(float) * (kdtree_query_num * kdtree_dim), cudaMemcpyHostToDevice);
	// 调用kernel函数，并行查询
	query_all_gpu<<<32, 256>>>(kdtree_query_num, split_gpu, q_cur_id_gpu, 
					query_node_gpu, query_result_gpu, n_gpu, 
					kdtree_max_neighbor_num, kdtree_dim, kdtree_node_num);
	//同步，所有的线程执行完
	cudaDeviceSynchronize();
	//将结果拷贝回主机端
	cudaMemcpy(query_result, query_result_gpu, sizeof(Pair_my) * kdtree_max_neighbor_num  * kdtree_query_num, cudaMemcpyDeviceToHost);
	//释放设备端显存
	cudaFree(n_gpu);
	cudaFree(query_result_gpu);
	cudaFree(split_gpu);
	cudaFree(query_node_gpu);
	cudaFree(q_cur_id_gpu);
}
//cpu上查询一个点
void kdtreegpu::query_one(int left, int right, int id)
{

	Stack_my sta[20], temp;

	int cou = 0;
	sta[0].first = left;
	sta[0].second = right;
	cou++;
	while (cou){

		temp = sta[cou - 1];
		cou--;

		if (temp.first > temp.second){
			continue;
		}
		if (que.size() == kdtree_max_neighbor_num - 1 && temp.val > que.top().dis){
			continue;
		}
		
		int mid = (temp.first + temp.second) >> 1;
		int cur = split[mid];
		float ans = 0;
		for (int i = 0; i < kdtree_dim; i++) 
		{
			ans += (query_node[id * kdtree_dim + i] - n[mid].point[i]) * 
					(query_node[id * kdtree_dim + i] - n[mid].point[i]);
		}
		if (cnt < kdtree_max_neighbor_num - 1) 
		{
			Pair_my tmp;
			tmp.id = mid;
			tmp.dis = ans;
			que.push(tmp);
			cnt++;
		}
		else if (ans < que.top().dis) 
		{
			Pair_my tmp;
			tmp.id = mid;
			tmp.dis = ans;
			que.pop();
			que.push(tmp);
		}
		float radiu = abs(query_node[id * kdtree_dim + cur] - n[mid].point[cur]);
		if (query_node[id * kdtree_dim + cur] < n[mid].point[cur]){

			sta[cou].first = mid + 1;
			sta[cou].second = temp.second;
			sta[cou].val = radiu;
			cou++;
			
			sta[cou].first = temp.first;
			sta[cou].second = mid - 1;
			sta[cou].val = 0;
			cou++;
		}
		else{
			sta[cou].first = temp.first;
			sta[cou].second = mid - 1;
			sta[cou].val = radiu;
			cou++;
			sta[cou].first = mid + 1;
			sta[cou].second = temp.second;
			sta[cou].val = 0;
			cou++;
		}
	}

}

// cpu上查询，并验证cpu的查询和gpu的查询结果是否一致
int kdtreegpu::query_cpu_and_check()
{

	int error = 0;
	for (int i = 0; i < kdtree_query_num; i++)
	{
		cnt = 0;
		query_one(0, kdtree_node_num-1, i);

		int count = 0;
		int num = que.size();
		while (!que.empty())
		{
			Pair_my tmp;
			tmp.id = que.top().id;
			tmp.dis = que.top().dis;

			for (int k = 1; k < kdtree_max_neighbor_num; k++)
			{
				if (query_result[i * kdtree_max_neighbor_num + k].id == tmp.id)
				{
					count++;
				}
			}
			
			que.pop();
		}

		if (count != num)
		{
			printf("%d %d\n", i, count);
			printf("EROOR\n");
			error++;
		}
	}
	return error;
}
//析构函数
kdtreegpu::~kdtreegpu() {

	cudaFree(n);
	cudaFree(query_node);
	cudaFree(split);
	cudaFreeHost(query_result);

}

int main()
{
	int n = 60000;
	int neighbor_num = 20;
	int query_num = 60000;
	int dim = 3;
	srand( (unsigned)time( NULL )); 

	int test_times = 100;
	float ave_build_cpu = 0;
	float ave_query_gpu = 0;
	float ave_err_times = 0;
	for (int i = 0; i < test_times; i++)
	{
		printf("%d\n", i);
		kdtreegpu kd(n, neighbor_num, query_num, dim);

		begin = clock();
		kd.build();
		end = clock();
		ave_build_cpu += double(end - begin)/CLOCKS_PER_SEC;

		begin = clock();
		kd.query_gpu();
		end = clock();

		ave_query_gpu += double(end - begin)/CLOCKS_PER_SEC;
		//std::cout<< "kd tree time:" << double(end - begin)/CLOCKS_PER_SEC << std::endl;
		ave_err_times += kd.query_cpu_and_check();
	}
	printf("ave_build_cpu times: %lf\n", ave_build_cpu / test_times);
	printf("ave_query_gpu times: %lf\n", ave_query_gpu / test_times);
	printf("ave_err_times: %lf\n", ave_err_times / query_num / test_times);

	return 0;
}
