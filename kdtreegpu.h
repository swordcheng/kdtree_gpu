/*
 *  kdtreegpu.h
 *
 *  Created on: Sep 26, 2018
 *  Author: swordcheng
 */

// 该版本的kd树同时实现了cpu和gpu的查询，cpu端的查询主要是用来验证gpu端的查询是否正确

#include <iostream>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <stack>
#include <cstdlib>
#include <algorithm>
#include <queue>


using namespace std;

int split_node;

clock_t begin, end;

//kd树的node
struct Node {

	float point[3];
	// 运算符重载
	bool operator<(const Node &n)const {
		return point[split_node] < n.point[split_node];
	}

};
// 自定义pair，cuda c不支持STL
struct Pair_my{
	int id;
	float dis;

	bool operator<(const Pair_my &p)const {
		return dis < p.dis;
	}
};

// 自定义stack，cuda c不支持STL
struct Stack_my
{
	int first;
	int second;
	float val;
};


class kdtreegpu {

public:

	kdtreegpu(int max_node_num, int max_neighbor_num, int query_num, int dim); //构造函数
	void build(); // 建树
	void query_gpu(); // GPU上查询
	void query_one(int left, int right, int id); //CPU上查询一个点
	int query_cpu_and_check(); // 此为验证函数，验证CPU版本的kd树和GPU版本的kd树查询结果是否一致
	virtual ~kdtreegpu(); 

private:

	int kdtree_dim; // kd树维度
	int kdtree_max_neighbor_num; // 最大最近邻个数
	int kdtree_query_num; // 查询点的数目
	int kdtree_node_num; // kd树节点数目
	int cnt;

	int *split; // kd树分割点
	Node *n;  // kd树节点
	Pair_my *query_result; // 查询结果
	float * query_node; // 查询节点

	stack< pair<int,int> > s; // 栈，用在CPU上建树
	priority_queue< Pair_my > que; // 优先队列，CPU上查询的时候用到的
};
