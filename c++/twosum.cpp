// ConsoleApplication2.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include<iostream>
#include<vector>
#include<map>

using namespace std;

int nums[]= { 0,1,2,3,4,5,6,7,8 };
int target = 5;



class solution {
public:
	vector<int>twoSum(vector<int>&nums, int target)
	{
		map<int,int> m;
		vector<int> vec;
		for (int i = 0; i < nums.size(); i++)
		{
			int b = target - nums[i];
			if (m.count(b))
			{
				vec.push_back(m[b]);
				vec.push_back(i);
				break
			}
			m[nums[i]] = i;
		}
		return vec;

	}
};