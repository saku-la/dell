//定义一个数组和一个目标，判断数组里
//是否有两个数能组成目标

#include "stdafx.h"
#include<iostream>
#include<vector>
#include<map>

using namespace std;
class solution
{
public:
	vector<int> twoSum(vector<int> nums, int target)
	{

		map<int, int> m;
		vector<int> vec;
		for (int i = 0; i < nums.size(); i++)
		{
			int b = target - nums[i];
			if (m.count(b))
			{
				vec.push_back(m[b]);
				vec.push_back(i);
				break;
			}
			m[nums[i]] = i;
		}
		return vec;
	}
};



void main()
{
	vector<int> nums = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
	int target = 7;
	solution solution1;
	vector<int> a = solution1.twoSum(nums, target);
	
	for (int i = 0; i<a.size(); i++)
		std::cout << a[i] << endl;
	system("Pause");

}

