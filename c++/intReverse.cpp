// ConsoleApplication2.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include<iostream>
#include<vector>
#include<map>

using namespace std;
class Solution {
public:
	int reverse(int x) {
		int rev = 0;
		while (x != 0) {
			int pop = x % 10;
			x /= 10;
			if (rev > INT_MAX / 10 || (rev == INT_MAX / 10 && pop > 7))return 0;
			if (rev < INT_MIN / 10 || (rev == INT_MIN / 10 && pop < -8))return 0;
			rev = rev * 10 + pop;
		}
		return rev;
	}
};


void main()
{
	int a = -236;
	Solution a1;
	int c = a1.reverse(a);
	//vector<int> nums = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
	//int target = 7;
	//solution solution1;
	//vector<int> a = solution1.twoSum(nums, target);
	////vector<int> a = twoSum(nums, target);
	//for(int i =0;i<a.size();i++)
	//	std::cout << a[i] << endl;
	cout << c << endl;
	system("Pause");
	
}
