
#include "stdafx.h"
#include<iostream>
#include<vector>
#include<map>
#include <algorithm>
#include<algorithm>

using namespace std;
class Sloution {
public:
	int remove(vector<int>& nums) {
		auto iter = unique(nums.begin(), nums.end());//将相邻重复元素放在最后
		nums.erase(iter, nums.end());//删除最后的元素
		return nums.size();
	}
};
int main() {
	vector<int> a = { 1,3,2,4,6,5,7,3,4,5,3,2,5,3 };
	sort(a.begin(), a.end());//将数组排序
	Sloution a1;//实例
	int c=a1.remove(a);
	cout << c << endl;
	for (int i = 0; i < a.size(); i++)
		cout << a[i] << endl;
	system("Pause");
}