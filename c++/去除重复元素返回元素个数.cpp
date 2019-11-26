
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
		auto iter = unique(nums.begin(), nums.end());
		nums.erase(iter, nums.end());
		return nums.size();
	}
};
int main() {
	vector<int> a = { 1,3,2,4,6,5,7,3,4,5,3,2,5,3 };
	sort(a.begin(), a.end());
	Sloution a1;
	int c=a1.remove(a);
	cout << c << endl;
	for (int i = 0; i < a.size(); i++)
		cout << a[i] << endl;
	system("Pause");
}