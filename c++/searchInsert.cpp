#include "stdafx.h"
#include<iostream>
#include<vector>
#include<map>
#include <algorithm>
#include<algorithm>

using namespace std;
class Sloution {
public:
	int search(vector<int>& nums, int target) {
		int len = nums.size();
		if (len == 0) return 0;
		for (int i = 0; i < len; i++) {
			if (nums[i] >= target) return i;
		}
		return len;

	}
};
int main() {
	vector<int> a = { 1,2,2,4,5,6,7,8 };
	sort(a.begin(), a.end());
	int target = 3;
	Sloution a1;
	int c=a1.search(a,target);
	cout << c << endl;
	system("Pause");
}