#include "stdafx.h"
#include<iostream>
#include<vector>
#include<map>
#include <algorithm>
#include<algorithm>

using namespace std;
class Sloution {
public:
	int removeRepeat(vector<int>& nums,int val) {
		int i = 0;
		for (int j = 0; j < nums.size(); j++) {
			if (nums[j] != val) {
				nums[i] = nums[j];
					i++;
			}
		}
		return i;
	}
};
int main() {
	vector<int> a = { 1,3,2,4,6,5,7,3,4,5,3,2,5,3 };
	//sort(a.begin(), a.end());
	Sloution a1;
	int val = 3;
	int c=a1.removeRepeat(a,val);
	//cout << c << endl;
	//for (int i = 0; i < a.size(); i++)
		//cout << a[i] << endl;
	cout << c << endl;
	system("Pause");
}