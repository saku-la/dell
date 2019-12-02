//判断val_str（all）是否出现在str里，并返回出现的初始位置
#include "stdafx.h"
#include<iostream>
#include<vector>
#include<map>
#include <algorithm>
#include<algorithm>

using namespace std;
class Sloution {
public:
	int strStr(string str, string val_str) {
		if (str.length() < val_str.length()) return -1;
		if (val_str.length() == 0) return -1;
		int i = 0, j = 0;
		while (i < str.length()) {
			if (str[i] == val_str[j]) {
				i++;
				j++;
				if (j >= val_str.length()) {
					return i - j;
				}
				else {
					if (j < val_str.length()) {
						i = i - j + 1;
						j = 0;
					}
					else {
						i++;
					}

				}
			}
		}
		return -1;
	}
};
int main() {
	//vector<int> a = { 1,3,2,4,6,5,7,3,4,5,3,2,5,3 };
	string a = "shiyingjieniubi";
	string b = "niube";
	//sort(a.begin(), a.end());
	Sloution a1;
	//int val = 3;
	int c=a1.strStr(a,b);
	//cout << c << endl;
	//for (int i = 0; i < a.size(); i++)
		//cout << a[i] << endl;
	cout << c << endl;
	system("Pause");
}