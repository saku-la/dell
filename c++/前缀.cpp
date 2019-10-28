//编写一个函数来查找字符串数组中的最长公共前缀。
//
//如果不存在公共前缀，返回空字符串 ""。
//
//示例 1:
//
//输入: ["flower", "flow", "flight"]
//	输出 : "fl"
//	示例 2 :
//
//	输入 : ["dog", "racecar", "car"]
//	输出 : ""
//	解释 : 输入不存在公共前缀。
//	说明 :
//
//	所有输入只包含小写字母 a - z 。
//
//1028 a.at()


#include "stdafx.h"
#include<iostream>
#include<vector>
#include<map>
#include<string>

using namespace std;
class Solution {
public:
	string dell(vector<string>& strs) {
		int lenVec = strs.size();
		cout << lenVec << endl;
		if (lenVec = 0)
			return " ";
		if (lenVec == 1) {
			cout << strs.at(0) << endl;
			return strs.at(0);
			
		}
		int j = 0;
		while (1) {
			for (int i = 0; i < lenVec; i++) {
				if (j >= strs.at(i).size())
					return strs.at(i).substr(0, j);
				char c = strs.at(0).at(j);
				if (c != strs.at(i).at(j))
					return strs.at(i).substr(0, j);

			}
			j++;
		}
	}
};


void main()
{
	vector<string> a = { "hello","hi","ho","hio" };
	Solution a1;
	//vector<string> c = a1.dell(a);
	//for(int i=0;i<c.size();i++)
	//	cout << c [i]<< endl;
	string out = a1.dell(a);
	cout << out << endl;
	
	system("Pause");
	
}

