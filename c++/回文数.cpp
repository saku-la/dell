
#include "stdafx.h"
#include<iostream>
#include<vector>
#include<map>

using namespace std;
class Solution {
public :
		bool IsPalindrome(int x) {
		if (x < 0 || (x % 10 == 0 && x != 0)) {
			return false;
		}
		int revertedNumber = 0;
		while (x > revertedNumber) {
			revertedNumber = revertedNumber * 10 + x % 10;
			x /= 10;
		}
		return x == revertedNumber || x == revertedNumber / 10;

	}
	
};


void main()
{
	int a = 123454321;
	Solution a1;
	int c = a1.IsPalindrome(a);
	cout << c << endl;
	system("Pause");
	
}