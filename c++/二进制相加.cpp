#include "stdafx.h"
#include<iostream>
#include<vector>
#include<map>
#include <algorithm>
#include<algorithm>

using namespace std;
class Sloution {
public:
	string sumTwo(string a, string b) {
		int A = a.size();
		int B = b.size();
		while (A > B) {
			a = '0' + a;
			++A;
		}
		while (A < B) {
			b = '0' + b;
			++B;
		}
		for (int j = a.size() - 1; j > 0; --j) {
			a[j] = a[j] - '0' + b[j];
			if (a[j] >= '2') {
				a[j] = (a[j] - '0') % 2 + '0';
				a[j - 1] = a[j - 1] + 1;
			}
		}
		a[0] = a[0] - '0' + b[0];
		if (a[0] >= '2') {
			a[0] = (a[0] - '0') % 2 + '2';
			a = '1' + a;
		}
		return a;
	}
};


int main() {
	string a = "1010101010101";
	string b = "111110";
	Sloution a1;
	string c = a1.sumTwo(a,b);
	cout <<" c "<< endl;
	system("Pause");
}