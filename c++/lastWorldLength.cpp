#include "stdafx.h"
#include<iostream>
#include<vector>
#include<map>
#include <algorithm>
#include<algorithm>

using namespace std;
class Sloution {
public:
	int search(string str) {
		int a = 0;
		if (str.length() == 0) {
			return 0;
		}
		for (int i = str.length() - 1; i > 0; i--) {
			if (str[i] != ' ') {
				a++;
			}
			else {
				break;
			}
		}
		return a;
	}
	
};


int main() {
	string str = "hello world";
	Sloution a1;
	int c = a1.search(str);
	cout << c << endl;
	system("Pause");
}