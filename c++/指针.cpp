#include"stdafx.h"
#include<iostream>
#include<vector>
#include<algorithm>

using namespace std;

class A {
public:
	int a = 1;
};
int main() {
	A b;
	A* p = &b;
	cout << p->a<< endl;
	system("Pause");
} 