// 三联bot.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include<iostream>
using namespace std;

int main()
{
	char part1[50] = { 0 };
	char part2[50] = { 0 };
	char part3[50] = { 0 };
	char part4[50] = { 0 };
	char part5[50] = { 0 };
	char part6[50] = { 0 };
	cout << "主体是：";
	cin >> part1 ;
	cout << "不要做：";
	cin >> part2;
	cout << "应该做：";
	cin >> part3;
	cout << "好处是：";
	cin >> part4;
	cout << "让人类：";
	cin >> part5;
	cout << "这是什么：";
	cin >> part6;
	system("cls");
	cout << "对于这样一种新的" << part1<<",也许我们不必着急"<<part2<<endl;
	cout << "而是应该" << part3 << "。起码这样可以" << part4 << endl;
	cout << "好让未来的人类" << part5 << endl;
	cout << "而这，才是更高级的" << part6<<endl;
	system("pause");
    return 0;
}

