// ����bot.cpp : �������̨Ӧ�ó������ڵ㡣
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
	cout << "�����ǣ�";
	cin >> part1 ;
	cout << "��Ҫ����";
	cin >> part2;
	cout << "Ӧ������";
	cin >> part3;
	cout << "�ô��ǣ�";
	cin >> part4;
	cout << "�����ࣺ";
	cin >> part5;
	cout << "����ʲô��";
	cin >> part6;
	system("cls");
	cout << "��������һ���µ�" << part1<<",Ҳ�����ǲ����ż�"<<part2<<endl;
	cout << "����Ӧ��" << part3 << "��������������" << part4 << endl;
	cout << "����δ��������" << part5 << endl;
	cout << "���⣬���Ǹ��߼���" << part6<<endl;
	system("pause");
    return 0;
}

