
#include<iostream>
#include<vector>
#include<algorithm>
using namespace std;

struct ListNode {
	int val;
	ListNode *next;
	ListNode(int x):val(x),next(NULL){}
};
class solution {
public:
	ListNode* mergeTwoLists(ListNode* L1, ListNode* L2) {
		vector<int> vec;
		ListNode* p1 = l1;
		ListNode* p2 = l2;
		if (p1 == NULL) {
			return p2;
		}
		if (p2 == NULL) {
			return p1;
		}

		while (p1 != NULL) {
			vec.push_back(p1->val);
			p1 = p1->next;
		}
		while (p2 != NULL) {
			vec.push_back(p2->val);
			p2 = p2->next;
		}
		sort(vec.begin(), vec.end());
		p1 = l1;
		for (int i = 0; i < vec.size(); i++) {
			if (p1 == NULL) {
				p1 = l2;
			
			}
			p1->val = vec[i];
			p1 = p1->next;
		}
		return l1;
	}
};