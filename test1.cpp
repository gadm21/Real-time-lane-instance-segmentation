#include <bits/stdc++.h>
using namespace std;

void f(int i){
    cout<<"hello"<<endl;
}

int main(){
    void (*foo)(int);
    foo= &f;
    foo(2);
    (*foo)(2);
}