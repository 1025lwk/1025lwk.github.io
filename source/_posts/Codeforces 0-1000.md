---
title: Codeforces 0-1000
date: 2024/6/14 21:06:41
categories:
  - Codeforces
tags:
  - 题解
cover: https://th.bing.com/th/id/R.8764d162d1e9be6b2cf6d348e2da99f0?rik=TLZjyw4Nspa%2b0w&riu=http%3a%2f%2fpic.616pic.com%2fys_bnew_img%2f00%2f62%2f36%2fr9dfcyoyjR.jpg&ehk=SdJuWe8fxNWlX58TeKImDSWatngZpwGh6ann2DJ%2fXN0%3d&risl=&pid=ImgRaw&r=0
---


## 首先声明！！！

---
* 1.题解为本人原作，如有使用注明出处。
* 2.如有改进地方欢迎批评指正~

---



## 题目1：https://codeforces.com/contest/1845/problem/A

### 主要思路：先判断不满足条件的特殊值，然后偶数直接构造2，奇数最后一位构造3。
```c++
#include <iostream>
#include <algorithm>
#include <cstring>
 
#define x first
#define y second
#define int long long
 
using namespace std;
 
typedef long long LL;
 
typedef pair<int,int> PII;
 
const int N=1e6+10;
 
int a[N],b[N];
int T,n,x,k;
 
 
void slove()
{
    cin>>n>>k>>x;
    if(x==1&&n%k==1&&k==2||k==1) cout<<"NO"<<'\n';
    else if(x!=1){
        cout<<"YES"<<'\n';
        cout<<n<<'\n';
        for(int i=1;i<=n;i++) cout<<1<<' ';
        cout<<'\n';
    }
    else{
        cout<<"YES"<<'\n';
        cout<<n/2<<'\n';
        for(int i=1;i<=n/2-1;i++) cout<<2<<' ';
        if(n%2) cout<<3<<' ';
        else cout<<2<<' ';
        cout<<'\n';
    }
}
 
 
signed main()
{
    ios::sync_with_stdio(false);
	
    cin>>T;
	
    while(T--) slove();
	
    return 0;
}
```



## 题目2：https://codeforces.com/contest/1845/problem/B

### 主要思路：分别判断横纵坐标是否位于起点的同一侧，符合条件即累加其与起点之间的最小值，答案为累加后的结果。
```c++
#include <iostream>
#include <algorithm>
#include <cstring>
 
#define x first
#define y second
#define int long long
 
using namespace std;
 
typedef long long LL;
 
typedef pair<int,int> PII;
 
const int N=1e6+10;
 
int a[N],b[N];
int T,n,m,k;
 
 
void slove()
{
    int xa,xb,xc,ya,yb,yc;
    cin>>xa>>ya>>xb>>yb>>xc>>yc;
    
    int sum=1;
    if(xb<xa&&xc<xa||xb>xa&&xc>xa) sum+=min(abs(xa-xb),abs(xa-xc));
    if(yb<ya&&yc<ya||yb>ya&&yc>ya) sum+=min(abs(ya-yb),abs(ya-yc));
    cout<<sum<<'\n';
}
 
 
signed main()
{
    ios::sync_with_stdio(false);
	
    cin>>T;
	
    while(T--) slove();
	
    return 0;
}
```



## 题目3：https://codeforces.com/problemset/problem/1744/C

### 主要思路：首先把输入字符串复制一倍加在原来的字符串中（为了解决环的问题），而后双指针从开头字符c判断，不断取最小值，如果开头字符为g则输出0，否则输出最小值。


```c++
#include <iostream>
#include <algorithm>
#include <cstring>

#define x first
#define y second
#define int long long

using namespace std;

typedef long long LL;

typedef pair<int,int> PII;

const int N=1e6+10;

int a[N],b[N];
int T,n,m,k;


void slove()
{
    char c;
    string s;
    cin>>n>>c>>s;
	
    s+=s;
    int sum=0;
    for(int i=0;i<2*n;i++){
        if(s[i]==c){
            int j=i+1;
            while(j<2*n&&s[j]!='g') j++;
            sum=max(sum,j-i);
            i=j-1;
        }
    }
	
    if(c=='g') sum=0;
    cout<<sum<<'\n';
}


signed main()
{
    ios::sync_with_stdio(false);
	
    cin>>T;
	
    while(T--) slove();
	
    return 0;
}
```

---

