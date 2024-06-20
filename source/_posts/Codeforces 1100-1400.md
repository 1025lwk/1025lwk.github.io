---
title: Codeforces 1100-1400
date: 2024/6/12 21:06:41
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



## 题目1：https://codeforces.com/contest/1845/problem/C

### 主要思路：对于l和r的每一位进行暴搜，cnt数组记录，如果每一位存在则t右移，当t==m时不符合条件输出NO，反之输出YES。

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
 
int a[N],cnt[10];
int T,n,m,k;
 
 
bool check(int l,int r)
{
    for(int i=l;i<=r;i++)
        if(!cnt[i])
            return false;
	
    return true;
}
 
 
void slove()
{
    string s,l,r;
    cin>>s>>m>>l>>r;
    
    fill(cnt,cnt+10,0);
    
    int t=0;
    for(int i=0;i<s.size();i++){
        int k=s[i]-'0';
        cnt[k]++;
    	
        int x=l[t]-'0',y=r[t]-'0';
        if(check(x,y)){
            fill(cnt,cnt+10,0);
            t++;
        }
    }
    
    if(t==m) cout<<"NO"<<'\n';
    else cout<<"YES"<<'\n';
}
 
 
signed main()
{
    ios::sync_with_stdio(false);
	
    cin>>T;
	
    while(T--) slove();
	
    return 0;
}
```



## 题目2：https://codeforces.com/problemset/problem/1794/C

### 主要思路：先遍历子序列的终点，而后从后往前二分查找起点，再把子序列个数累加即可。


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
    cin>>n;
    for(int i=0;i<n;i++) cin>>a[i];
	
    int res=1;
    for(int i=0;i<n;i++){
        int l=0,r=i;
        while(l<r){
            int mid=l+r>>1;
            if(a[mid]>=i-mid+1) r=mid;
            else l=mid+1;
        }
        res=max(res,i-l+1);
        cout<<res<<' ';
    }
	
    cout<<'\n';
}
 
 
signed main()
{
    ios::sync_with_stdio(false);
	
    cin>>T;
	
    while(T--) slove();
	
    return 0;
}
```



## 题目3：https://codeforces.com/problemset/problem/1692/F

### 主要思路：由于数据量2*10^5直接暴力枚举肯定会超时，换个方向枚举，先把每一个元素的个位存进来，而后暴力枚举三位相加取余10后符合的情况，而后判断是否存在就可以了。

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
 
int cnt[N];
int T,n,m,k;
 
 
int get(int i,int j,int k){
    return (i==j)+(i==j&&j==k)+1;
}

 
void slove()
{
    fill(cnt,cnt+20,0);
	
    cin>>n;
    for(int i=0;i<n;i++){
        int x;
        cin>>x;
        cnt[x%10]++;
    }
	
    for(int i=0;i<10;i++)
        for(int j=0;j<10;j++)
            for(int k=0;k<10;k++)
                if((i+j+k)%10==3){
                    cnt[i]--,cnt[j]--,cnt[k]--;
                    if(cnt[i]>=0&&cnt[j]>=0&&cnt[k]>=0){
                        cout<<"YES"<<endl;
                        return;
                    }
                    cnt[i]++,cnt[j]++,cnt[k]++;
                }
				
    cout<<"NO"<<'\n';
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

