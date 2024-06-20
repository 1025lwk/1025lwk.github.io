---
title: Codeforces 1500-1800
date: 2024/6/10 21:06:41
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




## 题目1：https://codeforces.com/problemset/problem/1843/E

### 主要思路：经典二分+前缀和，二分搜索结果，check函数里先前缀和预处理a[i]，而后遍历判断区间1的个数是否超过区间长度的一半。

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
 
int a[N],d[N],l[N],r[N],s[N];
int T,n,m,q;
 
 
bool check(int x)
{
    for(int i=1;i<=x;i++) a[d[i]]=1;
    for(int i=1;i<=n;i++) s[i]=s[i-1]+a[i];
	
    for(int i=1;i<=m;i++)
        if(2*(s[r[i]]-s[l[i]-1])>r[i]-l[i]+1)
            return true;
			
    return false;
}
 
 
void slove()
{
    cin>>n>>m;
    for(int i=1;i<=m;i++) cin>>l[i]>>r[i];
    cin>>q;
    for(int i=1;i<=q;i++) cin>>d[i];
	
    fill(a,a+n+1,0);
    fill(s,s+n+1,0);
	
    int x=1,y=q+1;
    while(x<y){
        int mid=x+y>>1;
        fill(a,a+n+1,0);
        if(check(mid)) y=mid;
        else x=mid+1;
    }
	
    if(y==q+1) y=-1;
    cout<<y<<'\n';
}
 
 
signed main()
{
    ios::sync_with_stdio(false);
	
    cin>>T;
	
    while(T--) slove();
	
    return 0;
}
```



## 题目2：https://codeforces.com/problemset/problem/1833/F

### 主要思路：先用map记录一下啊a[i]出现的次数，而后求连续m个a[i]+1=a[i+1]的组合数个数，求组合数时用到快速幂求逆元。

```c++
#include <iostream>
#include <algorithm>
#include <cstring>
#include <map>

#define x first
#define y second

using namespace std;

typedef long long LL;

typedef pair<int,int> PII;

const int N=1e6+10,mod=1e9+7;

LL a[N],b[N];
int T,n,m,k;


LL qmi(LL a,LL b,LL p)
{
    LL sum=1;
    while(b){
        if(b&1) sum=(LL)sum*a%p;
        a=(LL)a*a%p;
        b>>=1;
    }
    return sum%p;
}


void slove()
{
    cin>>n>>k;
	
    int cnt=0;
    map<int,int> mp;
    for(int i=0;i<n;i++){
        int x;
        cin>>x;
        if(!mp[x]) a[cnt++]=x;
        mp[x]++;
    }
    
    sort(a,a+cnt);
	
    LL sum=1,res=0;
    if(a[k-1]-a[0]==k-1){
        for(int i=0;i<k;i++) 
            sum=sum*mp[a[i]]%mod;
        res+=sum%mod;
    }
	
    for(int i=1;i<cnt-k+1;i++){
        if(a[i+k-1]-a[i]==k-1){
            sum=(LL)sum*qmi(mp[a[i-1]],mod-2,mod)%mod;
            sum=(LL)sum*mp[a[i+k-1]]%mod;
            res+=sum%mod;
        }
    }
	
    cout<<res%mod<<'\n';
}


int main()
{
    ios::sync_with_stdio(false);
	
    cin>>T;
	
    while(T--) slove();
	
    return 0;
}
```



## 题目3：https://codeforces.com/problemset/problem/1833/E

### 主要思路：主要思路是判断环的最小个数和最多个数，dfs记录起点和走过的点，如果遇到环的大小为2即ans++，如果回到起点即res++，最后统计个数即可。

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

int a[N];
bool st[N];
int T,n,m;
int res,ans;


void dfs(int x,int y,int z,int k,int cnt)
{
    if(st[x]){
        if(x==z) ans++;
        else if(x==k) res++;
        return;
    }
	
    st[x]=true;
    z=y,y=x,x=a[x];
    dfs(x,y,z,k,cnt+1);
}


void slove()
{
    cin>>n;
    for(int i=1;i<=n;i++) cin>>a[i];
	
    fill(st,st+n+1,false);
	
    res=0,ans=0;
    for(int i=1;i<=n;i++) 
        if(!st[i])
            dfs(i,i,i,i,0);
	
    cout<<res+(ans>0)<<' '<<res+ans<<endl;
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

