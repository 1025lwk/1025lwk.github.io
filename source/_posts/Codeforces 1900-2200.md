---
title: Codeforces 1900-2200
date: 2024/6/6 21:06:41
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



## 题目一：[Problem - 1925D - Codeforces](https://codeforces.com/problemset/problem/1925/D)



### 介绍一种二项式组合数方法：


#### 众所周知，$期望 = 概率 * 值$。
#### 我们设期望为 $E$，总概率为 $p$，总值为 $sum$。则：
$$
E=p\times sum
$$

#### 由于 $k$ 的次数是固定的，则设
$$
s=\sum_{i=1}^m{f_i}
$$

#### 且 $s$ 每回合每个固定增加 $1$ （类似于等差数列求和的过程），则有：
$$
sum\gets s+sum 
$$
$$
s\gets s+m
$$

#### 解决完 $sum$ 后，我们还剩下 $p$ 没有搞定，那么总概率 $p$ 怎么求呢？

#### 因为每一个个体选中的概率是相等的，且都为
$$
\frac{1}{C_{n}^{2}}=\frac{2}{n(n-1)}
$$

#### 所以我们可以先设选中的概率为 $x$，没被选中的概率为 $y$。

#### 根据二项式定理得：
$$
p=\sum_{i=1}^k{C_{k}^{i}}x^iy^{k-i}
$$

$$
x=\frac{2}{n(n-1)},y=1-x
$$

#### 因为事件是独立的，每个值对应着对应的概率，则总式为：

$$
sum_i\gets s+sum_{i-1}
$$

$$
\sum_{i=1}^k{E} \gets C_{k}^{i}x^iy^{k-i}\times sum_i
$$

$$
s\gets s+m
$$

#### 最后我们把公式实现一下就搞定了，时间复杂度为：
$$
O(m+k\log mod)
$$

#### 注意：有数据点当 $n$ 为 100000 时，数据会爆 longlong，所以算 $x$ 时先取模（我就是因为这个而 wa6 了 ~T_T~）。




### MainCode：
```cpp
void solve()
{
    cin>>n>>m>>k;
    for(int i=1;i<=m;i++) cin>>a[i]>>b[i]>>f[i];
    
    int s=0;
    for(int i=1;i<=m;i++) s=(s+f[i])%mod;
    
    int sum=0,res=0,p=(n-1)*n/2%mod;   //p记得取模
    int x=qmi(p,mod-2),y=(1-x+mod)%mod;  
    for(int i=1;i<=k;i++){
        sum=(s+sum)%mod;
        res=(res+C(k,i)*qmi(x,i)%mod*qmi(y,k-i)%mod*sum%mod)%mod;
        s=(s+m)%mod;
    }
    cout<<res<<'\n';
}
```

---
