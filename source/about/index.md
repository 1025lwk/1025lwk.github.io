---
title: 介绍
date: 2023-02-02 11:24:09
layout: "about"

---



# 个人博客



### 介绍

* 喜欢用cin和cout的程序猿小白(,,Ծ‸Ծ,, )
* 致力于更有效的学习！！！
* 博客记录学习的过程碎片~




###  我的初始板子(๑¯ω¯๑)：

```c++
#include <bits/stdc++.h>

#define x first
#define y second

#define int long long

using namespace std;

typedef pair<int,int> PII;

typedef long long LL;

const int N=1e6+10;

int a[N],b[N];
int T,n,m,k;   


void slove()
{
    //此处解题
}


signed main()
{
    ios::sync_with_stdio(false);
	
    cin.tie(nullptr),cout.tie(nullptr);
	
    cin>>T;
	
    while(T--) slove();
	
    return 0;
}
```

  

### 使用说明

1. 转发注明出处！！！
2. 自己使用的模板，望大家批评指正~
3. 可供学习参考。



### 主要功能

1. Fork 本仓库；
2. 使用 master 分支；
3. 提交参考代码；
4. 有问题反馈评论；
5. 不得发表不正当言论！！！
6. 还有好多内容没有完善。。。敬请谅解~

---



### 提供git上传仓库教程：

* git下载地址：https://npm.taobao.org/mirrors/git-for-windows/

  

### 基于git搭建hexo博客教程

* 1.右键打开Git Bash，输入如下命令，配置git操作的用户名、邮箱：
    $ git config --global user.name "你的名字或昵称"
    $ git config --global user.email "你的邮箱"
* 2.配置ssh公钥：
    在Git Bash中输入如下命令，生成SSH key：
    $ ssh-keygen -t rsa -C "你的邮箱"
* 3.输入如下第1行命令，打印刚才生成的SSH key：
    $ cat ~/.ssh/id_rsa.pub
* 4.主页右上角 「个人设置」->「安全设置」->「SSH公钥」->「添加公钥」 ，复制生成的 public key，添加到当前账户中。

* 5.测试SSH key是否配置ok：
   $ ssh -T git@github.com
   Hi XXX! You've successfully authenticated, but GITEE.COM does not provide shell access.

### 上传文件操作
* $ git add .                          #将当前目录所有文件添加到git暂存区
* $ git commit -m "my commit"          #提交并备注提交信息
* $ git push                           #将本地提交推送到远程仓库


### 下载文件操作
* 打开==自己账号==下的仓库，点击 “克隆/下载” 按钮，选择 “SSH”, 点击“复制”
* $ git clone git@github.com

### hexo 部署命令：
* $ cd 你的博客文件夹
* $ hexo cl && hexo g && hexo s 
* $ hexo d （部署）  
