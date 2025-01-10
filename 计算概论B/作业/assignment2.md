# Assignment #2: 语法练习

Updated 0126 GMT+8 Sep 24, 2024

2024 fall, Complied by ==同学的姓名、院系==



**说明：**

1）请把每个题目解题思路（可选），源码Python, 或者C++（已经在Codeforces/Openjudge上AC），截图（包含Accepted），填写到下面作业模版中（推荐使用 typora https://typoraio.cn ，或者用word）。AC 或者没有AC，都请标上每个题目大致花费时间。

3）课程网站是Canvas平台, https://pku.instructure.com, 学校通知9月19日导入选课名单后启用。**作业写好后，保留在自己手中，待9月20日提交。**

提交时候先提交pdf文件，再把md或者doc文件上传到右侧“作业评论”。Canvas需要有同学清晰头像、提交文件有pdf、"作业评论"区有上传的md或者doc附件。

4）如果不能在截止前提交作业，请写明原因。



## 1. 题目

### 263A. Beautiful Matrix

https://codeforces.com/problemset/problem/263/A



思路：列表嵌套形成矩阵，通过循环找到1的位置，最后计算1到中心的距离

用时：10分钟左右

##### 代码

```python
A = list(map(int,input().split()))
B = list(map(int,input().split()))
C = list(map(int,input().split()))
D = list(map(int,input().split()))
E = list(map(int,input().split()))
l = [A, B, C, D, E]
for x in range(5):
    for y in range(5):
        if l[x][y] == 1:
            print(abs(x-2) + abs(y-2))
        else:continue
```

（python）

代码运行截图 ==（至少包含有"Accepted"）==

![](C:\Users\yangljf\Pictures\Screenshots\屏幕截图 2024-09-30 093128.png)





### 1328A. Divisibility Problem

https://codeforces.com/problemset/problem/1328/A



思路：先计算a除以b的上取整k，k*a-b即为所求

用时：5分钟以内

##### 代码

```python
t = int(input())
for _ in range(t):
    a, b = map(int, input().split())
    k = -((-a)//b)
    print(k*b-a)
```

（python）

代码运行截图 ==（至少包含有"Accepted"）==

![](C:\Users\yangljf\Pictures\Screenshots\屏幕截图 2024-09-30 093538.png)





### 427A. Police Recruits

https://codeforces.com/problemset/problem/427/A



思路：同时更新警察数量和案件数量

用时：7分钟左右

##### 代码

```python
n = int(input())
l = list(map(int, input().split()))
p = 0
c = 0
for i in range(n):
    if l[i] != -1:
        p += l[i]
    else:
        if p > 0:
            p -= 1
        else:
            c += 1
print(c)
```

（python）

代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![](C:\Users\yangljf\Pictures\Screenshots\屏幕截图 2024-09-30 093950.png)





### 02808: 校门外的树

http://cs101.openjudge.cn/practice/02808/



思路：通过集合进行操作，不断从原几何中挖去占用的集合

用时：20分钟左右

##### 代码

```python
l, m = map(int, input().split())
s = set(range(l + 1))
for _ in range(m):
    x, y = map(int, input().split())
    s -= set(range(x, y + 1))
print(len(s)) 
```

（python）

代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![](C:\Users\yangljf\Pictures\Screenshots\屏幕截图 2024-09-30 094238.png)





### sy60: 水仙花数II

https://sunnywhy.com/sfbj/3/1/60



思路：直接进行计算

用时：5分钟左右

##### 代码

```python
def t(x):
    y = str(x)
    x = int(x)
    if x == int(y[0])**3 + int(y[1])**3 + int(y[2])**3:
        return True
    return False
a, b = map(int, input().split())
l = []
for n in range(a, b+1):
    if t(n) is True:
        l.append(n)
if len(l) == 0:
    print('NO')
else:
    for i in range(len(l)-1):
        print(l[i],end=' ')
    print(l[len(l)-1])
```

（python）

代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![](C:\Users\yangljf\Pictures\Screenshots\屏幕截图 2024-09-30 124721.png)



### 01922: Ride to School

http://cs101.openjudge.cn/practice/01922/



思路：该问题只需转化为这些骑手最快的到达时间

用时：20分钟左右

##### 代码

```python
import math
d = 4.5
while True:
    n = int(input())
    if n == 0:
        break
    w = float('inf')
    for _ in range(n):
        v, t = map(int, input().split())
        v = v / 3600
        if t >= 0:
            if d / v + t < w:
                w = d / v + t
    print(math.ceil(w))# 

```

（python）

代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![](C:\Users\yangljf\Pictures\Screenshots\屏幕截图 2024-09-30 132347.png)





## 2. 学习总结和收获

==如果作业题目简单，有否额外练习题目，比如：OJ“计概2024fall每日选做”、CF、LeetCode、洛谷等网站题目。==

这次作业题目整体感觉不是很难，但部分语句书写还是需要一定时间的思考。

在加练其他网站的题目时，发现有些题目虽然叙述较繁琐，但进一步思考后，发现其内核还是比较简单的。

在做题时，发现时常会错误处理特殊情况的数据而导致错误，这一点仍需加强。

总的来说，加练更多题目不仅巩固了我的语法知识，也让我意识到了自己的不足之处。我计划在今后的学习中更加注重细节，同时培养更全面的问题分析能力。



