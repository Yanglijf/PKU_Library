## 1. 题目

### 02733: 判断闰年

思路：根据题目条件直接判断

时间：15分钟左右

代码 python

```python
a = int(input())
if (a % 4 == 0 and a % 100 != 0) or (a % 400 == 0 and a % 3200 != 0):
    print("Y")
else:
    print("N")
```

代码运行截图

![](C:\Users\yangljf\Pictures\Screenshots\屏幕截图 2024-09-12 223842.png)

### 02750: 鸡兔同笼

思路：按照mod 4计算

时间：10分钟左右

代码 python

```python
a = int(input())
if a % 4 == 0:
    b = a/4
    c = a/2
elif a % 4 ==2:
    b = (a+2)/4
    c = a/2
else:
    b = 0
    c = 0

print(str(int(b)) + ' ' + str(int(c)))
```

代码运行截图

![](C:\Users\yangljf\Pictures\Screenshots\屏幕截图 2024-09-12 224231.png)

### 50A. Domino piling

思路：直接算

时间：5分钟左右

代码 python

```python
m, n = map(int, input().split())
print(int(m*n/2))
```

代码运行截图

![](C:\Users\yangljf\Pictures\Screenshots\屏幕截图 2024-09-12 225610.png)

### 1A. Theatre Square

思路：先比较长和宽的大小再输出最优解

时间：10分钟左右

代码 python

```python
n, m, i = map(int, input().split())
import math
a = math.ceil(n/i)
b = math.ceil(m/i)
x = a*b
print(x)
```

代码运行截图

![](C:\Users\yangljf\Pictures\Screenshots\屏幕截图 2024-09-12 225659.png)

### 112A. Petya and Strings

思路：将字母对应成数字进行比较

时间：20分钟左右

代码 python

```python
A = input().lower()
B = input().lower()
M = []
N = []
for x in range(len(A)):
    M.append(ord(A[x]))
    N.append(ord(B[x]))
if all(M[i] == N[i] for i in range(len(A))):
    print(0)
else:
    for i in range(len(A)):
        if M[i] > N[i]:
            print(1)
            break
        elif M[i] < N[i]:
            print(-1)
            break
        else:continue
```

代码运行截图

![](C:\Users\yangljf\Pictures\Screenshots\屏幕截图 2024-09-12 225835.png)

### 231A. Team

思路：数 1 的个数

时间：15分钟左右

代码 python

```python
n = int(input())
x = 0
for i in range(0, n):
    p = list(map(int, input().split()))
    if p.count(1) >= 2:
        x += 1
    else:
        x = x
print(x)
```

代码运行截图

![](C:\Users\yangljf\Pictures\Screenshots\屏幕截图 2024-09-12 225957.png)

## 2. 学习总结和收获

虽然是零基础起手，不过经过大量做题，边做边学，并且依靠gpt的帮助，对于本次作业难度的题目已经可以很快解决。

在加练oj和cf的题目时，发现自己在优化代码，缩短计算时间的方面还需要进一步提高。