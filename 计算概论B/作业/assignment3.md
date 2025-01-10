# Assign #3: Oct Mock Exam暨选做题目满百

Updated 1537 GMT+8 Oct 10, 2024

2024 fall, Complied by Hongfei Yan==（请改为同学的姓名、院系）==



**说明：**

1）Oct⽉考： AC6==（请改为同学的通过数）== 。考试题⽬都在“题库（包括计概、数算题目）”⾥⾯，按照数字题号能找到，可以重新提交。作业中提交⾃⼰最满意版本的代码和截图。

2）请把每个题目解题思路（可选），源码Python, 或者C++/C（已经在Codeforces/Openjudge上AC），截图（包含Accepted, 学号），填写到下面作业模版中（推荐使用 typora https://typoraio.cn ，或者用word）。AC 或者没有AC，都请标上每个题目大致花费时间。

3）提交时候先提交pdf文件，再把md或者doc文件上传到右侧“作业评论”。Canvas需要有同学清晰头像、提交文件有pdf、作业评论有md或者doc。

4）如果不能在截止前提交作业，请写明原因。



## 1. 题目

### E28674:《黑神话：悟空》之加密

http://cs101.openjudge.cn/practice/28674/



思路：使用ord和chr函数完成字母和数字的对应，再通过模26的计算找出解后的字母

用时：10分钟左右

代码

```python
k = int(input())
s = input()
d = ''
for i in range(len(s)):
    t = ord(s[i])
    if 65<= t <= 90:
        t = (t-65-k)%26 + 65
        d += str(chr(t))
    else:
        t = (t-97-k)%26 + 97
        d += str(chr(t))
print(d)

```

（python）

代码运行截图 ==（至少包含有"Accepted"）==

![](C:\Users\yangljf\Pictures\Screenshots\屏幕截图 2024-10-12 093848.png)



### E28691: 字符串中的整数求和

http://cs101.openjudge.cn/practice/28691/



思路：提取数字并计算

用时：2分钟

代码

```python
a,b = input().split()
x = int(a[0:2])
y = int(b[0:2])
print(x+y)

```

（python）

代码运行截图 ==（至少包含有"Accepted"）==

![](C:\Users\yangljf\Pictures\Screenshots\屏幕截图 2024-10-12 094132.png)



### M28664: 验证身份证号

http://cs101.openjudge.cn/practice/28664/



思路：将身份证号的每一位分别提取出来，乘以对应的系数并求和，模11进行检测

用时：15分钟左右

代码

```python
n = int(input())
l = [7,9,10,5,8,4,2,1,6,3,7,9,10,5,8,4,2]
q = ['1','0','X','9','8','7','6','5','4','3','2']
for _ in range(n):
    s = input()
    k = 0
    for i in range(17):
        k += int(s[i])*l[i]
    r = k % 11
    if str(q[r]) == s[17]:
        print('YES')
    else:
        print('NO')

```

（python）

代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![](C:\Users\yangljf\Pictures\Screenshots\屏幕截图 2024-10-12 094210.png)



### M28678: 角谷猜想

http://cs101.openjudge.cn/practice/28678/



思路：判断奇偶进行相应计算并逐行输出

用时：5分钟左右

代码

```python
n = int(input())
while n != 1:
    if n % 2 == 0:
        print(f'{n}/2={int(n/2)}')
        n = n//2
    else:
        print(f'{n}*3+1={n*3+1}')
        n = n*3+1
print('End')
```

（python）

代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![](C:\Users\yangljf\Pictures\Screenshots\屏幕截图 2024-10-12 094718.png)



### M28700: 罗马数字与整数的转换

http://cs101.openjudge.cn/practice/28700/



思路：将罗马数字与正整数相对应，并将特殊情况列出单独处理

用时：30分钟左右

##### 代码

```python
d = {'I':1,'V':5,'X':10,'L':50,'C':100,'D':500,'M':1000}
s = input()
if s[0] not in d:
    s = int(s)
    t = ''
    d1 = s // 1000
    t += d1*'M'
    s -= d1*1000
    d2= s // 900
    t += d2*'CM'
    s -= d2*900
    d3 = s // 500
    t += d3*'D'
    s -= d3*500
    d4 = s // 400
    t += d4*'CD'
    s -= d4*400
    d5 = s // 100
    t += d5*'C'
    s -= d5*100
    d6 = s // 90
    t += d6*'XC'
    s -= d6*90
    d7 = s // 50
    t += d7*'L'
    s -= d7*50
    d8 = s // 40
    t += d8*'XL'
    s -= d8*500
    d9 = s // 10
    t += d9*'X'
    s -= d9*10
    d10 = s // 9
    t += d10*'IX'
    s -= d10*9
    d11 = s // 5
    t += d11*'V'
    s -= d11*5
    d12 = s // 4
    t += d12*'IV'
    s -= d12*4
    t += s*'I'
    print(t)
else:
    t = 0
    for i in range(len(s)-1):
        if s[i] == 'I' and (s[i+1] == 'V' or s[i+1] == 'X'):
            t -= d[s[i]]
        elif s[i] == 'X' and (s[i+1] == 'C' or s[i+1] == 'L'):
            t -= d[s[i]]
        elif s[i] == 'C' and (s[i+1] == 'D' or s[i+1] == 'M'):
            t -= d[s[i]]
        else:
            t += d[s[i]]
    t += d[s[len(s)-1]]
    print(int(t))

```

（python）

代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![](C:\Users\yangljf\Pictures\屏幕截图_12-10-2024_95257_cs101.openjudge.cn.jpeg)



### *T25353: 排队 （选做）

http://cs101.openjudge.cn/practice/25353/



思路：判断每一个数最多能到达的位置并按升序输出

用时：截至提交作业是仍然超时未通过

代码

```python


```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==





## 2. 学习总结和收获

==如果作业题目简单，有否额外练习题目，比如：OJ“计概2024fall每日选做”、CF、LeetCode、洛谷等网站题目。==

前4道题较简单，但需要注意细节的处理（如角谷猜想最后需要输出End）

罗马数字排列中没有几十想到运用字典和元组建立起对应关系，导致代码冗长，蛮力求解时WA多次，并且耗时较长

排队这道题明白了思路，但还是TLE，目前还在优化



总而言之，语法的熟练度还是有些不够，仍然有待提高









