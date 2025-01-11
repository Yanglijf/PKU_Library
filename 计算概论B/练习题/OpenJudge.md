## 19960:恩尼格玛

[OpenJudge - 19960:恩尼格玛](http://cs101.openjudge.cn/practice/19960/)

##### 描述

Frank老师在计算概论的第一堂课上让同学们观看了“模仿游戏”这部电影。看完以后，Hitler同学对其中的恩尼格玛加密机产生了浓厚的兴趣。在网上查阅资料以后，他决定从简单的情况入手（输入文本只包含abcdef六个字母），用电脑实现一个恩尼格玛加密机。
加密机的原理如下（参照示意图）：

![img](http://media.openjudge.cn/images/upload/1576072105.png)

三个转子上都有线路，彼此相连；反射器将六个槽位两两相连；输入某一字母后经过左中右三个转子到达反射器，再返回，依次通过右中左三个转子，最后得到加密字母（全小写）。（如图中，输入b得到d）

注意：每输入一个字母，左边的转子都会向下转一位，左边的转子转了一圈以后中间的转子向下转一位，中间的转子转一圈以后最右边的转子向下转一位



##### 输入

前18行，每行2个数字，以空格隔开，表示转子进出的初始线路连接（1-6行是左转子，7-12行是中间转子，13-18行是右转子）
第19-21行，每行2个数字，以空格隔开，表示反射器中两两连接的情况
最后一行是待加密的字符串（只含有abcdef六个字母，全小写）

##### 输出

加密后的字符串（全小写）

##### 样例输入

```
1 6
2 3
3 5
4 4
5 2
6 1
1 3
2 1
3 4
4 2
5 6
6 5
1 5
2 1
3 4
4 6
5 3
6 2
1 6
2 3
4 5
abcdefabcdefabcdef
```

##### 样例输出

```
caecaeccbffefdabbc

解释：处理完第一个字符'a'以后，左转子变为（2 1；3 4；4 6；5 5；6 3；1 2），中间转子、右转子不变
```

```python
def rotate(l):
    return {k%6+1:v%6+1 for k,v in l.items()}

cha = ['a','b','c','d','e','f']
num = [1,2,3,4,5,6]
dct1 = dict(zip(cha,num))
dct2 = dict(zip(num,cha))
l1 = {}
l2 = {}
for _ in range(6):
    x,y = map(int, input().split())
    l1[x]=y
    l2[y]=x
m1 = {}
m2 = {}
for _ in range(6):
    x,y = map(int, input().split())
    m1[x]=y
    m2[y]=x
r1 = {}
r2 = {}
for _ in range(6):
    x,y = map(int, input().split())
    r1[x]=y
    r2[y]=x
ref = {}
for _ in range(3):
    x,y = map(int, input().split())
    ref[x] = y
    ref[y] = x
string = input()

result = ''
i = 0
for x in string:
    s = dct2[l2[m2[r2[ref[r1[m1[l1[dct1[x]]]]]]]]]
    result += s
    l1 = rotate(l1)
    l2 = rotate(l2)
    i += 1
    if i%6 == 0:
        m1 = rotate(m1)
        m2 = rotate(m2)
    if i%36 == 0:
        r1 = rotate(r1)
        r2 = rotate(r2)
print(result)
```

## 16532:北大杯台球比赛

[OpenJudge - 16532:北大杯台球比赛](http://cs101.openjudge.cn/practice/16532/)

##### 描述

北大杯台球比赛进入白热化的黑八大战环节，台球桌尺寸为17x6，以台球桌左上角建立坐标系如下图所示：



```
0——1——2——3——4——5——6——7——8——...——16——   x轴
|  |  |  |  |  |  |  |  |       |
1——.——.——.——.——.——.——.——.——...——.
|  |  |  |  |  |  |  |  |       |
2——.——.——.——.——.——.——.——.——...——.
|  |  |  |  |  |  |  |  |       |
3——.——.——.——.——.——.——.——.——...——.
|  |  |  |  |  |  |  |  |       |
4——.——.——.——.——.——.——.——.——...——.
|  |  |  |  |  |  |  |  |       |
5——.——.——.——.——.——.——.——.——...——.
|

y轴
```



  y轴

其中（0，0）（8，0）（16，0）（0，5）（8，5）（16，5）是台球桌上的六个入袋口，台球只有运动到这六个网格顶点时才能进球。
现在已知台球桌上仅剩下一个白色母球和一个黑球，并且知道他们的x,y坐标（均为整数）。击球时只能击打白色母球，且击球方向只有左上（-1，-1）、左下（-1，1）、右上（1，-1）、右下（1，1）四种方向。
击球后白色母球会沿击球方向运动，若碰到球桌壁则会发生反弹，若碰到黑球则会发生完全弹性碰撞导致动能完全传递（即白球静止，黑球获得白球的速度继续运动）。球向以上4个方向移动一次就会消耗一单位能量。 请计算最后的胜负情况。



##### 输入

输入为四行
第一行为白色母球的坐标
第二行为黑球的坐标
第三行为击球方向
第四行为击球力度，即施加给白球的初始动能是多少单位

输入保证两球的坐标落在球桌内，不会在球壁上,且不重复，保证击球方向为上述四方向之一，保证击球能量>=0

##### 输出

输出为一行
假如白球入库输出-1，黑球入库则输出1，无球入库则输出0

##### 样例输入

```
Sample1 Input:
1 1
2 2
-1 -1
10
Sample1 Output:
-1
```

##### 样例输出

```
Sample2 Input:
2 2
1 1
-1 -1
10
Sample2 Output:
1

Sample3 Input:
2 2
1 1
-1 -1
0
Sample3 Output:
0
```

```python
end = [(0,0),(8,0),(16,0),(0,5),(8,5),(16,5)]
x0, y0 = map(int, input().split())
x1, y1 = map(int, input().split())
dx, dy = map(int, input().split())
e = int(input())

t = 1
result = 0
while e > 0:
    e -= 1
    x0, y0 = x0+dx, y0+dy
    if (x0,y0) in end:
        result = -1
        break
    if (x0,y0) == (x1,y1):
        t += 1
    if x0 == 0 or x0 == 16:
        dx *= -1
    if y0 == 0 or y0 == 5:
        dy *= -1

print(result**t)
```

## 25566:CPU 调度

[OpenJudge - 提交状态](http://cs101.openjudge.cn/practice/solution/47691509/)

##### 描述

进程（Process）是运行中的程序。你平时使用计算机时打开的每一个不同的窗口，都对应着一个进程。一个进程通常需要进行两类操作：读写数据和计算。前者需要占用内存或硬盘中的某个文件，而后者需要独占所有进程共享的 CPU 资源。CPU 在每个时间周期内可以选择一个进程为其执行计算操作，这被称为 CPU 调度。假设现在有一组进程，每个进程 i 需要先累计占用 compute[i] 个 CPU 周期（不一定连续）进行计算，计算完成后需要 write[i] 个时间周期将结果写文件，随后进程结束。所有进程写的文件均不同，即写的过程可以同步进行。请你计算如何调度 CPU 能够使所有进程都结束的时间最早？



##### 样例解释

样例 1 和样例 2 的一种最优调度如下图所示，其中 x 代表占用 CPU 进行计算，〇 代表写文件，√ 代表该时刻进程结束。注意最优调度不一定唯一且例子所举的方法不一定与正确算法相关。

样例 1：

![img](http://media.openjudge.cn/images/upload/9375/1669784415.png)



样例 2：

![img](http://media.openjudge.cn/images/upload/2288/1669784443.png)

##### 输入

第一行是一个正整数 n (1 <= n <= 200)，表示进程的个数。

接下来 n 行，每行为空格分隔的 2 个正整数 compute[i] 和 write[i] ，表示每个进程 i 需要的 CPU 计算时间和写文件时间。

##### 输出

一个正整数，表示所有进程完结的最早时间。假设时间从 0 开始。

##### 样例输入

```
Sample Input1:
3
1 2
4 3
3 1

Sample Output1:
9
```

##### 样例输出

```
Sample Input1:
4
1 2
2 1
3 2
2 1

Sample Output2:
9
```

##### 提示

tags: greedy

```python
n = int(input())
cpu = []
for _ in range(n):
    c, w = map(int, input().split())
    cpu.append((c,w))
cpu.sort(key=lambda x:(-x[1],x[0]))
time = com = 0
for i in range(n):
    com += cpu[i][0]
    time = max(time,com + cpu[i][1])
print(time)
```



## 02811:熄灯问题

[OpenJudge - 02811:熄灯问题](http://cs101.openjudge.cn/practice/02811/)

##### 描述

有一个由按钮组成的矩阵，其中每行有6个按钮，共5行。每个按钮的位置上有一盏灯。当按下一个按钮后，该按钮以及周围位置(上边、下边、左边、右边)的灯都会改变一次。即，如果灯原来是点亮的，就会被熄灭；如果灯原来是熄灭的，则会被点亮。在矩阵角上的按钮改变3盏灯的状态；在矩阵边上的按钮改变4盏灯的状态；其他的按钮改变5盏灯的状态。

![img](http://media.openjudge.cn/images/2811_1.jpg)在上图中，左边矩阵中用X标记的按钮表示被按下，右边的矩阵表示灯状态的改变。对矩阵中的每盏灯设置一个初始状态。请你按按钮，直至每一盏等都熄灭。与一盏灯毗邻的多个按钮被按下时，一个操作会抵消另一次操作的结果。在下图中，第2行第3、5列的按钮都被按下，因此第2行、第4列的灯的状态就不改变。

![img](http://media.openjudge.cn/images/2811_2.jpg)请你写一个程序，确定需要按下哪些按钮，恰好使得所有的灯都熄灭。根据上面的规则，我们知道1）第2次按下同一个按钮时，将抵消第1次按下时所产生的结果。因此，每个按钮最多只需要按下一次；2）各个按钮被按下的顺序对最终的结果没有影响；3）对第1行中每盏点亮的灯，按下第2行对应的按钮，就可以熄灭第1行的全部灯。如此重复下去，可以熄灭第1、2、3、4行的全部灯。同样，按下第1、2、3、4、5列的按钮，可以熄灭前5列的灯。



##### 输入

5行组成，每一行包括6个数字（0或1）。相邻两个数字之间用单个空格隔开。0表示灯的初始状态是熄灭的，1表示灯的初始状态是点亮的。

##### 输出

5行组成，每一行包括6个数字（0或1）。相邻两个数字之间用单个空格隔开。其中的1表示需要把对应的按钮按下，0则表示不需要按对应的按钮。

##### 样例输入

```
0 1 1 0 1 0
1 0 0 1 1 1
0 0 1 0 0 1
1 0 0 1 0 1
0 1 1 1 0 0
```

##### 样例输出

```
1 0 1 0 0 1
1 1 0 1 0 1
0 0 1 0 1 1
1 0 0 1 0 0
0 1 0 0 0 0
```

```python
def in_matrix(x,y):
    if 0 <= x < 5 and 0 <= y < 6:
        return True
    return False

def change(x,y):
    changes = [(x,y)]
    for dx,dy in move:
        nx,ny = x+dx,y+dy
        if in_matrix(nx,ny):
            changes.append((nx,ny))
    return changes

def press(x,y,current):
    for nx,ny in change(x,y):
        current[nx][ny] = 1 - current[nx][ny]

def check(current):
    for l in current:
        if any(l):
            return False
    return True

def press_plan():
    for i in range(1<<6):
        presses = [[0] * 6 for _ in range(5)]
        ini_matrix = [l[:] for l in matrix]
        for c in range(6):
            if (i>>c) & 1:
                presses[0][c] = 1
                press(0,c,ini_matrix)
        for k in range(1, 5):
            for t in range(6):
                if ini_matrix[k-1][t] == 1:
                    presses[k][t] = 1
                    press(k,t,ini_matrix)
        if check(ini_matrix):
            return presses
    return None

move = [(0,1),(0,-1),(1,0),(-1,0)]
matrix = [list(map(int, input().split())) for _ in range(5)]
plan = press_plan()
for l in plan:
    print(*l)
```



## 04030:统计单词数

[OpenJudge - 04030:统计单词数](http://cs101.openjudge.cn/practice/04030/)

##### 描述

 一般的文本编辑器都有查找单词的功能，该功能可以快速定位特定单词在文章中的位置，有的还能统计出特定单词在文章中出现的次数。
现在，请你编程实现这一功能，具体要求是：给定一个单词，请你输出它在给定的文章中出现的次数和第一次出现的位置。注意：匹配单词时，不区分大小写，但要求完全匹配，即给定单词必须与文章中的某一独立单词在不区分大小写的情况下完全相同 （参见样例 1） ，如果给定单词仅是文章中某一单词的一部分则不算匹配（参见样例 2） 。

##### 输入

第 1 行为一个字符串，其中只含字母，表示给定单词；
第 2 行为一个字符串，其中只可能包含字母和空格，表示给定的文章。

##### 输出

只有一行， 如果在文章中找到给定单词则输出两个整数， 两个整数之间用一个空格隔开，分别是单词在文章中出现的次数和第一次出现的位置（即在文章中第一次出现时，单词首字母在文章中的位置，位置从 0 开始） ；如果单词在文章中没有出现，则直接输出一个整数-1。

##### 样例输入

```
样例 #1:
To 
to be or not to be is a question 

样例 #2:
to 
Did the Ottoman Empire lose its power at that time
```

##### 样例输出

```
样例 #1:
2 0

样例 #2:
-1
```

##### 提示

【输入输出样例 1 说明】
输出结果表示给定的单词 To 在文章中出现两次，第一次出现的位置为 0。

【输入输出样例 2 说明】
表示给定的单词 to 在文章中没有出现，输出整数-1。

【数据范围】
1 ≤单词长度≤10。
1 ≤文章长度≤1,000,000。

```python
target = ' ' + input().lower() + ' '
text = ' ' + input().lower() + ' '
cnt = (text.split()).count(target[1:-1])
if cnt:
    index = text.find(target)
    print(cnt,index)
else:
    print(-1)
```



## 27384:候选人追踪

[OpenJudge - 27384:候选人追踪](http://cs101.openjudge.cn/2024fallroutine/27384/)

##### 描述

超大型偶像团体HIHO314159总选举刚刚结束了。制作人小Hi正在复盘分析投票过程。 

小Hi获得了N条投票记录，每条记录都包含一个时间戳Ti以及候选人编号Ci，代表有一位粉丝在Ti时刻投了Ci一票。 

给定一个包含K名候选人集合S={S1, S2, ... SK}，小Hi想知道从投票开始(0时刻)，到最后一张票投出的时刻(max{Ti})，期间有多少时间得票最多的前K名候选人恰好是S中的K名候选人。

注意这里对前K名的要求是"严格"的，换句话说，S中的每一名候选人得票都要大于任何一名S之外的候选人。S集合内名次先后不作要求。 

注：HIHO314159这个团体有314159名团员，编号是1~314159。

##### 输入

第一行包含两个整数N和K。

第二行包含2N个整数：T1, C1, T2, C2, ... TN, CN。

第三行包含K个整数：S1, S2, ... SK。

对于30%的数据，1 ≤ N, K ≤ 100

对于60%的数据，1 ≤ N, K ≤ 1000

对于100%的数据, 1 ≤ N, K ≤ 314159 1 ≤ Ti ≤ 1000000 1 ≤ Ci, SK ≤ 314159

##### 输出

一个整数，表示前K名恰好是S一共持续了多少时间。

##### 样例输入

```
10 2  
3 1 4 1 5 1 4 3 6 5 8 3 7 5 8 5 9 1 10 5  
1 5
```

##### 样例输出

```
3
```

```python
import heapq

m = 314160
cnt = [0]*m
n, k = map(int, input().split())
l = list(map(int, input().split()))
vote = [(l[i],l[i+1]) for i in range(0,2*n,2)]
vote.sort()
S = list(map(int, input().split()))
visited = [False]*m
queue = []
for k in S:
    visited[k] = True
    heapq.heappush(queue,(0,k))

if k == 314159:
    print(vote[n-1][0])
    exit()

result = 0
max_ns = 0
for i in range(n):
    cnt[vote[i][1]] += 1
    if visited[vote[i][1]]:
        while cnt[queue[0][1]]:
            current = heapq.heappop(queue)
            current = (current[0] + cnt[current[1]], current[1])
            heapq.heappush(queue, current)
            cnt[current[1]] = 0
    else:
        max_ns = max(max_ns,cnt[vote[i][1]])

    if i != n-1 and queue[0][0] > max_ns:
        result += vote[i+1][0] - vote[i][0]

print(result)
```

## 26646:建筑修建

[OpenJudge - 26646:建筑修建](http://cs101.openjudge.cn/2024fallroutine/26646/)

##### 描述

小雯打算对一个线性街区进行开发，街区的坐标为[0,m)。

现在有n个开发商要承接建筑的修建工作，第i个承包商打算修建宽度为y[i]的建筑，并保证街区包含了x[i]这个整数坐标。

建筑为一个左闭右开的区间，为了方便规划建筑的左侧必须为整数坐标，且左右边界不能超出街区范围。

例如，当m=7, x[i]=5, y[i]=3时，[3,6),[4,7)是仅有的两种合法建筑，[2,5),[5,8)则是不合法的建筑。

两个开发商修建的建筑不能有重叠。例如，[3,5)+[4,6)是不合法的，而[3,5)+[5,7)则是合法的。

小雯想要尽量满足更多开发商的修建工作，请问在合理安排的情况下，最多能满足多少个开发商的需求？

##### 输入

第一行两个整数n,m（n, m ≤ 1000）

之后n行，每行两个整数表示开发商的计划，其中第i行的整数为x[i],y[i]。

输入保证x[i]从小到大排列，且都在[0,m)之间。并且保证y[i] > 0。

##### 输出

一个整数，表示最多能满足多少个开发商的需求。

##### 样例输入

```
3 5
0 1
3 2
3 2
```

##### 样例输出

```
2
```

```python
n, m = map(int, input().split())
x_y = []
for i in range(n):
    x, y = map(int, input().split())
    if i == 0:
        x_y.append((x,y))
    elif x == x_y[-1][0]:
        if y < x_y[-1][1]:
            x_y.pop()
            x_y.append((x,y))
    else:
        x_y.append((x, y))
s_e = []
for x,y in x_y:
    for start in range(max(0,x-y+1),min(m-y,x)+1):
        s_e.append((start,start+y))
s_e.sort(key=lambda x:(x[1],x[0]))

count = 0
current = 0
for start,end in s_e:
    if start >= current:
        current = end
        count += 1
print(count)
```

## 20106:走山路

[OpenJudge - 20106:走山路](http://cs101.openjudge.cn/2024fallroutine/20106/)

##### 描述

某同学在一处山地里，地面起伏很大，他想从一个地方走到另一个地方，并且希望能尽量走平路。
现有一个m*n的地形图，图上是数字代表该位置的高度，"#"代表该位置不可以经过。
该同学每一次只能向上下左右移动，每次移动消耗的体力为移动前后该同学所处高度的差的绝对值。现在给出该同学出发的地点和目的地，需要你求出他最少要消耗多少体力。

##### 输入

第一行是整数 m,n,p，m是行数，n是列数，p是测试数据组数。 0 <= m,n,p <= 100
接下来m行是地形图
再接下来n行每行前两个数是出发点坐标（前面是行，后面是列），后面两个数是目的地坐标（前面是行，后面是列）（出发点、目的地可以是任何地方，出发点和目的地如果有一个或两个在"#"处，则将被认为是无法达到目的地）

##### 输出

n行，每一行为对应的所需最小体力，若无法达到，则输出"NO"

##### 样例输入

```
4 5 3
0 0 0 0 0
0 1 1 2 3
# 1 0 0 0
0 # 0 0 0
0 0 3 4
1 0 1 4
3 4 3 0
```

##### 样例输出

```
2
3
NO

解释：
第一组：从左上角到右下角，要上1再下来，所需体力为2
第二组：一直往右走，高度从0变为1，再变为2，再变为3，消耗体力为3
第三组：左下角周围都是"#"，不可以经过，因此到不了
```

```python
import heapq

movements = [(-1, 0), (1, 0), (0, -1), (0, 1)]
def dijkstra(x1, y1, x2, y2):
    if matrix[x1][y1] == '#' or matrix[x2][y2] == '#':
        return 'NO'
    if (x1,y1) == (x2,y2):
        return 0
    pq = [(0,x1,y1)]
    visited = set()
    min_power = [[float('inf')] * n for _ in range(m)]
    min_power[x1][y1] = 0
    while pq:
        power,x,y = heapq.heappop(pq)
        if (x,y) in visited:
            continue
        visited.add((x,y))
        if (x,y) == (x2,y2):
            return power

        for dx,dy in movements:
            nx, ny = x + dx, y + dy
            if 0 <= nx < m and 0 <= ny < n and matrix[nx][ny] != '#' and (nx, ny) not in visited:
                npower = power + abs(int(matrix[nx][ny])-int(matrix[x][y]))
                if npower < min_power[nx][ny]:
                    min_power[nx][ny] = npower
                    heapq.heappush(pq,(npower,nx,ny))
    return 'NO'



m, n, p = map(int, input().split())
matrix = [list(input().split()) for _ in range(m)]
for _ in range(p):
    x1,y1, x2,y2 = map(int,input().split())
    result = dijkstra(x1,y1,x2,y2)
    print(result)
```

## 04102:宠物小精灵之收服

[OpenJudge - 04102:宠物小精灵之收服](http://cs101.openjudge.cn/2024fallroutine/04102/)

##### 描述

宠物小精灵是一部讲述小智和他的搭档皮卡丘一起冒险的故事。

![img](http://media.openjudge.cn/images/upload/1340073461.jpg)

一天，小智和皮卡丘来到了小精灵狩猎场，里面有很多珍贵的野生宠物小精灵。小智也想收服其中的一些小精灵。然而，野生的小精灵并不那么容易被收服。对于每一个野生小精灵而言，小智可能需要使用很多个精灵球才能收服它，而在收服过程中，野生小精灵也会对皮卡丘造成一定的伤害（从而减少皮卡丘的体力）。当皮卡丘的体力小于等于0时，小智就必须结束狩猎（因为他需要给皮卡丘疗伤），而使得皮卡丘体力小于等于0的野生小精灵也不会被小智收服。当小智的精灵球用完时，狩猎也宣告结束。

我们假设小智遇到野生小精灵时有两个选择：收服它，或者离开它。如果小智选择了收服，那么一定会扔出能够收服该小精灵的精灵球，而皮卡丘也一定会受到相应的伤害；如果选择离开它，那么小智不会损失精灵球，皮卡丘也不会损失体力。

小智的目标有两个：主要目标是收服尽可能多的野生小精灵；如果可以收服的小精灵数量一样，小智希望皮卡丘受到的伤害越小（剩余体力越大），因为他们还要继续冒险。

现在已知小智的精灵球数量和皮卡丘的初始体力，已知每一个小精灵需要的用于收服的精灵球数目和它在被收服过程中会对皮卡丘造成的伤害数目。请问，小智该如何选择收服哪些小精灵以达到他的目标呢？

##### 输入

输入数据的第一行包含三个整数：N(0 < N < 1000)，M(0 < M < 500)，K(0 < K < 100)，分别代表小智的精灵球数量、皮卡丘初始的体力值、野生小精灵的数量。
之后的K行，每一行代表一个野生小精灵，包括两个整数：收服该小精灵需要的精灵球的数量，以及收服过程中对皮卡丘造成的伤害。

##### 输出

输出为一行，包含两个整数：C，R，分别表示最多收服C个小精灵，以及收服C个小精灵时皮卡丘的剩余体力值最多为R。

##### 样例输入

```
样例输入1：
10 100 5
7 10
2 40
2 50
1 20
4 20

样例输入2：
10 100 5
8 110
12 10
20 10
5 200
1 110
```

##### 样例输出

```
样例输出1：
3 30

样例输出2：
0 100
```

提示

对于样例输入1：小智选择：(7,10) (2,40) (1,20) 这样小智一共收服了3个小精灵，皮卡丘受到了70点伤害，剩余100-70=30点体力。所以输出3 30
对于样例输入2：小智一个小精灵都没法收服，皮卡丘也不会收到任何伤害，所以输出0 100

```python
n, m, k = map(int, input().split())
dp = [[-1]*(m+1) for _ in range(k+1)]
dp[0][m] = n
for t in range(k):
    dn, dm= map(int, input().split())
    for i in range(t+1,0,-1):
        for j in range(m):
            if j + dm <= m and dp[i-1][j + dm] != -1:
                dp[i][j] = max(dp[i][j],dp[i-1][j+dm]-dn)
result = False
for i in range(k,-1,-1):
    for j in range(m,-1,-1):
        if dp[i][j] != -1:
            print(i,j)
            result = True
            break
    if result:
        break
```

## 16531:上机考试

[OpenJudge - 16531:上机考试](http://cs101.openjudge.cn/2024fallroutine/16531/)

##### 描述

一个班的同学在M行*N列的机房考试，给出学生的座位分布和每个人的做题情况，统计做题情况与周围（上下左右）任一同学相同的学生人数。
另外，由于考试的优秀率不能超过40%，请统计这个班的优秀人数（可能为0，相同分数的同学必须要么全是优秀，要么全不是优秀）。

##### 输入

第一行为两个整数，M和N。
接下来M行每行N个整数，代表对应学生的编号，编号各不相同，范围由0到M*N-1。
接下来M*N行，顺序给出编号由0到n-1的同学的做题情况，1代表做对，0代表做错。

##### 输出

两个整数，以空格分隔。分别代表“与周围（考虑上下左右四个方向）任一同学做题情况相同（不是“答题总数一样”，答题情况要一模一样）的学生人数”，和“班级优秀的人数”。

##### 样例输入

```
sample1 input:
2 5
0 1 2 3 4
5 6 7 8 9
1 1 1
1 0 1
0 0 0
0 0 1
0 0 0
1 1 1
1 1 1
1 1 1
1 1 1
1 1 1

sample1 output:
6 0

#编号为0 5 6 7 8 9的同学做题情况与周围同学相同，因此第一个整数是6。
全对的同学人数已经超过了40%，因此优秀的人数为0
```

##### 样例输出

```
sample2 input:
1 3
1 0 2
0 1 0 0
0 0 0 0
0 0 0 0

sample2 output:
0 1

#并不存在与相邻同学做题情况相同的同学，并且做对一题的同学比例小于40%，因此有一人优秀
```

##### 提示

tag: matrix
1）M*N行做题情况可能为空。因为如果所有学生做题过程中，一直没有提交，则空。
2）空行(即ASCII的回车符)。空行与全零不一样，不是“答题情况相同”。
3）考试的题目数不定。

```python
def in_matrix(x, y):
    return 0 <= x < m and 0 <= y < n

def same(x, y, dx, dy):
    nx, ny = x + dx, y + dy
    if in_matrix(nx, ny):
        return grades[matrix[x][y]] == grades[matrix[nx][ny]]
    return False

m, n = map(int, input().split())
matrix = [list(map(int, input().split())) for _ in range(m)]
grades = {}
for i in range(m * n):
    grades[i] = tuple(map(int, input().split()))
move = [(1, 0), (-1, 0), (0, 1), (0, -1)]
same_cnt = 0
for i in range(m):
    for j in range(n):
        for dx, dy in move:
            if same(i, j, dx, dy):
                same_cnt += 1
                break
scores = {}
for id, ans in grades.items():
    score = sum(ans)
    if score in scores:
        scores[score] += 1
    else:
        scores[score] = 1
sorted_scores = sorted(scores.items(), key=lambda x: x[0],reverse=True)
max_e = int(m * n * 0.4)
exc = 0

for score, count in sorted_scores:
    if exc + count > max_e:
        break
    exc += count
print(same_cnt, exc)
```