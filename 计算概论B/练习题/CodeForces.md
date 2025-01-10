## In Love

[Problem - D - Codeforces](https://codeforces.com/contest/1883/problem/D)

Initially, you have an empty multiset of segments. You need to process q operations of two types:

- + l r — Add the segment (l,r)(l,r) to the multiset,
- − l r — Remove **exactly** one segment (l,r)(l,r) from the multiset. It is guaranteed that this segment exists in the multiset.

After each operation, you need to determine if there exists a pair of segments in the multiset that do not intersect. A pair of segments (l,r)(l,r) and (a,b)(a,b) do not intersect if there does not exist a point xx such that l≤x≤r and a≤x≤b.

##### Input

The first line of each test case contains an integer q (1≤q≤1051≤q≤105) — the number of operations.

The next q lines describe two types of operations. If it is an addition operation, it is given in the format + l r. If it is a deletion operation, it is given in the format −− l r (1≤l≤r≤1091≤l≤r≤109).

##### Output

After each operation, print "YES" if there exists a pair of segments in the multiset that do not intersect, and "NO" otherwise.

You can print the answer in any case (uppercase or lowercase). For example, the strings "yEs", "yes", "Yes", and "YES" will be recognized as positive answers.

##### Example

###### Input

```
12
+ 1 2
+ 3 4
+ 2 3
+ 2 2
+ 3 4
- 3 4
- 3 4
- 1 2
+ 3 4
- 2 2
- 2 3
- 3 4
```

###### Output

```
NO
YES
YES
YES
YES
YES
NO
NO
YES
NO
NO
NO
```

##### Note

In the example, after the second, third, fourth, and fifth operations, there exists a pair of segments (1,2)(1,2) and (3,4)(3,4) that do not intersect.

Then we remove exactly one segment (3,4)(3,4), and by that time we had two segments. Therefore, the answer after this operation also exists.

```python
import heapq
from collections import defaultdict
seg1 = []
seg2 = []
l_cnt = defaultdict(int)
r_cnt = defaultdict(int)
q = int(input())
for _ in range(q):
    o,l,r = input().split()
    l, r = int(l), int(r)
    if o == '+':
        heapq.heappush(seg1,-l)
        heapq.heappush(seg2,r)
    else:
        l_cnt[-l] += 1
        r_cnt[r] += 1
 
    while seg1 and l_cnt[seg1[0]] > 0:
        a = seg1[0]
        k = l_cnt[a]
        for _ in range(k):
            heapq.heappop(seg1)
        l_cnt[a] = 0
 
    while seg2 and r_cnt[seg2[0]] > 0:
        b = seg2[0]
        k = r_cnt[b]
        for _ in range(k):
            heapq.heappop(seg2)
        r_cnt[b] = 0
 
    if len(seg1) <= 1 or len(seg2) <= 1:
        print('NO')
    else:
        if seg1[0] + seg2[0] < 0:
            print('YES')
        else:
            print('NO')
```

## C. Kefa and Park

[Problem - C - Codeforces](https://codeforces.com/contest/580/problem/C)

Kefa decided to celebrate his first big salary by going to the restaurant.

He lives by an unusual park. The park is a rooted tree consisting of *n* vertices with the root at vertex 1. Vertex 1 also contains Kefa's house. Unfortunaely for our hero, the park also contains cats. Kefa has already found out what are the vertices with cats in them.

The leaf vertices of the park contain restaurants. Kefa wants to choose a restaurant where he will go, but unfortunately he is very afraid of cats, so there is no way he will go to the restaurant if the path from the restaurant to his house contains more than *m* **consecutive** vertices with cats.

Your task is to help Kefa count the number of restaurants where he can go.

##### Input

The first line contains two integers, *n* and *m* (2 ≤ *n* ≤ 105, 1 ≤ *m* ≤ *n*) — the number of vertices of the tree and the maximum number of consecutive vertices with cats that is still ok for Kefa.

The second line contains *n* integers *a*1, *a*2, ..., *a**n*, where each *a**i* either equals to 0 (then vertex *i* has no cat), or equals to 1 (then vertex *i* has a cat).

Next *n* - 1 lines contains the edges of the tree in the format "*x**i* *y**i*" (without the quotes) (1 ≤ *x**i*, *y**i* ≤ *n*, *x**i* ≠ *y**i*), where *x**i* and *y**i* are the vertices of the tree, connected by an edge.

It is guaranteed that the given set of edges specifies a tree.

##### Output

A single integer — the number of distinct leaves of a tree the path to which from Kefa's home contains at most *m* consecutive vertices with cats.

##### Examples

###### Input

```
4 1
1 1 0 0
1 2
1 3
1 4
```

###### Output

```
2
```

###### Input

```
7 1
1 0 1 1 0 0 0
1 2
1 3
2 4
2 5
3 6
3 7
```

###### Output

```
2
```

##### Note

Let us remind you that a *tree* is a connected graph on *n* vertices and *n* - 1 edge. A *rooted* tree is a tree with a special vertex called *root*. In a rooted tree among any two vertices connected by an edge, one vertex is a parent (the one closer to the root), and the other one is a child. A vertex is called a *leaf*, if it has no children.

Note to the first sample test:![img](https://espresso.codeforces.com/ea0b8a71188dc621ff257d8ea62daabe83faceae.png)The vertices containing cats are marked red. The restaurants are at vertices 2, 3, 4. Kefa can't go only to the restaurant located at vertex 2.

Note to the second sample test:![img](https://espresso.codeforces.com/57346bb186d8cea221155adfb66ad154ea3ba13f.png)The restaurants are located at vertices 4, 5, 6, 7. Kefa can't go to restaurants 6, 7.

```python
from collections import deque
def dfs(point,cat):
    result = 0
    visited = set([])
    stack = deque([(point,cat)])
    while stack:
        point,cat = deque.popleft(stack)
        if point in visited:
            continue
        visited.add(point)
        if point != 1 and len(dct[point]) <= 1:
            result += 1
        for n_point in dct[point]:
            if n_point not in visited:
                if cats[n_point - 1] == 0:
                    stack.append((n_point, 0))
                elif cat + 1 <= m:
                    stack.append((n_point, cat + 1))
 
    return result
n, m = map(int,input().split())
cats = list(map(int,input().split()))
dct = {}
for i in range(1,n+1):
    dct[i] = []
for _ in range(n-1):
    x,y = map(int, input().split())
    dct[x].append(y)
    dct[y].append(x)
result = dfs(1,cats[0])
print(result)
```

## D. Flowers

[Problem - D - Codeforces](https://codeforces.com/contest/474/problem/D)

We saw the little game Marmot made for Mole's lunch. Now it's Marmot's dinner time and, as we all know, Marmot eats flowers. At every dinner he eats some red and white flowers. Therefore a dinner can be represented as a sequence of several flowers, some of them white and some of them red.

But, for a dinner to be tasty, there is a rule: Marmot wants to eat white flowers only in groups of size *k*.

Now Marmot wonders in how many ways he can eat between *a* and *b* flowers. As the number of ways could be very large, print it modulo 1000000007 (109 + 7).

##### Input

Input contains several test cases.

The first line contains two integers *t* and *k* (1 ≤ *t*, *k* ≤ 105), where *t* represents the number of test cases.

The next *t* lines contain two integers *a**i* and *b**i* (1 ≤ *a**i* ≤ *b**i* ≤ 105), describing the *i*-th test.

##### Output

Print *t* lines to the standard output. The *i*-th line should contain the number of ways in which Marmot can eat between *a**i* and *b**i* flowers at dinner modulo 1000000007 (109 + 7).

##### Examples

###### Input

```
3 2
1 3
2 3
4 4
```

###### Output

```
6
5
5
```

##### Note

- For *K* = 2 and length 1 Marmot can eat (*R*).
- For *K* = 2 and length 2 Marmot can eat (*RR*) and (*WW*).
- For *K* = 2 and length 3 Marmot can eat (*RRR*), (*RWW*) and (*WWR*).
- For *K* = 2 and length 4 Marmot can eat, for example, (*WWWW*) or (*RWWR*), but for example he can't eat (*WWWR*).

```python
mod = 1000000007
t, k = map(int, input().split())
dp = [0]*(100001)
dp[0] = 1
pre_list = [0]*(100001)
for i in range(1,100001):
    dp[i] = dp[i-1]
    if i >= k:
        dp[i] = (dp[i] + dp[i-k])%mod
    pre_list[i] = pre_list[i-1] + dp[i]
for _ in range(t):
    a, b = map(int, input().split())
    result = (pre_list[b] - pre_list[a-1])%mod
    print(result)
```

## A. Boredom

[Problem - A - Codeforces](https://codeforces.com/contest/455/problem/A)

Alex doesn't like boredom. That's why whenever he gets bored, he comes up with games. One long winter evening he came up with a game and decided to play it.

Given a sequence *a* consisting of *n* integers. The player can make several steps. In a single step he can choose an element of the sequence (let's denote it *a**k*) and delete it, at that all elements equal to *a**k* + 1 and *a**k* - 1 also must be deleted from the sequence. That step brings *a**k* points to the player.

Alex is a perfectionist, so he decided to get as many points as possible. Help him.

##### Input

The first line contains integer *n* (1 ≤ *n* ≤ 105) that shows how many numbers are in Alex's sequence.

The second line contains *n* integers *a*1, *a*2, ..., *a**n* (1 ≤ *a**i* ≤ 105).

##### Output

Print a single integer — the maximum number of points that Alex can earn.

##### Examples

###### Input

```
2
1 2
```

###### Output

```
2
```

###### Input

```
3
1 2 3
```

###### Output

```
4
```

###### Input

```
9
1 2 1 3 2 2 2 2 3
```

###### Output

```
10
```

##### Note

Consider the third test example. At first step we need to choose any element equal to 2. After that step our sequence looks like this [2, 2, 2, 2]. Then we do 4 steps, on each step we choose any element equals to 2. In total we earn 10 points.

```python
n = int(input())
nums = list(map(int,input().split()))
max_num = max(nums)
count = [0]*(max_num+1)
dp = [0]*(max_num+1)
for x in nums:
    count[x] += 1
for i in range(1,max_num+1):
    dp[i] = max(dp[i-1],dp[i-2] + i*count[i])
print(dp[max_num])
```

## B. Light It Up

[Problem - B - Codeforces](https://codeforces.com/contest/1000/problem/B)

Recently, you bought a brand new smart lamp with programming features. At first, you set up a schedule to the lamp. Every day it will turn power on at moment 00 and turn power off at moment MM. Moreover, the lamp allows you to set a program of switching its state (states are "lights on" and "lights off"). Unfortunately, some program is already installed into the lamp.

The lamp allows only *good* programs. Good program can be represented as a non-empty array aa, where 0<a1<a2<⋯<a|a|<M0<a1<a2<⋯<a|a|<M. All ai must be integers. Of course, preinstalled program is a good program.

The lamp follows program aa in next manner: at moment 00 turns power and light on. Then at moment ai the lamp flips its state to opposite (if it was lit, it turns off, and vice versa). The state of the lamp flips instantly: for example, if you turn the light off at moment 11 and then do nothing, the total time when the lamp is lit will be 11. Finally, at moment MM the lamp is turning its power off regardless of its state.

Since you are not among those people who read instructions, and you don't understand the language it's written in, you realize (after some testing) the only possible way to alter the preinstalled program. You can **insert at most one** element into the program aa, so it still should be a *good* program after alteration. Insertion can be done between any pair of consecutive elements of aa, or even at the beginning or at the end of aa.

Find such a way to alter the program that the total time when the lamp is lit is maximum possible. Maybe you should leave program untouched. If the lamp is lit from xx till moment yy, then its lit for y−xy−x units of time. Segments of time when the lamp is lit are summed up.

##### Input

First line contains two space separated integers nn and MM (1≤n≤1051≤n≤105, 2≤M≤1092≤M≤109) — the length of program aa and the moment when power turns off.

Second line contains nn space separated integers a1,a2,…,ana1,a2,…,an (0<a1<a2<⋯<an<M0<a1<a2<⋯<an<M) — initially installed program aa.

##### Output

Print the only integer — maximum possible total time when the lamp is lit.

##### Examples

###### Input

```
3 10
4 6 7
```

###### Output

```
8
```

###### Input

```
2 12
1 10
```

###### Output

```
9
```

###### Input

```
2 7
3 4
```

###### Output

```
6
```

##### Note

In the first example, one of possible optimal solutions is to insert value x=3x=3 before a1a1, so program will be [3,4,6,7][3,4,6,7] and time of lamp being lit equals (3−0)+(6−4)+(10−7)=8(3−0)+(6−4)+(10−7)=8. Other possible solution is to insert x=5x=5 in appropriate place.

In the second example, there is only one optimal solution: to insert x=2x=2 between a1a1 and a2a2. Program will become [1,2,10][1,2,10], and answer will be (1−0)+(10−2)=9(1−0)+(10−2)=9.

In the third example, optimal answer is to leave program untouched, so answer will be (3−0)+(7−4)=6(3−0)+(7−4)=6.

```python
n, M = map(int, input().split())
a = [0] + list(map(int, input().split())) + [M]
total_time = 0
for i in range(1,len(a),2):
    total_time += a[i] - a[i-1]
final_time = total_time
current_time = 0
k = 0
for i in range(2,len(a),2):
    current_time += a[i-1] - a[i-2]
    if a[i] - a[i-1] > 1:
        final_time = max(final_time, current_time+M-a[i-1]-total_time+current_time-1)
print(final_time)
```

## C. Number of Ways

[Problem - C - Codeforces](https://codeforces.com/contest/466/problem/C)

You've got array *a*[1], *a*[2], ..., *a*[*n*], consisting of *n* integers. Count the number of ways to split all the elements of the array into three contiguous parts so that the sum of elements in each part is the same.

More formally, you need to find the number of such pairs of indices *i*, *j* (2 ≤ *i* ≤ *j* ≤ *n* - 1), that ![img](https://espresso.codeforces.com/669a2f09a3b9e143f54b1f1d9fd6b7dddf403680.png).

##### Input

The first line contains integer *n* (1 ≤ *n* ≤ 5·105), showing how many numbers are in the array. The second line contains *n* integers *a*[1], *a*[2], ..., *a*[*n*] (|*a*[*i*]| ≤  109) — the elements of array *a*.

##### Output

Print a single integer — the number of ways to split the array into three parts with the same sum.

##### Examples

###### Input

```
5
1 2 3 0 3
```

###### Output

```
2
```

###### Input

```
4
0 1 -1 0
```

###### Output

```
1
```

###### Input

```
2
4 1
```

###### Output

```
0
```

```python
n = int(input())
a = list(map(int, input().split()))
a_sum = sum(a)
if a_sum % 3 != 0:
    print(0)
else:
    t = a_sum // 3
    tt = 2 * t
    current_sum = 0
    count_t = 0
    result = 0
    for i in range(n - 1):
        current_sum += a[i]
        if current_sum == tt and 1 <= i <= n-2:
            result += count_t
        if current_sum == t and 0 <= i <= n-3:
            count_t += 1
    print(result)
```

## B. Spreadsheets

[Problem - B - Codeforces](https://codeforces.com/contest/1/problem/B)

In the popular spreadsheets systems (for example, in Excel) the following numeration of columns is used. The first column has number A, the second — number B, etc. till column 26 that is marked by Z. Then there are two-letter numbers: column 27 has number AA, 28 — AB, column 52 is marked by AZ. After ZZ there follow three-letter numbers, etc.

The rows are marked by integer numbers starting with 1. The cell name is the concatenation of the column and the row numbers. For example, BC23 is the name for the cell that is in column 55, row 23.

Sometimes another numeration system is used: RXCY, where X and Y are integer numbers, showing the column and the row numbers respectfully. For instance, R23C55 is the cell from the previous example.

Your task is to write a program that reads the given sequence of cell coordinates and produce each item written according to the rules of another numeration system.

##### Input

The first line of the input contains integer number *n* (1 ≤ *n* ≤ 105), the number of coordinates in the test. Then there follow *n* lines, each of them contains coordinates. All the coordinates are correct, there are no cells with the column and/or the row numbers larger than 106 .

##### Output

Write *n* lines, each line should contain a cell coordinates in the other numeration system.

##### Examples

###### Input

```
2
R23C55
BC23
```

###### Output

```
BC23
R23C55
```

```python
n = int(input())
num = ['0','1','2','3','4','5','6','7','8','9']
for _ in range(n):
    s = input()
    if s[0] == 'R' and s[1]in num and 'C' in s[2:]:
        k = s.index('C')
        x = int(s[1:k])
        y = int(s[k+1:])
        l = ''
        while y > 0:
            y -= 1
            l = chr(y%26 + 65) + l
            y = y//26
        print(l + str(x))
    else:
        k = 0
        for i in range(len(s)):
            if s[i] in num:
                k = i
                break
        r = s[k:]
        c = 0
        for i in range(k):
            c += (int(ord(s[i]))-64)*26**(k-1-i)
        print('R' + str(r) + 'C' + str(c))
```

## B. T-primes

[Problem - B - Codeforces](https://codeforces.com/contest/230/problem/B)

We know that prime numbers are positive integers that have exactly two distinct positive divisors. Similarly, we'll call a positive integer *t* Т-prime, if *t* has exactly three distinct positive divisors.

You are given an array of *n* positive integers. For each of them determine whether it is Т-prime or not.

##### Input

The first line contains a single positive integer, *n* (1 ≤ *n* ≤ 105), showing how many numbers are in the array. The next line contains *n* space-separated integers *x**i* (1 ≤ *x**i* ≤ 1012).

Please, do not use the %lld specifier to read or write 64-bit integers in С++. It is advised to use the cin, cout streams or the %I64d specifier.

##### Output

Print *n* lines: the *i*-th line should contain "YES" (without the quotes), if number *x**i* is Т-prime, and "NO" (without the quotes), if it isn't.

##### Examples

###### Input

```
3
4 5 6
```

###### Output

```
YES
NO
NO
```

##### Note

The given test has three numbers. The first number 4 has exactly three divisors — 1, 2 and 4, thus the answer for this number is "YES". The second number 5 has two divisors (1 and 5), and the third number 6 has four divisors (1, 2, 3, 6), hence the answer for them is "NO".

```python
import math
def sieve(n):
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n+1, i):
                is_prime[j] = False
    return is_prime
limit = 10**6
is_prime = sieve(limit)
n = int(input())
numbers = map(int, input().split())
for x in numbers:
    root = int(math.isqrt(x))
    if root * root != x:
        print('NO')
    elif root <= limit and is_prime[root]:
        print('YES')
    else:
        print('NO')
```

