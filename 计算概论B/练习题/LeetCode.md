# 困难

## 517.超级洗衣机

https://leetcode.cn/problems/super-washing-machines/

假设有 `n` 台超级洗衣机放在同一排上。开始的时候，每台洗衣机内可能有一定量的衣服，也可能是空的。

在每一步操作中，你可以选择任意 `m` (`1 <= m <= n`) 台洗衣机，与此同时将每台洗衣机的一件衣服送到相邻的一台洗衣机。

给定一个整数数组 `machines` 代表从左至右每台洗衣机中的衣物数量，请给出能让所有洗衣机中剩下的衣物的数量相等的 **最少的操作步数** 。如果不能使每台洗衣机中衣物的数量相等，则返回 `-1` 。

 **示例 1：**

```
输入：machines = [1,0,5]
输出：3
解释：
第一步:    1     0 <-- 5    =>    1     1     4
第二步:    1 <-- 1 <-- 4    =>    2     1     3    
第三步:    2     1 <-- 3    =>    2     2     2   
```

**示例 2：**

```
输入：machines = [0,3,0]
输出：2
解释：
第一步:    0 <-- 3     0    =>    1     2     0    
第二步:    1     2 --> 0    =>    1     1     1     
```

**示例 3：**

```
输入：machines = [0,2,0]
输出：-1
解释：
不可能让所有三个洗衣机同时剩下相同数量的衣物。
```

 **提示：**

- `n == machines.length`
- `1 <= n <= 104`
- `0 <= machines[i] <= 105`



#### 解题思路

找出送最多次的洗衣机（左右均送需叠加处理）

```python
class Solution:
    def findMinMoves(self, machines: List[int]) -> int:
        n = len(machines)
        s = sum(machines)
        if s % n != 0:
            return -1
        a = s // n
        left = []
        right = []
        l_clo = r_clo = 0
        for i in range(n):
            l_clo += machines[i] - a
            left.append(l_clo)
            r_clo += machines[n - 1 - i] - a
            right.append(r_clo)
        right = right[::-1]
        result = 0
        for i in range(n):
            result = max(result, left[i], right[i], left[i] + right[i])
        return result
```

## 135.分发糖果

https://leetcode.cn/problems/candy/

`n` 个孩子站成一排。给你一个整数数组 `ratings` 表示每个孩子的评分。

你需要按照以下要求，给这些孩子分发糖果：

- 每个孩子至少分配到 `1` 个糖果。
- 相邻两个孩子评分更高的孩子会获得更多的糖果。

请你给每个孩子分发糖果，计算并返回需要准备的 **最少糖果数目** 。

 **示例 1：**

```
输入：ratings = [1,0,2]
输出：5
解释：你可以分别给第一个、第二个、第三个孩子分发 2、1、2 颗糖果。
```

**示例 2：**

```
输入：ratings = [1,2,2]
输出：4
解释：你可以分别给第一个、第二个、第三个孩子分发 1、2、1 颗糖果。
     第三个孩子只得到 1 颗糖果，这满足题面中的两个条件。
```

 **提示：**

- `n == ratings.length`
- `1 <= n <= 2 * 104`
- `0 <= ratings[i] <= 2 * 104`



#### 解题思路

向左向右各循环一次

```python
class Solution:
    def candy(self, ratings: List[int]) -> int:
        lth = len(ratings)
        dp = [1]*lth
        for i in range(1,lth):
            if ratings[i] > ratings[i-1]:
                dp[i] = dp[i-1] + 1
        for j in range(lth-2,-1,-1):
            if ratings[j] > ratings[j+1]:
                dp[j] = max(dp[j],dp[j+1]+1)
        return sum(dp)
```

## 42.接雨水

https://leetcode.cn/problems/trapping-rain-water/

给定 `n` 个非负整数表示每个宽度为 `1` 的柱子的高度图，计算按此排列的柱子，下雨之后能接多少雨水。

 **示例 1：**

![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/10/22/rainwatertrap.png)

```
输入：height = [0,1,0,2,1,0,1,3,2,1,2,1]
输出：6
解释：上面是由数组 [0,1,0,2,1,0,1,3,2,1,2,1] 表示的高度图，在这种情况下，可以接 6 个单位的雨水（蓝色部分表示雨水）。 
```

**示例 2：**

```
输入：height = [4,2,0,3,2,5]
输出：9
```

 **提示：**

- `n == height.length`
- `1 <= n <= 2 * 104`
- `0 <= height[i] <= 105`



#### 解题思路

相向双指针；“谁小谁先动”原则

```python
class Solution:
    def trap(self, height: List[int]) -> int:
        lth = len(height)
        result = left = left_max = right_max = 0
        right = lth - 1
        while left < right:
            left_max = max(left_max, height[left])
            right_max = max(right_max, height[right])
            if left_max < right_max:
                result += left_max - height[left]
                left += 1
            else:
                result += right_max - height[right]
                right -= 1
        return result
```

# 中等

## 72.编辑距离

https://leetcode.cn/problems/edit-distance/

给你两个单词 `word1` 和 `word2`， *请返回将 `word1` 转换成 `word2` 所使用的最少操作数* 。

你可以对一个单词进行如下三种操作：

- 插入一个字符
- 删除一个字符
- 替换一个字符

 **示例 1：**

```
输入：word1 = "horse", word2 = "ros"
输出：3
解释：
horse -> rorse (将 'h' 替换为 'r')
rorse -> rose (删除 'r')
rose -> ros (删除 'e')
```

**示例 2：**

```
输入：word1 = "intention", word2 = "execution"
输出：5
解释：
intention -> inention (删除 't')
inention -> enention (将 'i' 替换为 'e')
enention -> exention (将 'n' 替换为 'x')
exention -> exection (将 'n' 替换为 'c')
exection -> execution (插入 'u')
```

 **提示：**

- `0 <= word1.length, word2.length <= 500`
- `word1` 和 `word2` 由小写英文字母组成



#### 动态规划

```python
class Solution:
    def minDistance(self, text1: str, text2: str) -> int:
        lth1, lth2 = len(text1), len(text2)

        dp = [[0] * (lth2 + 1) for _ in range(lth1 + 1)]

        for i in range(1, lth1 + 1):
            dp[i][0] = i
        for j in range(1, lth2 + 1):
            dp[0][j] = j

        for i in range(1, lth1 + 1):
            for j in range(1, lth2 + 1):
                if text1[i - 1] == text2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1]) + 1

        return dp[lth1][lth2]
```

