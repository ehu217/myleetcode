class Solution(object):
    def two_sum(self, l, target):
        dict = {}
        for i in range(len(l)):
            n = l[i]
            if n in dict:
                return [dict[n], i]
            else:
                dict[target-n] = i

    def int_reverse(self, x):
        s = -1 if x < 0 else 1
        n = s * int(str(abs(x))[::-1])
        return n if n.bit_length() < 32 else 0

    def bubble_sort(self, lst):
        for i in range(1, len(lst)):
            flag = False
            for j in range(len(lst) - i):
                if lst[j] > lst[j+1]:
                    lst[j], lst[j+1] = lst[j+1], lst[j]
                    flag = True
            if flag is False:
                break
        return lst

    def select_sort(self, lst):
        for i in range(len(lst) - 1):
            min_idx = i
            min_val = lst[i]
            for j in range(i+1, len(lst)):
                if lst[j] < min_val:
                    min_idx = j
                    min_val = lst[j]
            lst[i], lst[min_idx] = lst[min_idx], lst[i]
        return lst

class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution2:

    def addTwoNumbers(self, l1, l2):
        s1 = str(l1.val)
        s2 = str(l2.val)
        while l1.next is not None:
            s1 += str(l1.next.val)
            l1 = l1.next
        while l2.next is not None:
            s2 += str(l2.next.val)
            l2 = l2.next

        print(s1,s2)
        sr = str(int(s1[::-1]) + int(s2[::-1]))[::-1]
        res = list(sr)
        res_node = h = ListNode(res[0])
        for i in range(1, len(res)):
            a = ListNode(res[i])
            h.next = a
            h = a

        return res_node

    def addTwoNumbers2(self, l1, l2):
        res = n = ListNode(0)
        carry = 0
        while l1 or l2 or carry:
            v1 = v2 = 0
            if l1:
                v1 = l1.val
                l1 = l1.next
            if l2:
                v2 = l2.val
                l2 = l2.next
            carry, val = divmod(v1+v2+carry, 10)
            n.next = ListNode(val)
            n = n.next
        return res.next

    def lengthOfLongestSubstring(self, s):
        start = 0
        max_len = 0 if s == '' else 1
        count = max_len
        for i in range(1, len(s)):
            sub = s[start:i]
            if s[i] in sub:
                start += sub.index(s[i]) + 1
                count = i - start + 1
            else:
                count += 1
                max_len = count if count > max_len else max_len
        return max_len

    def findMedianSortedArrays0(self, nums1, nums2):
        m = len(nums1)
        n = len(nums2)
        i = j = 0
        res = []
        while i < m and j < n:
            if nums1[i] < nums2[j]:
                res.append(nums1[i])
                i += 1
            else:
                res.append(nums2[j])
                j += 1
        if i < m:
            res = res + nums1[i:m]
        if j < n:
            res = res + nums2[j:n]
        print(res)
        return res[(m+n)//2] if (m+n)%2==1 else (res[(m+n)//2] + res[(m+n)//2 -1])/2

    def middlesort(self, lst, key):
        left, right = 0, len(lst) - 1
        while left <= right:
            mid = left + (right - left)//2
            if lst[mid] == key:
                return mid
            elif lst[mid] > key:
                right = mid - 1
            else:
                left = mid + 1
        return -1

    def findMedianSortedArrays(self, nums1, nums2):
        import sys
        m, n = len(nums1), len(nums2)
        if m > n:
            return self.findMedianSortedArrays(nums2, nums1)
        left, right = 0, m
        llen = (m + n + 1)//2
        while left <= right:
            # i means the number of element come from nums1 in left of merged list.
            # j means the number of element come from nums2 in left of merged list.
            i = left + (right - left)//2
            j = llen - i
            Aleftmax = -sys.maxsize-1 if i == 0 else nums1[i-1]
            Arightmin = sys.maxsize if i == m else nums1[i]
            Bleftmax = -sys.maxsize - 1 if j == 0 else nums2[j - 1]
            Brightmin = sys.maxsize if j == n else nums2[j]

            if Aleftmax <= Brightmin and Bleftmax <= Arightmin:
                return (max(Aleftmax, Bleftmax) + min(Arightmin, Brightmin))/2.0 if (m+n) % 2 == 0 else max(Aleftmax, Bleftmax)
            elif Aleftmax > Brightmin:
                right = i - 1
            else:
                left = i + 1

    def is_pal(self, ss):
        # print(ss)
        if ss == ss[::-1]:
            return True
        return False

    def longestPalindrome0(self, s):
        i, j = 0, len(s)-1
        start = end = 0
        while i <= j:
            if s[i] == s[j] and (j-i) > (end-start):
                a = s[i:j+1]
                if a == a[::-1]:
                    start, end = i, j
                    i = i+1
                    j = len(s)-1
                else:
                    j = j-1
            elif (j-i) <= (end-start):
                i = i+1
                j = len(s)-1
            else:
                j = j-1

        return s[start:end+1]

    def longestPalindrome(self, s):
        n = len(s)
        longestsub = s[0:1]
        longestlen = 1
        dp = [[0]*n for _ in range(n)]
        for i in range(n-1, -1, -1):
            for j in range(i, n):
                if j - i <= 1 and s[i] == s[j]:
                    dp[i][j] = 1
                elif s[i] == s[j] and dp[i+1][j-1] == 1:
                    dp[i][j] = 1
                if dp[i][j] == 1 and j-i+1 > longestlen:
                    longestsub = s[i:j+1]
                    longestlen = j - i + 1

        return longestsub

    def convert0(self, s, rows):
        if rows == 1:
            return s
        result = ''
        flag = 0
        k = len(s) // rows
        if rows*k + (rows-2)*(k-1) >= len(s):
            col = k + (rows-2)*(k-1)
        else:
            # col = k + (rows-2)*(k-1) + len(s) - row*k - (row-2)*(k-1)
            col = len(s) - (rows-1)*k
        p = [['']*col for _ in range(rows)]

        i = j = 0
        for index in list(s):
            if (0 <= i < rows) and (0 <= j < col):
                p[i][j] = index
            if flag == 0:
                i = i+1
            else:
                i = i-1
                j = j+1
            if i == rows:
                i = i - 2
                j = j + 1
                flag = 1
            if i == -1:
                i = i+2
                j = j-1
                flag = 0

        print(p)
        for i in range(rows):
            for j in range(col):
                if p[i][j] != '':
                    result += p[i][j]

        return result

    def convert(self, s, rows):
        if rows == 1 or rows >= len(s):
            return s
        lst = ['']*rows
        index = 0
        step = 1

        for x in s:
            lst[index] += x
            if index == rows - 1:
                step = -1
            if index == 0:
                step = 1
            index += step
        return ''.join(lst)

    def myAtoi(self, ss):
        res = 0
        flag = 0
        s = ss.strip()
        if s == '' or (len(s) == 1 and not s[0].isdigit()):
            return 0
        if s[0].isdigit():
            start = 0
        elif s[0] == '-'and s[1].isdigit():
            flag = 1
            start = 1
        elif s[0] == '+' and s[1].isdigit():
            start = 1
        else:
            return 0
        for x in s[start:]:
            if x.isdigit():
                res = res*10 + int(x)
            else:
                break
        if flag == 1:
            res = -1 * res
        if res > 2**31 - 1:
            res = 2**31 - 1
        if res < -1*(2**31):
            res = -1*(2**31)
        return res

    def isPalindrome(self, x):
        res = 0
        if x < 0:
            return False
        y = x
        while y > 0:
            res = res*10 + y%10
            y = y//10
        if res == x:
            return True
        else:
            return False

    def isMatch(self, s, p):
        import re
        res = re.match(p, s)
        print(res)
        if res:
            return True
        else:
            return False




if __name__ == '__main__':
    l1 = ListNode(4)
    l11 = ListNode(6)
    l12 = ListNode(7)
    l1.next = l11
    l11.next = l12
    l2 = ListNode(8)
    l21 = ListNode(3)
    l22 = ListNode(5)
    l2.next = l21
    l21.next = l22
    test = Solution2()
    # res = test.addTwoNumbers(l1, l2)
    # res = test.addTwoNumbers2(l1, l2)
    #     # while True:
    #     #     print(res.val)
    #     #     if res.next is None:
    #     #         break
    #     #     res = res.next
    # print(test.lengthOfLongestSubstring('abcabcbb'))
    # print(test.findMedianSortedArrays([1,5,7,9], [2,8,12,13]))
    # print(test.middlesort([2,8,12,13,88], 13))
    # print(test.longestPalindrome('aaaaaa'))
    # print(test.convert('PAYPALISHIRING',5))
    # print(test.myAtoi('     '))
    # print(test.isPalindrome(12321))
    print(test.isMatch('mississippi', 'mis*is*p*.'))






# if __name__ == '__main__':
#     test = Solution()
#     lst = [88, 5, 2, 56, 9, 3, 11, 4, 0, 7]
#
#     # print(test.two_sum([2, 2], 6))
#     # print(test.int_reverse(2147483649))
#     # print(test.bubble_sort(lst))
#     print(test.select_sort(lst))
