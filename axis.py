import numpy as np


class So:
    def solution(s, strs, n):
        ret = 0
        index = []
        for i in range(len(strs)):
            if strs[i] == "X":
                index.append(i)
        k = len(index)
        rand = [0]*k  ##构成赋值矩阵
        j=0
        while(j<10**k):
            m=j
            for i in range(k):
                yushu=m%10
                rand[k-1-j]=yushu
                m=(m-yushu)//10

            for i in rand:
                strtem = strs
                strtem[index[i]] = rand[i]
                num = int(strtem)
                if num % n == 0:
                    ret += 1
                j+=1
        return ret


if __name__ == '__main__':
    s = So()
    str = input('k=')
    n = int(input('n='))
    print(s.solution(str, n))