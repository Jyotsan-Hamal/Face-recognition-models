


def count(stri,ele):
    counto = 0
    for i in stri:
        if i == ele:
            counto+=1
    print(f"{stri}={counto}")
    return counto

def maxScore(s):
    lis = list(s)
    a = 0
    b = 0
    ma = 0
    for i in range(len(lis)-1):
        a = count(lis[:i+1],"0")
        b = count(lis[i+1:],"1")
        
        if ma<(a+b):
            ma  = a+b
        print(f"ma={ma}")
    return ma
maxScore("011101")