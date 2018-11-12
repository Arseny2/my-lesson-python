n=int(input())
t=n
mult = 2
summa =2
a = 0
for i in range(2,n):
    mult =mult*n
    summa=summa+n
    a =summa//mult
    print(i,end =' ')
    print(a)
    


