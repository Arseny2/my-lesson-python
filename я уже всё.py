#сумма
n=int(input())
s = 0
p = 1
for i in range(1,n+1):
    s =i+s
    p =i*p
print(s/p)
print(p)
print(s)
