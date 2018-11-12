n = int(input())
mult = 1
summa = 0
while n > 0:
	if n%10 != 0:
		mult = mult * (n%10)
	summa = summa + n%10
	n = n // 10
 
print("Сумма цифр:", summa)
print("Произведение значащих цифр:", mult)
