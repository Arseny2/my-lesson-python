v = float(input('ввелите cкорость течения'))
v1 = float(input("введите скорость плота"))
s1 = float(input("в ведите  путь плота который находится в задании"))
t1= s1/v1
v2=s1/(t1/2)
v3=v2-v
v4=v2 + v
print('время за которое плот проплыл 30 км',t1)
print('скорость лодки ',v2)
print('скорость лодки против течения рики',v3)
print('скорость лодки по течению реки',v4)
