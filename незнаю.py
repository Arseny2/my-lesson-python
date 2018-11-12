from turtle import *
for k in range (3):
    for j in range (3):
        for i in range (4):
            fd(50)
            lt(90)
        fd(50)
    if k<2:    
       fd(-150)
       rt(90)
       fd(50)
       lt(90)
