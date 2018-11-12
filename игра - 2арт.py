
import pygame



WIN_WIDTH = 800
WIN_HEIGHT = 640
x = WIN_WIDTH-WIN_WIDTH/3
y = 0
x2 = 0
x3 = 0
y4 = WIN_HEIGHT/3
y5 =2*WIN_HEIGHT/3

pygame.init()
screen = pygame.display.set_mode((WIN_WIDTH,WIN_HEIGHT))
pygame.display.set_caption("тест")
bg1 = pygame.Surface((WIN_WIDTH/3,WIN_HEIGHT/3))
bg2 = pygame.Surface((WIN_WIDTH/3,WIN_HEIGHT/3))
finish=False
bg1.fill(pygame.Color("#ff0000"))
while finish==False:
        screen.fill(pygame.Color("#ffffff"))
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                finish=True
        screen.blit(bg1,(x,y))
        if x>0:
                x+=1        
        screen.blit(bg2,(x2,y))
        if x2<2*WIN_WIDTH/3:
                x2-=1
        pygame.display.update()
        
