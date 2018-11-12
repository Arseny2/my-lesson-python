
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
bg3 = pygame.Surface((WIN_WIDTH/3,WIN_HEIGHT/3))
bg4 = pygame.Surface((WIN_WIDTH/3,WIN_HEIGHT/3))
bg5 = pygame.Surface((WIN_WIDTH/3,WIN_HEIGHT/3))
bg6 = pygame.Surface((WIN_WIDTH/3,WIN_HEIGHT/3))
bg7 = pygame.Surface((WIN_WIDTH/3,WIN_HEIGHT/3))
bg8 = pygame.Surface((WIN_WIDTH/3,WIN_HEIGHT/3))
bg9 = pygame.Surface((WIN_WIDTH/3,WIN_HEIGHT/3))
finish=False
bg1.fill(pygame.Color("#FF0000"))
bg2.fill(pygame.Color("#0000FF"))
bg3.fill(pygame.Color("#00FF00"))
bg4.fill(pygame.Color("#FF0000"))
bg5.fill(pygame.Color("#00FF00"))
bg6.fill(pygame.Color("#00FF00"))
bg7.fill(pygame.Color("#0000FF"))
bg8.fill(pygame.Color("#00FF00"))
bg9.fill(pygame.Color("#FF0000"))
while finish==False:
        screen.fill(pygame.Color("#000000"))
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                finish=True
        screen.blit(bg1,(x,y))
        if x>0:
                x-=1        
        screen.blit(bg2,(x2,y))
        if x2<2*WIN_WIDTH/3:
                x2+=1
        screen.blit(bg3,(x3,y))
        if x3<1*WIN_WIDTH/3:
                x3+=1
        screen.blit(bg4,(x3,y4))
        screen.blit(bg5,(x2,y4))
        screen.blit(bg6,(x,y4))
        screen.blit(bg7,(x,y5))
        screen.blit(bg8,(x3,y5))
        screen.blit(bg9,(x2,y5))
        pygame.display.update()
        
