import pygame



WIN_WIDTH = 800
WIN_HEIGHT = 640
x = 0
y = 0
x1 = 1
y1 = 1
pygame.init()
screen = pygame.display.set_mode((WIN_WIDTH,WIN_HEIGHT))
pygame.display.set_caption("тест")
bg1 = pygame.Surface((WIN_WIDTH/3,WIN_HEIGHT/3))
bg2 = pygame.Surface((WIN_WIDTH/3,WIN_HEIGHT/3))
finish=False
bg1.fill(pygame.Color("#FF0000"))
bg2.fill(pygame.Color("#FF0000"))
while finish==False:
        screen.fill(pygame.Color("#000000"))
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                finish=True
        screen.blit(bg1,(x,y))
        if x<WIN_WIDTH-WIN_WIDTH/3:
                x=x+1
        screen.blit(bg2,(x1,y1))
        if y1<WIN_HEIGHT-WIN_HEIGHT/3:
                y1=y1+1
        pygame.display.update()
        #при достижение границы двигались назад
