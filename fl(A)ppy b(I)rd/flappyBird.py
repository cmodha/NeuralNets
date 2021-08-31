import pygame
import neat
import time
import os
import random
pygame.font.init()
#setting window dimensions
WIN_WIDTH = 500
WIN_HEIGHT = 800
#loading the assets
BIRD_IMGS = [pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","bird1.png"))),pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","bird2.png"))),pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","bird3.png")))]

PIPE_IMG  = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","pipe.png")))

BASE_IMG  = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","base.png")))

BG_IMG  = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","bg.png")))

STAT_FONT = pygame.font.SysFont("comicsans",50)

#defining the class for the bird
class Bird:
    #defining the constants
    IMGS = BIRD_IMGS
    MAX_ROTATION = 25
    ROT_VEL = 20
    ANIMATION_TIME = 5
    # defining the initialisation method where x and y are the coordinates on the screen
    def __init__(self,x,y):
        self.x = x
        self.y = y
        self.tilt = 0
        self.tick_count = 0
        self.vel = 0
        self.height = self.y
        self.img_count = 0
        self.img = self.IMGS[0]
    #defining the method for making the bird jump
    def jump(self):
        self.vel = -10.5
        self.tick_count = 0
        self.height = self.y 
    #defining the method for making the bird move
    def move(self):
        # setting the downward acceleration
        self.tick_count += 1
        d = self.vel*self.tick_count +1.5*self.tick_count**2
        #setting terminal velocity
        if d>= 16:
            d = 16 
        if d < 0:
            d -= 2

        self.y = self.y + d

        if d < 0 or self.y < self.height +50:
            if self.tilt < self.MAX_ROTATION:
                self.tilt = self.MAX_ROTATION
            else:
                if self.tilt > -90:
                    self.tilt -= self.ROT_VEL
    
    #defining the method for drawing the bird
    def draw(self,win):
        self.img_count += 1
        #loop which switches the bird images therefore animating the bird
        if self.img_count < self.ANIMATION_TIME:
            self.img = self.IMGS[0]
        elif self.img_count < self.ANIMATION_TIME*2:
            self.img = self.IMGS[1]
        elif self.img_count < self.ANIMATION_TIME*3:
            self.img = self.IMGS[2]
        elif self.img_count < self.ANIMATION_TIME*4:
            self.img = self.IMGS[1]
        elif self.img_count == self.ANIMATION_TIME*4 + 1:
            self.img = self.IMGS[0]
            self.img_count = 0

        if self.tilt <= -80:
            self.img = self.IMGS[1]
            self.img = self.ANIMATION_TIME * 20

        rotated_image = pygame.transform.rotate(self.img, self.tilt)
        new_rect = rotated_image.get_rect(center = self.img.get_rect(topleft = (self.x,self.y)).center)
        win.blit(rotated_image,new_rect.topleft)
    #method to return mask of the bird for collusion detection
    def get_mask(self):
        return pygame.mask.from_surface(self.img)

#defining the class for the pipe
class Pipe:
    GAP = 200
    VEL = 5
    #defining the initialisation method where x is the x coordinate
    def __init__(self,x):
        self.x = x
        self.height = 0

        self.top = 0
        self.bottom = 0
        self.PIPE_TOP = pygame.transform.flip(PIPE_IMG, False,True)
        self.PIPE_BOTTOM = PIPE_IMG

        self.passed = False
        self.set_height()
    #defining the method to randomise the height of the pipes
    def set_height(self):
        self.height = random.randrange(50,450)
        self.top = self.height - self.PIPE_TOP.get_height()
        self.bottom = self.height + self.GAP
    #defining the method to move the pipes across the screen 
    def move(self):
        self.x -= self.VEL
    #defining the method to draw the pipes
    def draw(self,win):
        win.blit(self.PIPE_TOP, (self.x, self.top))
        win.blit(self.PIPE_BOTTOM, (self.x,self.bottom))
    #defining the method which detects collision between the pipe mask and the bird mask
    def collide(self,bird):
        bird_mask = bird.get_mask()
        top_mask = pygame.mask.from_surface(self.PIPE_TOP)
        bottom_mask = pygame.mask.from_surface(self.PIPE_BOTTOM)

        top_offset = (self.x - bird.x,self.top - round(bird.y))
        bottom_offset = (self.x - bird.x,self.bottom - round(bird.y))

        b_point = bird_mask.overlap(bottom_mask,bottom_offset)
        t_point = bird_mask.overlap(top_mask,top_offset)

        if t_point or b_point:
            return True
        else:
            return False

#defining the class for the base of the game
class Base:
    VEL = 5
    WIDTH = BASE_IMG.get_width()
    IMG = BASE_IMG
    #defining the initialisation method where y is the y coordinate
    def __init__(self,y):
        self.y = y
        self.x1 = 0
        self.x2 = self.WIDTH
    #defining the move method
    def move(self):
        self.x1 -= self.VEL
        self.x2 -= self.VEL

        if self.x1 + self.WIDTH < 0:
            self.x1 = self.x2 + self.WIDTH
        
        if self.x2 + self.WIDTH < 0:
            self.x2 = self.x1 + self.WIDTH
    #defining the method to draw the ground in the window 
    def draw(self, win):
        win.blit(self.IMG, (self.x1, self.y))
        win.blit(self.IMG, (self.x2, self.y))
#function to draw the window
def draw_window(win, birds, pipes, base, score):
    win.blit(BG_IMG,(0,0))

    for pipe in pipes:
        pipe.draw(win)

    text = STAT_FONT.render("Score: " + str(score), 1, (255,255,255))
    win.blit(text, (WIN_WIDTH - 10 - text.get_width(), 10))

    base.draw(win)
    for bird in birds:
        bird.draw(win)
    pygame.display.update()

#defining the main function 
def main(genomes, config):
    #lists to hold the genomes and the neural networks associated with the genomes
    nets = []
    ge = []
    birds = []

    for _,g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g,config)
        nets.append(net)
        birds.append(Bird(230,350))
        #start with a fitness of 0
        g.fitness = 0
        ge.append(g)
    
    base = Base(700)
    pipes = [Pipe(700)]
    win = pygame.display.set_mode((WIN_WIDTH,WIN_HEIGHT))
    run = True
    clock = pygame.time.Clock()
    
    score  = 0
    
    while run:
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()

        
        pipe_ind = 0
        if len(birds) > 0 :
            #determines which pipe to use as 2 can be on screen at the same time
            if len(pipes) > 1 and birds[0].x > pipes[0].x + pipes[0].PIPE_TOP.get_width():
                pipe_ind = 1
        else:
            run = False
            break
        for x, bird in enumerate(birds):
            bird.move()
            ge[x].fitness+= 0.1
            #the bird location and the top and bottom pipe location are sent as inputs into the neural network
            output = nets[x].activate((bird.y, abs(bird.y - pipes[pipe_ind].height), abs(bird.y - pipes[pipe_ind].bottom)))
            #using a sigmoid function the activation threshold is 0.75
            if output[0] > 0.75:
                bird.jump()
        
        add_pipe = False
        rem = []
        for pipe in pipes:
            for x, bird in enumerate(birds):
                if pipe.collide(bird):
                    ge[x].fitness -= 1
                    birds.pop(x)
                    nets.pop(x)
                    ge.pop(x)

                if not pipe.passed and pipe.x < bird.x:
                    pipe.passed = True
                    add_pipe = True

            if pipe.x + pipe.PIPE_TOP.get_width() < 0:
                rem.append(pipe)

            pipe.move()
        #rewarding the network for going through the pipe
        if add_pipe:
            score += 1
            for g in ge:
                g.fitness += 5
            
            pipes.append(Pipe(700))
        
        for r in rem:
            pipes.remove(r)
        
        for x, bird in enumerate(birds):
            if bird.y + bird.img.get_height() >= 730 or bird.y < 0:
                birds.pop(x)
                nets.pop(x)
                ge.pop(x)

        
        base.move()
        draw_window(win, birds, pipes, base, score)       
                    
  

#runs the neat algorthm to train the birds using the configs
def run(config_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,neat.DefaultSpeciesSet, neat.DefaultStagnation,config_path)
    #create the population 
    p = neat.Population(config)
    #add a reporter to show the progress in the terminal 
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    #run for up to 50 generations
    winner = p.run(main,50)


#used to determine the configuration path file
if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-feedforward.txt")
    run(config_path)