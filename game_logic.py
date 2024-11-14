from enum import Enum
import random
import numpy as np

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

class GameState:
    def __init__(self, width=400, height=400, block_size=20):
        self.width = width
        self.height = height
        self.block_size = block_size
        self.reset()
    
    def reset(self):
        self.direction = Direction.RIGHT
        self.head = [self.width//2, self.height//2]
        self.snake = [
            self.head,
            [self.head[0]-self.block_size, self.head[1]],
            [self.head[0]-(2*self.block_size), self.head[1]]
        ]
        self.score = 0
        self.food = None
        self.place_food()
        self.frame_iteration = 0
        self.moves_left = 50
        return self.get_state()
    
    def place_food(self):
        x = random.randint(0, (self.width-self.block_size)//self.block_size)*self.block_size
        y = random.randint(0, (self.height-self.block_size)//self.block_size)*self.block_size
        self.food = [x, y]
        if self.food in self.snake:
            self.place_food()

    def step(self, action):
        self.frame_iteration += 1
        self.moves_left -= 1

        self._move(action)
        self.snake.insert(0, list(self.head))
        
        reward = 0
        game_over = False
        
        if self.moves_left <= 0:
            game_over = True
            reward = -10
            return reward, game_over, self.score, self.get_state()

        if self.is_collision():
            game_over = True
            reward = -10
            return reward, game_over, self.score, self.get_state()
        
        if self.head == self.food:
            self.score += 1
            reward = 10
            self.place_food()
            self.moves_left += 50
        else:
            self.snake.pop()
            reward -= 0.1
        
        return reward, game_over, self.score, self.get_state()
    
    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        if pt[0] > self.width - self.block_size or pt[0] < 0 or pt[1] > self.height - self.block_size or pt[1] < 0:
            return True
        if pt in self.snake[1:]:
            return True
        return False
    
    def _move(self, action):
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)
        
        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]
        else:
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]
        self.direction = new_dir
        x = self.head[0]
        y = self.head[1]
        if self.direction == Direction.RIGHT:
            x += self.block_size
        elif self.direction == Direction.LEFT:
            x -= self.block_size
        elif self.direction == Direction.DOWN:
            y += self.block_size
        elif self.direction == Direction.UP:
            y -= self.block_size
        
        self.head = [x, y]

    def get_state(self):
        head = self.snake[0]
        point_l = [head[0] - self.block_size, head[1]]
        point_r = [head[0] + self.block_size, head[1]]
        point_u = [head[0], head[1] - self.block_size]
        point_d = [head[0], head[1] + self.block_size]
        
        dir_l = self.direction == Direction.LEFT
        dir_r = self.direction == Direction.RIGHT
        dir_u = self.direction == Direction.UP
        dir_d = self.direction == Direction.DOWN
        
        state = [
            (dir_r and self.is_collision(point_r)) or
            (dir_l and self.is_collision(point_l)) or
            (dir_u and self.is_collision(point_u)) or
            (dir_d and self.is_collision(point_d)),
            
            (dir_u and self.is_collision(point_r)) or
            (dir_d and self.is_collision(point_l)) or
            (dir_l and self.is_collision(point_u)) or
            (dir_r and self.is_collision(point_d)),
            
            (dir_d and self.is_collision(point_r)) or
            (dir_u and self.is_collision(point_l)) or
            (dir_r and self.is_collision(point_u)) or
            (dir_l and self.is_collision(point_d)),
            
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            self.food[0] < self.head[0],
            self.food[0] > self.head[0],
            self.food[1] < self.head[1],
            self.food[1] > self.head[1]
        ]
        return np.array(state, dtype=float)
