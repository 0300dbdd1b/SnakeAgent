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
        self.moves_total = 0
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
        self.moves_total += 1

        # If action is invalid, terminate the game
        if np.array_equal(action, [0, 0, 0]):
            return -10, True, self.score, self.get_state()

        self._move(action)
        self.snake.insert(0, list(self.head))

        game_over = False

        if self.moves_left <= 0:
            game_over = True
            return self.moves_total, game_over, self.score, self.get_state()

        if self.is_collision():
            game_over = True
            return self.moves_total, game_over, self.score, self.get_state()

        if self.head == self.food:
            self.score += 1
            self.place_food()
            self.moves_left += 50
        else:
            self.snake.pop()

        return self.moves_total, game_over, self.score, self.get_state()
    
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
        
        def count_spaces(start_x, start_y, dx, dy):
            spaces = 0
            x, y = start_x, start_y
            
            # Count spaces until hitting wall or snake body
            while True:
                x += dx * self.block_size
                y += dy * self.block_size
                
                # Check wall collision
                if (x < 0 or x >= self.width or 
                    y < 0 or y >= self.height):
                    break
                    
                # Check snake body collision
                if [x, y] in self.snake:
                    break
                    
                spaces += 1
                
            # Normalize by maximum possible spaces in that direction
            max_spaces = max(self.width, self.height) // self.block_size
            return spaces / max_spaces

        # Get direction vectors for current orientation
        direction_vectors = {
            Direction.RIGHT: (1, 0),
            Direction.LEFT: (-1, 0),
            Direction.UP: (0, -1),
            Direction.DOWN: (0, 1)
        }
        
        # Calculate directional spaces based on current direction
        current_vector = direction_vectors[self.direction]
        left_vector = (-current_vector[1], current_vector[0])   # Rotate 90° left
        right_vector = (current_vector[1], -current_vector[0])  # Rotate 90° right
        
        # Get normalized distances in all three directions
        dist_straight = count_spaces(head[0], head[1], current_vector[0], current_vector[1])
        dist_right = count_spaces(head[0], head[1], right_vector[0], right_vector[1])
        dist_left = count_spaces(head[0], head[1], left_vector[0], left_vector[1])
        
        # Calculate food-related features
        food_delta = np.array(self.food) - np.array(head)
        food_dist = np.linalg.norm(food_delta)
        max_dist = np.sqrt(self.width**2 + self.height**2)
        
        # Calculate angle to food
        food_angle = np.arctan2(food_delta[1], food_delta[0])
        current_angle = {
            Direction.RIGHT: 0,
            Direction.LEFT: np.pi,
            Direction.UP: -np.pi/2,
            Direction.DOWN: np.pi/2
        }[self.direction]
        
        return [
            dist_straight,  # Normalized distance straight ahead
            dist_right,     # Normalized distance to the right
            dist_left,      # Normalized distance to the left
            (food_dist / max_dist),  # Normalized food distance
            ((food_angle - current_angle) / (2 * np.pi) + 0.5) % 1,  # Normalized angle to food
            len(self.snake) / (self.width * self.height / (self.block_size ** 2))  # Progress
        ]
