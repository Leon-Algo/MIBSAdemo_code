# Intelligent Path Planning System

## Project Overview

This is an intelligent path planning algorithm (MIBSA: Multi-factor Intelligent Biologic Search Algorithm) based on improved A* and hybrid Ant Colony Optimization. It implements various enhanced pathfinding algorithms applicable to robot navigation, logistics distribution, and other scenarios.

### Core Features

- Multiple improved versions of classic A* algorithm (4/8 directions)
- Adaptive A* algorithm with dynamic learning
- Hybrid optimization of ACO and A*
- Support for various map scenarios and obstacle configurations
- Path smoothing and optimization

## Environment Requirements

```bash
# Dependencies
pip install numpy matplotlib
```

## Algorithm Modules

### 1. Basic A* Algorithm
- Support for 4/8 directional movement
- Manhattan/Euclidean distance as heuristic function 
- Implementation file: `A_star_base.py`

### 2. Improved A* Algorithm
- Dynamic Learning Version(`A_star_DL.py`)
  - Heuristic function optimization through historical paths
  - Dynamic obstacle update support
- Smoothing Version(`A_star_smooth.py`)
  - Post-processing path smoothing
  - Corner optimization

### 3. Hybrid Optimization Algorithm
- ACO-A* combination(`hybrid_ACO.py`)
- Multi-objective optimization support

## Algorithm Parameters

### A* Key Parameters
- Heuristic weight α: Controls balance between g(n) and h(n), default α=0.5
- Movement directions: 4/8 directions, affects path smoothness
- Node expansion strategy: Priority queue sorting method

### Dynamic Learning Parameters
- Learning rate η: 0.01-0.1, controls historical information update speed
- Memory decay factor γ: 0.8-0.95, balances new and old path information
- Update period T: Parameters update every T iterations

### ACO Hybrid Algorithm Parameters
- Pheromone concentration τ: Initial value 0.1
- Pheromone evaporation rate ρ: 0.1-0.3
- Ant count m: 20-50 ants

## Parameter Sensitivity Analysis

### Impact of Heuristic Weight α
| α Value | Search Speed | Path Optimality | Memory Usage |
|---------|-------------|-----------------|--------------|
| 0.3     | Fast        | Poor           | Low          |
| 0.5     | Medium      | Good           | Medium       |
| 0.7     | Slow        | Optimal        | High         |

### Learning Rate η Sensitivity
![alt text](loss_function_sensitivity_analysis.png)

- Too small η: Slow learning, poor adaptability
- Too large η: Possible oscillation, unstable
- Recommended value: 0.05

### Memory Decay Factor γ Analysis
- Larger γ retains more historical information
- Optimal algorithm performance at γ=0.9
- Too large may lead to local optima

## Algorithm Optimization

### 1. Heuristic Function Improvements
- Dynamic weight adjustment
- Multi-objective hybrid heuristics
- Local information compensation

### 2. Search Strategy Optimization
- Bidirectional search
- Dynamic node expansion
- Pruning optimization

### 3. Hybrid Algorithm Enhancement
- Adaptive parameter adjustment
- Local search enhancement
- Multi-population collaborative optimization

## Usage Example

```python
from astar import AStar

# Create planner instance
planner = AStar(map_data="maps/map1.txt")

# Set start and goal points
start = (0, 0)
goal = (50, 50)

# Execute path planning
path = planner.plan(start, goal)

# Visualize results
planner.visualize(path)
```

## Map Format

Map file (`*.txt`) format specification:
```
0 - Traversable area
1 - Obstacle
2 - Start point
3 - End point
```

## Performance Tests

| Algorithm Version | Avg Planning Time(ms) | Path Length | Memory Usage(MB) |
|------------------|---------------------|-------------|-----------------|
| Basic A*         | 125                 | 100%        | 45             |
| Improved A*      | 85                  | 95%         | 48             |
| Hybrid Algorithm | 95                  | 92%         | 52             |

![improvement](improved_plot.png)

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Submit PR

## License

MIT License
