# HexFold

HexFold is a creative coding study project developed for the course **Creative Coding**. It explores the procedural growth of paths on a honeycomb grid layout based on **axial coordinates**, where animated agents traverse, expand, fork, and reconnect across a hexagonal structure. The name refers to the way the line network gradually unfolds edge by edge over time. The project combines algorithmic graph traversal, generative visual patterns and time-based animation. Agents use **breadth-first search (BFS)** to travel through existing edges toward the nearest frontier, creating an evolving visual system between controlled growth and emergent structure.

# Setup

- **Install dependencies (e.g. with `uv`):**
```commandline
uv sync
```

- **Follow the [py5 instructions and see requirements](https://py5coding.org/content/install.html)**

## Environment Variables

| Variable | Example | Description |
|---|---:|---|
| `HEXFOLD_ROWS` | `7` | Number of hexagon rows used to build the honeycomb grid. |
| `HEXFOLD_COLS` | `18` | Number of hexagon columns used to build the honeycomb grid. |
| `HEXFOLD_DEBUG` | `1` | Enables debug rendering. When enabled, the sketch draws overlays such as vertices, sparse coordinate labels, and underlying grid edges. |
| `HEXFOLD_STEPS_BEFORE_FORK` | `5` | Minimum number of growth steps an agent must perform before it is allowed to fork again. Also controls the visual growth progress until an agent appears fork-ready. |
| `HEXFOLD_AGENT_EXCLUDE_RADIUS` | `10` | Graph-distance radius used to check for nearby agents before allowing a fork. The check uses the static honeycomb topology. |
| `HEXFOLD_MAX_NEARBY_AGENTS` | `0` | Maximum number of other agents allowed within `HEXFOLD_AGENT_EXCLUDE_RADIUS` before a fork is blocked. `0` means no other agents may be nearby. |
| `HEXFOLD_SEED` | `123` | Random seed used to make simulation runs reproducible. Using the same grid size and seed should reproduce the same growth decisions. |
| `HEXFOLD_EDGE_TRAVERSE_MS` | `180` | Duration in milliseconds for an agent to visually move from one vertex to the next. |
| `HEXFOLD_TRAVEL_DWELL_MS` | `120` | Additional dwell time in milliseconds after each travel-mode edge traversal while an agent travels through existing edges toward the nearest frontier. |

### Example `.env`

```env
HEXFOLD_ROWS=7
HEXFOLD_COLS=18
HEXFOLD_DEBUG=1

HEXFOLD_STEPS_BEFORE_FORK=5
HEXFOLD_AGENT_EXCLUDE_RADIUS=10
HEXFOLD_MAX_NEARBY_AGENTS=0
HEXFOLD_SEED=123

HEXFOLD_EDGE_TRAVERSE_MS=180
HEXFOLD_TRAVEL_DWELL_MS=120
