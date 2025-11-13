import random
import time
from copy import deepcopy
from typing import List, Tuple, Optional, Dict, Set

Grid = List[List[int]]

class LatinCSP:
    def __init__(self, n: int, grid: Optional[Grid] = None):
        self.n = n
        if grid is None:
            self.grid = [[0]*n for _ in range(n)]
        else:
            if len(grid) != n or any(len(row) != n for row in grid):
                raise ValueError("Grid size does not match n")
            for r in range(n):
                for c in range(n):
                    v = grid[r][c]
                    if not (0 <= v <= n):
                        self.invalid = True
                        self.invalid_msg = f"Grid contains invalid value {v} at ({r + 1},{c + 1}). Allowed range is 0..{n}."
                        self.grid = deepcopy(grid)
                        self.vars = []
                        self.domains = {}
                        self.neighbors = {}
                        self.backtracks = 0
                        self.assignments = 0
                        self.start_time = None
                        return
        self.grid = deepcopy(grid)

        valid, msg = self._validate_initial_grid()
        if not valid:
            self.invalid = True
            self.invalid_msg = msg
            self.vars = []
            self.domains = {}
            self.neighbors = {}
            self.backtracks = 0
            self.assignments = 0
            self.start_time = None
            return
        else:
            self.invalid = False
            self.invalid_msg = ""

        self.vars = [(r, c) for r in range(n) for c in range(n) if self.grid[r][c] == 0]
        self.domains: Dict[Tuple[int,int], Set[int]] = {}
        self._init_domains()

        self.neighbors: Dict[Tuple[int,int], Set[Tuple[int,int]]] = {}
        for r in range(n):
            for c in range(n):
                if self.grid[r][c] == 0:
                    nbrs = set()
                    for k in range(n):
                        if k != c:
                            nbrs.add((r,k))
                        if k != r:
                            nbrs.add((k,c))
                    self.neighbors[(r,c)] = nbrs

        self.backtracks = 0
        self.assignments = 0
        self.start_time = None

    def _validate_initial_grid(self) -> Tuple[bool, str]:
        n = self.n
        for r in range(n):
            seen = {}
            for c in range(n):
                v = self.grid[r][c]
                if v == 0:
                    continue
                if v in seen:
                    msg = f"Conflict: value {v} appears twice in row {r+1} at columns {seen[v]+1} and {c+1}."
                    return False, msg
                seen[v] = c
        for c in range(n):
            seen = {}
            for r in range(n):
                v = self.grid[r][c]
                if v == 0:
                    continue
                if v in seen:
                    msg = f"Conflict: value {v} appears twice in column {c+1} at rows {seen[v]+1} and {r+1}."
                    return False, msg
                seen[v] = r
        return True, ""

    def _init_domains(self):
        n = self.n
        for r in range(n):
            row_vals = set(self.grid[r][c] for c in range(n) if self.grid[r][c] != 0)
            for c in range(n):
                if self.grid[r][c] == 0:
                    col_vals = set(self.grid[k][c] for k in range(n) if self.grid[k][c] != 0)
                    forbidden = row_vals | col_vals
                    domain = set(range(1, n+1)) - forbidden
                    self.domains[(r,c)] = domain

    def is_consistent(self, var: Tuple[int,int], val: int) -> bool:
        r, c = var
        for k in range(self.n):
            if self.grid[r][k] == val:
                return False
            if self.grid[k][c] == val:
                return False
        return True

    def select_unassigned_variable(self) -> Tuple[int,int]:
        vars_with_domains = [(v, self.domains[v]) for v in self.domains.keys()]
        min_size = min(len(dom) for _, dom in vars_with_domains)
        candidates = [v for v, dom in vars_with_domains if len(dom) == min_size]
        if len(candidates) == 1:
            return candidates[0]
        best = None
        best_deg = -1
        for v in candidates:
            deg = sum(1 for nbr in self.neighbors.get(v, []) if nbr in self.domains)
            if deg > best_deg:
                best_deg = deg
                best = v
        return best

    def order_domain_values(self, var: Tuple[int,int]) -> List[int]:
        domain = list(self.domains[var])
        impact_scores = []
        for val in domain:
            impact = 0
            for nbr in self.neighbors.get(var, []):
                if nbr in self.domains and val in self.domains[nbr]:
                    impact += 1
            impact_scores.append((impact, val))
        impact_scores.sort()
        return [val for _, val in impact_scores]

    def forward_check(self, var: Tuple[int,int], val: int, domains: Dict[Tuple[int,int], Set[int]]) -> Optional[Dict[Tuple[int,int], Set[int]]]:
        new_domains = deepcopy(domains)
        if var in new_domains:
            del new_domains[var]
        for nbr in self.neighbors.get(var, []):
            if nbr in new_domains:
                if val in new_domains[nbr]:
                    new_domains[nbr] = new_domains[nbr] - {val}
                    if not new_domains[nbr]:
                        return None
        return new_domains

    def backtracking_search(self, max_seconds: Optional[float] = None) -> Optional[Grid]:
        if self.invalid:
            return None
        self.start_time = time.time()
        assignment = deepcopy(self.grid)
        result = self._backtrack(self.domains, assignment, max_seconds)
        return result

    def _backtrack(self, domains: Dict[Tuple[int,int], Set[int]], assignment: Grid, max_seconds: Optional[float]) -> Optional[Grid]:
        if max_seconds is not None and time.time() - self.start_time > max_seconds:
            return None

        if not domains:
            if self._final_check(assignment):
                return assignment
            else:
                return None

        var = self.select_unassigned_variable_from(domains)
        for val in self.order_domain_values_from(var, domains):
            if self.is_consistent_with_assignment(var, val, assignment):
                r, c = var
                assignment[r][c] = val
                self.assignments += 1
                new_domains = self.forward_check(var, val, domains)
                if new_domains is not None:
                    result = self._backtrack(new_domains, assignment, max_seconds)
                    if result is not None:
                        return result
                assignment[r][c] = 0
                self.backtracks += 1
        return None

    def select_unassigned_variable_from(self, domains: Dict[Tuple[int,int], Set[int]]) -> Tuple[int,int]:
        min_size = min(len(dom) for dom in domains.values())
        candidates = [v for v, dom in domains.items() if len(dom) == min_size]
        if len(candidates) == 1:
            return candidates[0]
        best = None
        best_deg = -1
        for v in candidates:
            deg = sum(1 for nbr in self.neighbors.get(v, []) if nbr in domains)
            if deg > best_deg:
                best_deg = deg
                best = v
        return best

    def order_domain_values_from(self, var: Tuple[int,int], domains: Dict[Tuple[int,int], Set[int]]) -> List[int]:
        domain = list(domains[var])
        impact_scores = []
        for val in domain:
            impact = 0
            for nbr in self.neighbors.get(var, []):
                if nbr in domains and val in domains[nbr]:
                    impact += 1
            impact_scores.append((impact, val))
        impact_scores.sort()
        return [val for _, val in impact_scores]

    def is_consistent_with_assignment(self, var: Tuple[int,int], val: int, assignment: Grid) -> bool:
        r, c = var
        for k in range(self.n):
            if assignment[r][k] == val:
                return False
            if assignment[k][c] == val:
                return False
        return True

    def _final_check(self, assignment: Grid) -> bool:
        n = self.n
        for r in range(n):
            seen = set()
            for c in range(n):
                v = assignment[r][c]
                if not (1 <= v <= n):
                    return False
                if v in seen:
                    return False
                seen.add(v)
        for c in range(n):
            seen = set()
            for r in range(n):
                v = assignment[r][c]
                if v in seen:
                    return False
                seen.add(v)
        return True

def generate_partial_latin(n: int, num_clues: int, seed: Optional[int] = None) -> Grid:
    if seed is not None:
        random.seed(seed)

    full = [[((r + c) % n) + 1 for c in range(n)] for r in range(n)]
    row_perm = list(range(n)); random.shuffle(row_perm)
    col_perm = list(range(n)); random.shuffle(col_perm)
    permuted = [[full[row_perm[r]][col_perm[c]] for c in range(n)] for r in range(n)]

    positions = [(r, c) for r in range(n) for c in range(n)]
    random.shuffle(positions)
    if num_clues >= n*n:
        keep = set(positions)
    else:
        keep = set(positions[:num_clues])

    partial = [[0]*n for _ in range(n)]
    for r in range(n):
        for c in range(n):
            if (r,c) in keep:
                partial[r][c] = permuted[r][c]
    return partial

def pretty_print(grid: Grid):
    n = len(grid)
    for r in range(n):
        print(" ".join(str(x) if x != 0 else "." for x in grid[r]))

def main():
    try:
        n = int(input("Size of the square n (e.g., 5): ").strip())
    except Exception:
        print("Invalid input, using n=5")
        n = 5

    mode = input("Do you want to enter your own partially filled grid? (Y/N): ").strip().lower()
    if mode == 'Y':
        print("Enter rows using 0 for empty cells, numbers separated by space. For example, for n=3: '1 0 3'")
        grid = []
        for i in range(n):
            row = input(f"Row {i+1}: ").strip().split()
            if len(row) != n:
                raise SystemExit("Incorrect row length.")
            grid.append([int(x) for x in row])
    else:
        try:
            num_clues = int(input(f"How many filled cells (0..{n*n})? Enter a number: ").strip())
        except Exception:
            num_clues = n * n // 3
        num_clues = max(0, min(n*n, num_clues))
        grid = generate_partial_latin(n, num_clues, seed=None)
        print("\nGenerated partial Latin square ('.' = empty):")
        pretty_print(grid)
        print()

    solver = LatinCSP(n, grid)
    if solver.invalid:
        print("Initial grid is inconsistent:")
        print(solver.invalid_msg)
        print("Please correct the initial values and try again.")
        return

    print("Starting search... (MRV + forward checking + LCV)")
    t0 = time.time()
    solution = solver.backtracking_search(max_seconds=20.0)
    t1 = time.time()
    if solution is None:
        print("No solution found (or timeout).")
    else:
        print(f"Solution found in {t1-t0:.3f}s. Stats: assignments={solver.assignments}, backtracks={solver.backtracks}")
        pretty_print(solution)

if __name__ == "__main__":
    main()
 
