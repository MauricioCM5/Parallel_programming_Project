from abc import ABC, abstractmethod
import numpy as np
from mpi4py import MPI

class ColoringStrategy(ABC):
    def __init__(self, graph):
        self.graph = graph
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.iteration = 0
        self.debug = True

    @abstractmethod
    def initial_coloring(self):
        pass

    @abstractmethod
    def detect_conflicts(self):
        pass

    @abstractmethod
    def recolor_vertices(self):
        pass

class D1Strategy(ColoringStrategy):
    def __init__(self, graph):
        super().__init__(graph)
        self.max_degree = self._calculate_max_degree()
        self.use_edge_based = self.max_degree > 6000
        self.vertices_to_recolor = set()

    def _calculate_max_degree(self):
        max_local_degree = max(len(self.graph.get_neighbors(v)) for v in self.graph.V_local)
        return self.comm.allreduce(max_local_degree, op=MPI.MAX)

    def initial_coloring(self):
        if self.use_edge_based:
            self._color_edge_based()
        else:
            self._color_vertex_based()

    def _color_vertex_based(self):
        vertices_list = list(self.graph.V_local)
        np.random.shuffle(vertices_list)

        for v in vertices_list:
            self._assign_color_to_vertex(v)

    def _color_edge_based(self):
        edges = list(self.graph.E_local.union(self.graph.E_ghost))
        np.random.shuffle(edges)

        for v1, v2 in edges:
            if v1 in self.graph.V_local:
                if self.graph.colors_local[v1] is None:
                    self._assign_color_to_vertex(v1)
            if v2 in self.graph.V_local:
                if self.graph.colors_local[v2] is None:
                    self._assign_color_to_vertex(v2)

    def _assign_color_to_vertex(self, vertex):
        neighbors = self.graph.get_neighbors(vertex)
        used_colors = set()

        for n in neighbors:
            if n in self.graph.colors_local and self.graph.colors_local[n] is not None:
                used_colors.add(self.graph.colors_local[n])
            elif n in self.graph.colors_ghost and self.graph.colors_ghost[n] is not None:
                used_colors.add(self.graph.colors_ghost[n])

        color = 0
        while color in used_colors:
            color += 1
        self.graph.colors_local[vertex] = color

    def detect_conflicts(self):
        local_conflicts = 0
        self.vertices_to_recolor.clear()

        neighbor_colors = {}
        for v in self.graph.V_local:
            neighbor_colors[v] = set()
            for u in self.graph.get_neighbors(v):
                if u in self.graph.V_local:
                    neighbor_colors[v].add(self.graph.colors_local[u])
                elif u in self.graph.V_ghost:
                    neighbor_colors[v].add(self.graph.colors_ghost[u])

        for v in self.graph.V_local:
            v_color = self.graph.colors_local[v]
            if v_color is None:
                continue

            if len(neighbor_colors[v]) > 0 and v_color in neighbor_colors[v]:
                should_recolor = self._should_recolor_vertex(v, neighbor_colors[v])
                if should_recolor:
                    self.vertices_to_recolor.add(v)
                    local_conflicts += 1

        total_conflicts = self.comm.allreduce(local_conflicts, op=MPI.SUM)
        return total_conflicts

    def _should_recolor_vertex(self, v, neighbor_colors):
        v_degree = self._get_vertex_degree(v)
        probability = 1 - (v_degree / self.max_degree)

        return np.random.random() < probability

    def _get_vertex_degree(self, vertex):
        return len(self.graph.get_neighbors(vertex))

    def recolor_vertices(self):
        if not self.vertices_to_recolor:
            return

        vertices_list = sorted(list(self.vertices_to_recolor), key=self._get_vertex_degree)

        for v in vertices_list:
            old_color = self.graph.colors_local[v]
            self.graph.colors_local[v] = None

            self._assign_color_to_vertex(v, avoid_color=old_color)

        self.vertices_to_recolor.clear()

    def _assign_color_to_vertex(self, vertex, avoid_color=None):
        neighbors = self.graph.get_neighbors(vertex)
        used_colors = set()

        for n in neighbors:
            if n in self.graph.colors_local and self.graph.colors_local[n] is not None:
                used_colors.add(self.graph.colors_local[n])
            elif n in self.graph.colors_ghost and self.graph.colors_ghost[n] is not None:
                used_colors.add(self.graph.colors_ghost[n])

        if avoid_color is not None:
            used_colors.add(avoid_color)

        color = 0
        while color in used_colors:
            color += 1
        self.graph.colors_local[vertex] = color

    def increment_iteration(self):
        self.iteration += 1
        np.random.seed(self.rank + self.iteration) 

    def verify_coloring(self):
        local_conflicts = 0
        conflict_details = []
        
        for v in self.graph.V_local:
            v_color = self.graph.colors_local[v]
            neighbors = self.graph.get_neighbors(v)
            
            for u in neighbors:
                u_color = None
                if u in self.graph.V_local:
                    u_color = self.graph.colors_local[u]
                elif u in self.graph.V_ghost:
                    u_color = self.graph.colors_ghost[u]
                
                if u_color is not None and v_color == u_color:
                    local_conflicts += 1
                    conflict_details.append(f"Conflicto entre vértice {v} (color {v_color}) y {u} (color {u_color})")
        
        if local_conflicts > 0:
            print(f"Proceso {self.rank} encontró {local_conflicts} conflictos:")
            for detail in conflict_details:
                print(detail)
        
        total_conflicts = self.comm.allreduce(local_conflicts, op=MPI.SUM)
        return total_conflicts

class D12GLStrategy(ColoringStrategy):
    def __init__(self, graph):#, recolor_degrees=True):
        super().__init__(graph)
        self.recolor_degrees = True
        self.max_degree = self._calculate_max_degree()
        self.use_edge_based = self.max_degree > 6000
        self.second_layer_initialized = False
        
    def _calculate_max_degree(self):
        max_local_degree = max(len(self.graph.get_neighbors(v)) for v in self.graph.V_local)
        return self.comm.allreduce(max_local_degree, op=MPI.MAX)
    
    def _initialize_second_ghost_layer(self):
        if self.second_layer_initialized:
            return
            
        boundary_vertices = {v for v in self.graph.V_local 
                           if any(u in self.graph.V_ghost 
                                for u in self.graph.get_neighbors(v))}
        
        boundary_data = {v: list(self.graph.get_neighbors(v)) for v in boundary_vertices}
        
        all_boundary_data = self.comm.allgather(boundary_data)
        
        for proc_data in all_boundary_data:
            for vertex, neighbors in proc_data.items():
                if vertex in self.graph.V_ghost:  
                    for neighbor in neighbors:
                        if neighbor not in self.graph.V_local:
                            self.graph.V_ghost.add(neighbor)
                            self.graph.E_ghost.add(tuple(sorted([vertex, neighbor])))
        
        self.second_layer_initialized = True
    
    def initial_coloring(self):
        self._initialize_second_ghost_layer()
        
        if self.use_edge_based:
            self._color_edge_based()
        else:
            self._color_vertex_based()
    
    def _color_vertex_based(self):
        vertices_list = list(self.graph.V_local)
        np.random.shuffle(vertices_list)
        
        for v in vertices_list:
            self._assign_color_to_vertex(v)
    
    def _color_edge_based(self):
        edges = list(self.graph.E_local.union(self.graph.E_ghost))
        np.random.shuffle(edges)
        
        for v1, v2 in edges:
            if v1 in self.graph.V_local:
                if self.graph.colors_local[v1] is None:
                    self._assign_color_to_vertex(v1)
            if v2 in self.graph.V_local:
                if self.graph.colors_local[v2] is None:
                    self._assign_color_to_vertex(v2)
    
    def _assign_color_to_vertex(self, vertex):
        neighbors = self.graph.get_neighbors(vertex)
        used_colors = set()
        
        for n in neighbors:
            if n in self.graph.colors_local and self.graph.colors_local[n] is not None:
                used_colors.add(self.graph.colors_local[n])
            elif n in self.graph.colors_ghost and self.graph.colors_ghost[n] is not None:
                used_colors.add(self.graph.colors_ghost[n])
        
        color = 0
        while color in used_colors:
            color += 1
        self.graph.colors_local[vertex] = color
    
    def detect_conflicts(self):
        conflicts = 0
        self.vertices_to_recolor = set()
        
        for v in self.graph.V_local:
            neighbors = self.graph.get_neighbors(v)
            v_color = self.graph.colors_local[v]
            
            for u in neighbors:
                if u in self.graph.V_local:
                    if v_color == self.graph.colors_local[u]:
                        np.random.seed(self.rank + min(v, u) + self.iteration)
                        if self._handle_conflict(v, u):
                            conflicts += 1
                elif u in self.graph.V_ghost:
                    if v_color == self.graph.colors_ghost[u]:
                        np.random.seed(self.rank + u + self.iteration)
                        if self._handle_conflict(v, u):
                            conflicts += 1
        
        for ghost in self.graph.V_ghost:
            ghost_neighbors = {v2 for v1, v2 in self.graph.E_ghost 
                             if v1 == ghost or v2 == ghost}
            ghost_color = self.graph.colors_ghost[ghost]
            
            for neighbor in ghost_neighbors:
                if neighbor in self.graph.V_local:
                    if ghost_color == self.graph.colors_local[neighbor]:
                        np.random.seed(self.rank + ghost + self.iteration)
                        if self._handle_conflict(neighbor, ghost):
                            conflicts += 1
                elif neighbor in self.graph.V_ghost:
                    if ghost_color == self.graph.colors_ghost[neighbor]:
                        common_local_neighbors = set()
                        for v in self.graph.V_local:
                            if ghost in self.graph.get_neighbors(v) and neighbor in self.graph.get_neighbors(v):
                                common_local_neighbors.add(v)
                        
                        for local_v in common_local_neighbors:
                            np.random.seed(self.rank + local_v + self.iteration)
                            if np.random.random() > 0.5:  
                                self.vertices_to_recolor.add(local_v)
                                conflicts += 1
        
        total_conflicts = self.comm.allreduce(conflicts, op=MPI.SUM)
        return total_conflicts
    
    def _handle_conflict(self, v, u):
        if v not in self.graph.V_local:
            return False
            
        if np.random.random() > 0.1:  
            if self.recolor_degrees:
                v_degree = len(self.graph.get_neighbors(v))
                u_degree = len(self.graph.get_neighbors(u))
                
                if v_degree < u_degree:
                    self.vertices_to_recolor.add(v)
                    return True
                elif u_degree < v_degree:
                    return False
            
            self.vertices_to_recolor.add(v)
            return True
            
        return False
    
    def recolor_vertices(self):
        if not self.vertices_to_recolor:
            return
            
        for v in self.vertices_to_recolor:
            self.graph.colors_local[v] = None
        
        vertices_list = list(self.vertices_to_recolor)
        np.random.shuffle(vertices_list)
        
        for v in vertices_list:
            self._assign_color_to_vertex(v)
        
        self.vertices_to_recolor.clear()

    def increment_iteration(self):
        self.iteration += 1
        np.random.seed(self.rank + self.iteration)
    
    def verify_coloring(self):
        local_conflicts = 0
        conflict_details = []
        
        for v in self.graph.V_local:
            v_color = self.graph.colors_local[v]
            neighbors = self.graph.get_neighbors(v)
            
            for u in neighbors:
                u_color = None
                if u in self.graph.V_local:
                    u_color = self.graph.colors_local[u]
                elif u in self.graph.V_ghost:
                    u_color = self.graph.colors_ghost[u]
                
                if u_color is not None and v_color == u_color:
                    local_conflicts += 1
                    conflict_details.append(f"Conflicto entre vértice {v} (color {v_color}) y {u} (color {u_color})")
        
        if local_conflicts > 0:
            print(f"Proceso {self.rank} encontró {local_conflicts} conflictos:")
            for detail in conflict_details:
                print(detail)
        
        total_conflicts = self.comm.allreduce(local_conflicts, op=MPI.SUM)
        return total_conflicts


class D2Strategy(ColoringStrategy):
    def __init__(self, graph):
        super().__init__(graph)
        self.second_layer_initialized = False
        self.vertices_to_recolor = set()

    def _initialize_second_ghost_layer(self):
        if self.second_layer_initialized:
            return

        boundary_vertices = {v for v in self.graph.V_local
                            if any(u in self.graph.V_ghost
                                for u in self.graph.get_neighbors(v))}

        boundary_data = {v: list(self.graph.get_neighbors(v)) for v in boundary_vertices}

        all_boundary_data = self.comm.allgather(boundary_data)

        for proc_data in all_boundary_data:
            for vertex, neighbors in proc_data.items():
                if vertex in self.graph.V_ghost:  
                    for neighbor in neighbors:
                        if neighbor not in self.graph.V_local:
                            self.graph.V_ghost.add(neighbor)
                            self.graph.E_ghost.add(tuple(sorted([vertex, neighbor])))

        self.second_layer_initialized = True

    def initial_coloring(self):
        self._initialize_second_ghost_layer()

        vertices_list = list(self.graph.V_local)
        np.random.shuffle(vertices_list)

        for v in vertices_list:
            self._assign_color_to_vertex(v)

    def _assign_color_to_vertex(self, vertex):
        neighbors = self.graph.get_neighbors(vertex)
        used_colors = set()

        for n in neighbors:
            if n in self.graph.colors_local and self.graph.colors_local[n] is not None:
                used_colors.add(self.graph.colors_local[n])
            elif n in self.graph.colors_ghost and self.graph.colors_ghost[n] is not None:
                used_colors.add(self.graph.colors_ghost[n])

        for u in neighbors:
            for x in self.graph.get_neighbors(u):
                if x in self.graph.colors_local and self.graph.colors_local[x] is not None:
                    used_colors.add(self.graph.colors_local[x])
                elif x in self.graph.colors_ghost and self.graph.colors_ghost[x] is not None:
                    used_colors.add(self.graph.colors_ghost[x])

        color = 0
        while color in used_colors:
            color += 1
        self.graph.colors_local[vertex] = color

    def detect_conflicts(self):
        conflicts = 0
        self.vertices_to_recolor.clear()

        for v in self.graph.V_local:
            v_color = self.graph.colors_local[v]

            for u in self.graph.get_neighbors(v):
                if u in self.graph.V_local:
                    if v_color == self.graph.colors_local[u]:
                        np.random.seed(self.rank + min(v, u) + self.iteration)
                        if self._handle_conflict(v, u):
                            conflicts += 1
                elif u in self.graph.V_ghost:
                    if v_color == self.graph.colors_ghost[u]:
                        np.random.seed(self.rank + u + self.iteration)
                        if self._handle_conflict(v, u):
                            conflicts += 1

            for u in self.graph.get_neighbors(v):
                for x in self.graph.get_neighbors(u):
                    if x in self.graph.V_local:
                        if v_color == self.graph.colors_local[x]:
                            np.random.seed(self.rank + min(v, x) + self.iteration)
                            if self._handle_conflict(v, x):
                                conflicts += 1
                    elif x in self.graph.V_ghost:
                        if v_color == self.graph.colors_ghost[x]:
                            np.random.seed(self.rank + x + self.iteration)
                            if self._handle_conflict(v, x):
                                conflicts += 1

        total_conflicts = self.comm.allreduce(conflicts, op=MPI.SUM)
        return total_conflicts

    def _handle_conflict(self, v, u):
        if v not in self.graph.V_local:
            return False

        if np.random.random() > 0.1:  
            self.vertices_to_recolor.add(v)
            return True

        return False

    def recolor_vertices(self):
        if not self.vertices_to_recolor:
            return

        for v in self.vertices_to_recolor:
            self.graph.colors_local[v] = None

        vertices_list = list(self.vertices_to_recolor)
        np.random.shuffle(vertices_list)

        for v in vertices_list:
            self._assign_color_to_vertex(v)

        self.vertices_to_recolor.clear()

    def increment_iteration(self):
        self.iteration += 1
        np.random.seed(self.rank + self.iteration)

    def verify_coloring(self):
        local_conflicts = 0
        conflict_details = []

        for v in self.graph.V_local:
            v_color = self.graph.colors_local[v]
            neighbors = self.graph.get_neighbors(v)

            for u in neighbors:
                if u in self.graph.V_local:
                    if v_color == self.graph.colors_local[u]:
                        local_conflicts += 1
                        conflict_details.append(f"Conflict between vertex {v} (color {v_color}) and {u} (color {self.graph.colors_local[u]})")
                elif u in self.graph.V_ghost:
                    if v_color == self.graph.colors_ghost[u]:
                        local_conflicts += 1
                        conflict_details.append(f"Conflict between vertex {v} (color {v_color}) and ghost {u} (color {self.graph.colors_ghost[u]})")

            for u in neighbors:
                for x in self.graph.get_neighbors(u):
                    if x in self.graph.V_local:
                        if v_color == self.graph.colors_local[x]:
                            local_conflicts += 1
                            conflict_details.append(f"Distance-2 conflict between vertex {v} (color {v_color}) and {x} (color {self.graph.colors_local[x]})")
                    elif x in self.graph.V_ghost:
                        if v_color == self.graph.colors_ghost[x]:
                            local_conflicts += 1
                            conflict_details.append(f"Distance-2 conflict between vertex {v} (color {v_color}) and ghost {x} (color {self.graph.colors_ghost[x]})")

        if local_conflicts > 0:
            print(f"Process {self.rank} found {local_conflicts} conflicts:")
            for detail in conflict_details:
                print(detail)

        total_conflicts = self.comm.allreduce(local_conflicts, op=MPI.SUM)
        return total_conflicts
    

class PD2Strategy(ColoringStrategy):
    def __init__(self, graph, target_vertices):
        super().__init__(graph)
        self.second_layer_initialized = False
        self.vertices_to_recolor = set()
        self.V_target = target_vertices

    def _initialize_second_ghost_layer(self):
        if self.second_layer_initialized:
            return

        boundary_vertices = {v for v in self.graph.V_local
                            if any(u in self.graph.V_ghost
                                for u in self.graph.get_neighbors(v))}

        boundary_data = {v: list(self.graph.get_neighbors(v)) for v in boundary_vertices}

        all_boundary_data = self.comm.allgather(boundary_data)

        for proc_data in all_boundary_data:
            for vertex, neighbors in proc_data.items():
                if vertex in self.graph.V_ghost: 
                    for neighbor in neighbors:
                        if neighbor not in self.graph.V_local:
                            self.graph.V_ghost.add(neighbor)
                            self.graph.E_ghost.add(tuple(sorted([vertex, neighbor])))

        self.second_layer_initialized = True

    def initial_coloring(self):
        self._initialize_second_ghost_layer()

        vertices_list = [v for v in self.graph.V_local if v in self.V_target]
        np.random.shuffle(vertices_list)

        for v in vertices_list:
            self._assign_color_to_vertex(v)

    def _assign_color_to_vertex(self, vertex):
        """Assign a color to a vertex considering its two-hop neighborhood"""
        neighbors = self.graph.get_neighbors(vertex)
        used_colors = set()

        for n in neighbors:
            if n in self.graph.colors_local and self.graph.colors_local[n] is not None:
                used_colors.add(self.graph.colors_local[n])
            elif n in self.graph.colors_ghost and self.graph.colors_ghost[n] is not None:
                used_colors.add(self.graph.colors_ghost[n])

        for u in neighbors:
            for x in self.graph.get_neighbors(u):
                if x in self.graph.V_local and x in self.V_target:
                    if self.graph.colors_local[x] is not None:
                        used_colors.add(self.graph.colors_local[x])
                elif x in self.graph.V_ghost and x in self.V_target:
                    if self.graph.colors_ghost[x] is not None:
                        used_colors.add(self.graph.colors_ghost[x])

        color = 0
        while color in used_colors:
            color += 1
        self.graph.colors_local[vertex] = color

    def detect_conflicts(self, doPartialColoring=True):
        conflicts = 0
        self.vertices_to_recolor.clear()

        for v in self.graph.V_local:
            if v not in self.V_target:
                continue

            v_color = self.graph.colors_local[v]

            for u in self.graph.get_neighbors(v):
                if u in self.graph.V_local and u in self.V_target:
                    if v_color == self.graph.colors_local[u]:
                        np.random.seed(self.rank + min(v, u) + self.iteration)
                        if self._handle_conflict(v, u):
                            conflicts += 1
                elif u in self.graph.V_ghost and u in self.V_target:
                    if v_color == self.graph.colors_ghost[u]:
                        np.random.seed(self.rank + u + self.iteration)
                        if self._handle_conflict(v, u):
                            conflicts += 1

            for u in self.graph.get_neighbors(v):
                for x in self.graph.get_neighbors(u):
                    if x in self.graph.V_local and x in self.V_target:
                        if v_color == self.graph.colors_local[x]:
                            np.random.seed(self.rank + min(v, x) + self.iteration)
                            if self._handle_conflict(v, x):
                                conflicts += 1
                    elif x in self.graph.V_ghost and x in self.V_target:
                        if v_color == self.graph.colors_ghost[x]:
                            np.random.seed(self.rank + x + self.iteration)
                            if self._handle_conflict(v, x):
                                conflicts += 1

        total_conflicts = self.comm.allreduce(conflicts, op=MPI.SUM)
        return total_conflicts

    def _handle_conflict(self, v, u):
        if v not in self.graph.V_local:
            return False

        if np.random.random() > 0.1: 
            self.vertices_to_recolor.add(v)
            return True

        return False

    def recolor_vertices(self):
        """Recolor conflicting vertices"""
        if not self.vertices_to_recolor:
            return

        for v in self.vertices_to_recolor:
            self.graph.colors_local[v] = None

        vertices_list = list(self.vertices_to_recolor)
        np.random.shuffle(vertices_list)

        for v in vertices_list:
            self._assign_color_to_vertex(v)

        self.vertices_to_recolor.clear()

    def increment_iteration(self):
        self.iteration += 1
        np.random.seed(self.rank + self.iteration)

    def verify_coloring(self):
        local_conflicts = 0
        conflict_details = []

        for v in self.graph.V_local:
            if v not in self.V_target:
                continue

            v_color = self.graph.colors_local[v]
            neighbors = self.graph.get_neighbors(v)

            for u in neighbors:
                if u in self.graph.V_local and u in self.V_target:
                    if v_color == self.graph.colors_local[u]:
                        local_conflicts += 1
                        conflict_details.append(f"Conflict between vertex {v} (color {v_color}) and {u} (color {self.graph.colors_local[u]})")
                elif u in self.graph.V_ghost and u in self.V_target:
                    if v_color == self.graph.colors_ghost[u]:
                        local_conflicts += 1
                        conflict_details.append(f"Conflict between vertex {v} (color {v_color}) and ghost {u} (color {self.graph.colors_ghost[u]})")

            for u in neighbors:
                for x in self.graph.get_neighbors(u):
                    if x in self.graph.V_local and x in self.V_target:
                        if v_color == self.graph.colors_local[x]:
                            local_conflicts += 1
                            conflict_details.append(f"Partial distance-2 conflict between vertex {v} (color {v_color}) and {x} (color {self.graph.colors_local[x]})")
                    elif x in self.graph.V_ghost and x in self.V_target:
                        if v_color == self.graph.colors_ghost[x]:
                            local_conflicts += 1
                            conflict_details.append(f"Partial distance-2 conflict between vertex {v} (color {v_color}) and ghost {x} (color {self.graph.colors_ghost[x]})")

        if local_conflicts > 0:
            print(f"Process {self.rank} found {local_conflicts} conflicts:")
            for detail in conflict_details:
                print(detail)

        total_conflicts = self.comm.allreduce(local_conflicts, op=MPI.SUM)
        return total_conflicts