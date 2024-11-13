from abc import ABC, abstractmethod
import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt

import networkx as nx

from strategies import *

debug = True
num_vertices = 0

def init_num_vertices(n):
    global num_vertices
    num_vertices = n

def distribuir_vértices(rank, size):
    base_size = num_vertices // size
    extra_vertices = num_vertices % size
    start = rank * base_size + min(rank, extra_vertices)
    end = start + base_size + (1 if rank < extra_vertices else 0)
    return start, end


class DistributedGraph:
    def __init__(self, rank, size, V_local, V_ghost, E_local, E_ghost):
        self.rank = rank
        self.size = size
        self.comm = MPI.COMM_WORLD
        self.V_local = V_local      
        self.V_ghost = V_ghost      
        
        self.E_local = E_local       
        self.E_ghost = E_ghost      
        
        self.boundary_vertices = set()   
        self.interior_vertices = set()    
        
        for v in self.V_local:
            if any((v, g) in self.E_ghost or (g, v) in self.E_ghost for g in self.V_ghost):
                self.boundary_vertices.add(v)
            else:
                self.interior_vertices.add(v)
        
        self.colors_local = {v: None for v in V_local}  
        self.colors_ghost = {v: None for v in V_ghost}  

    def get_neighbors(self, vertex):
        neighbors = set()
        
        for v1, v2 in self.E_local:
            if v1 == vertex:
                neighbors.add(v2)
            elif v2 == vertex:
                neighbors.add(v1)
        
        for v1, v2 in self.E_ghost:
            if v1 == vertex:
                neighbors.add(v2)
            elif v2 == vertex:
                neighbors.add(v1)
        
        return neighbors

    
    def communicate_ghost_colors(self):
        comm = MPI.COMM_WORLD
        
        ghost_requests = {}  
        for ghost in self.V_ghost:
            for r in range(self.size):
                start, end = distribuir_vértices(r, self.size)
                if start <= ghost < end:
                    ghost_requests[ghost] = r
                    break

        for round_num in range(self.size):
            target_rank = (self.rank + round_num) % self.size
            
            requests_for_target = [ghost for ghost, owner in ghost_requests.items() if owner == target_rank]
            
            num_requests = len(requests_for_target)
            comm.send(num_requests, dest=target_rank, tag=0)
            
            if num_requests > 0:
                for ghost in requests_for_target:
                    comm.send(ghost, dest=target_rank, tag=1)
            
            source_rank = (self.rank - round_num) % self.size
            num_incoming = comm.recv(source=source_rank, tag=0)
            
            for _ in range(num_incoming):
                requested_vertex = comm.recv(source=source_rank, tag=1)
                if requested_vertex in self.colors_local:
                    color = self.colors_local[requested_vertex]
                    comm.send(color, dest=source_rank, tag=2)
            
            for ghost in requests_for_target:
                color = comm.recv(source=target_rank, tag=2)
                self.colors_ghost[ghost] = color
                if debug:
                    print(f"Proceso {self.rank}: Recibido color {color} para vértice fantasma {ghost} desde proceso {target_rank}")

        comm.barrier()

    def parallel_color(self, strategy_name='D1', **strategy_params):
        if strategy_name == 'D1':
            strategy = D1Strategy(self)#, **strategy_params)
        elif strategy_name == 'D12GL':
            strategy = D12GLStrategy(self)#, **strategy_params)
        elif strategy_name == 'D2':
            strategy = D2Strategy(self)#, **strategy_params)
        elif strategy_name == 'PD2':
            strategy = PD2Strategy(self)#, **strategy_params)

        strategy.initial_coloring()
        self.comm.barrier()
        
        total_conflicts = 1
        max_iterations = 1000
        
        while total_conflicts > 0 and strategy.iteration < max_iterations :# and strategy.iteration < max_iterations:
            if strategy.debug:
                print(f"Proceso {self.rank}: Iniciando iteración {strategy.iteration}")
            
            self.communicate_ghost_colors()
            self.comm.barrier()
            
            local_conflicts = strategy.detect_conflicts()
            total_conflicts = self.comm.allreduce(local_conflicts, op=MPI.SUM)
            
            if strategy.debug:
                print(f"Proceso {self.rank}: Detectados {local_conflicts} conflictos locales, {total_conflicts} totales")
            
            if total_conflicts > 0:
                strategy.recolor_vertices()
                strategy.increment_iteration()
            
            self.comm.barrier()
        
        
        remaining_conflicts = strategy.verify_coloring()
        print(f"Conflictos reales: {remaining_conflicts}")
        return self.colors_local

import random

def dfs(v, graph, visited):
    visited.add(v)
    for neighbor in graph[v]["neighbors"]:
        if neighbor not in visited:
            dfs(neighbor, graph, visited)

def build_complete_graph(num_processes, num_vertices_per_process, connections_per_vertex=3):
    graph = {}
    total_num_vertices = num_processes * num_vertices_per_process

    for i in range(total_num_vertices):
        graph[i] = {"neighbors": set()}

    for i in range(total_num_vertices):
        neighbors = np.random.choice(
            [v for v in range(total_num_vertices) if v != i],
            size=min(connections_per_vertex, total_num_vertices - 1),
            replace=False
        )
        for neighbor in neighbors:
            graph[i]["neighbors"].add(neighbor)
            graph[neighbor]["neighbors"].add(i)

    visited = set()
    dfs(0, graph, visited)

    if len(visited) != total_num_vertices:
        print(f"Warning: The graph is not connected. Only {len(visited)} out of {total_num_vertices} vertices were visited.")
        unvisited = set(range(total_num_vertices)) - visited
        for vertex in unvisited:
            graph[vertex]["neighbors"].add(0)  
            graph[0]["neighbors"].add(vertex)  

    return graph

def distribute_graph(full_graph, rank, size, vertices_per_process):
    start = rank * vertices_per_process
    end = start + vertices_per_process
    local_vertices = set(range(start, end))
    
    ghost_vertices = set()
    local_edges = set()
    ghost_edges = set()
    
    for v in local_vertices:
        if v in full_graph:
            neighbors = full_graph[v]["neighbors"]
            for neighbor in neighbors:
                if neighbor in local_vertices:
                    if (neighbor, v) not in local_edges:  
                        local_edges.add((v, neighbor))
                else:
                    ghost_vertices.add(neighbor)
                    ghost_edges.add((v, neighbor))
    
    for v in full_graph:
        if v not in local_vertices:
            neighbors = full_graph[v]["neighbors"]
            for neighbor in neighbors:
                if neighbor in local_vertices:
                    ghost_vertices.add(v)
                    if (neighbor, v) not in ghost_edges: 
                        ghost_edges.add((v, neighbor))
    
    return local_vertices, ghost_vertices, local_edges, ghost_edges



def visualize_global_graph(colors_per_process):
    global_graph = nx.Graph()
    
    for colors in colors_per_process:
        for node, color in colors.items():
            global_graph.add_node(node, color=color)  


    for node in global_graph.nodes:
        if node + 1 in global_graph.nodes:
            global_graph.add_edge(node, node + 1)

    colors = [global_graph.nodes[n]["color"] for n in global_graph.nodes]
    
    pos = nx.spring_layout(global_graph) 
    nx.draw(global_graph, pos, node_color=colors, with_labels=True, node_size=500, cmap=plt.cm.tab20)
    plt.show()


def plot_colored_graph(c_p_c, strategy="default"):
    G_global = nx.Graph()
    
    for rank, colores in enumerate(c_p_c):
        for nodo, color in colores.items():
            G_global.add_node(nodo, color=color, process=rank) 

    for u, v in graph.E_local:
        G_global.add_edge(u, v)
    for u, v in graph.E_ghost:
        G_global.add_edge(u, v)
    
    colors = [G_global.nodes[n]["color"] for n in G_global.nodes]
    processes = [G_global.nodes[n]["process"] for n in G_global.nodes]
    
    pos = nx.spring_layout(G_global) 
    fig, ax = plt.subplots(figsize=(12, 8))
    nx.draw(G_global, pos, node_color=colors, with_labels=True, node_size=500, cmap=plt.cm.tab20, ax=ax)
    
    ax.set_title(f"Grafo Coloreado - Estrategia: {strategy}")
    
    plt.show()




if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    num_vertices_por_proceso = 3
    total_vertices = num_vertices_por_proceso * size
    init_num_vertices(total_vertices)
    
    if rank == 0:
        grafo_completo = build_complete_graph(size, num_vertices_por_proceso)
    else:
        grafo_completo = None

    grafo_completo = comm.bcast(grafo_completo, root=0)

    V_local, V_ghost, E_local, E_ghost = distribute_graph(grafo_completo, rank, size, num_vertices_por_proceso)

    graph = DistributedGraph(rank, size, V_local, V_ghost, E_local, E_ghost)
    comm.barrier()


    strategy_for_coloring = 'D12GL'

    
    final_colors = graph.parallel_color(strategy_name=strategy_for_coloring, recolor_degrees=True)
    colores_por_proceso = comm.gather(final_colors, root=0)
    if rank == 0:
       plot_colored_graph(colores_por_proceso, strategy_for_coloring)
