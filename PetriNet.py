import tkinter as tk
from tkinter import ttk
import numpy as np
from collections import defaultdict
from math import atan2, cos, sin

class PetriNet:
    def __init__(self):
        self.places = {}
        self.transitions = {}
        self.arcs = []
        self.markings_history = []
        self.place_coords = {}
        self.transition_coords = {}
        
    def add_place(self, name, tokens=0, x=0, y=0):
        self.places[name] = tokens
        self.place_coords[name] = (x, y)
        
    def add_transition(self, name, x=0, y=0):
        self.transitions[name] = {'input': [], 'output': []}
        self.transition_coords[name] = (x, y)
        
    def add_arc(self, source, target, weight=1):
        if source in self.places and target in self.transitions:
            self.transitions[target]['input'].append((source, weight))
            self.arcs.append((source, target, weight))
        elif source in self.transitions and target in self.places:
            self.transitions[source]['output'].append((target, weight))
            self.arcs.append((source, target, weight))
        else:
            raise ValueError(f"Unknown source or target: {source} -> {target}")
            
    def fire_transition(self, transition_name):
        if transition_name not in self.transitions:
            raise ValueError(f"Unknown transition: {transition_name}")
            
        transition = self.transitions[transition_name]
        
        # Check if transition can fire
        can_fire = all(self.places[place] >= weight for place, weight in transition['input'])
        if not can_fire:
            return False
            
        # Remove tokens from input places
        for place, weight in transition['input']:
            self.places[place] -= weight
            
        # Add tokens to output places
        for place, weight in transition['output']:
            self.places[place] += weight
            
        self.markings_history.append(self.get_marking())
        return True
        
    def get_marking(self):
        return dict(self.places)
        
    def get_reachability_matrix(self):
        if not self.markings_history:
            return np.array([]), []
            
        unique_markings = []
        for marking in self.markings_history:
            if marking not in unique_markings:
                unique_markings.append(marking)
                
        size = len(unique_markings)
        matrix = np.zeros((size, size), dtype=int)
        
        for i in range(len(self.markings_history)-1):
            current = self.markings_history[i]
            next_m = self.markings_history[i+1]
            from_idx = unique_markings.index(current)
            to_idx = unique_markings.index(next_m)
            matrix[from_idx][to_idx] = 1
            
        return matrix, unique_markings


class PetriNetVisualizer:
    def __init__(self, root, petri_net):
        self.root = root
        self.petri_net = petri_net
        self.canvas = tk.Canvas(root, width=800, height=600, bg='white')
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.control_frame = tk.Frame(root)
        self.control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)
        
        self.add_request_btn = tk.Button(
            self.control_frame, 
            text="Добавить заявку", 
            command=self.add_request
        )
        self.add_request_btn.pack(pady=10)
        
        self.start_processing_btn = tk.Button(
            self.control_frame, 
            text="Начать обработку", 
            command=self.start_processing
        )
        self.start_processing_btn.pack(pady=10)
        
        self.finish_processing_btn = tk.Button(
            self.control_frame, 
            text="Завершить обработку", 
            command=self.finish_processing
        )
        self.finish_processing_btn.pack(pady=10)
        
        self.matrix_btn = tk.Button(
            self.control_frame, 
            text="Показать матрицу", 
            command=self.show_matrix
        )
        self.matrix_btn.pack(pady=10)
        
        self.status_label = tk.Label(self.control_frame, text="")
        self.status_label.pack(pady=10)
        
        self.token_visuals = []
        self.draw_network()
        
    def draw_network(self):
        self.canvas.delete("all")
        
        # Draw arcs
        for source, target, weight in self.petri_net.arcs:
            if source in self.petri_net.places:
                start = self.petri_net.place_coords[source]
                end = self.petri_net.transition_coords[target]
            else:
                start = self.petri_net.transition_coords[source]
                end = self.petri_net.place_coords[target]
                
            self.draw_arc(start, end, weight)
        
        # Draw places
        for place, tokens in self.petri_net.places.items():
            x, y = self.petri_net.place_coords[place]
            self.draw_place(x, y, place, tokens)
        
        # Draw transitions
        for transition in self.petri_net.transitions:
            x, y = self.petri_net.transition_coords[transition]
            self.draw_transition(x, y, transition)
    
    def draw_place(self, x, y, name, tokens):
        self.canvas.create_oval(x-30, y-30, x+30, y+30, outline='black', width=2, fill='lightblue')
        self.canvas.create_text(x, y, text=name, font=('Arial', 10))
        self.canvas.create_text(x, y+40, text=f"Токены: {tokens}", font=('Arial', 8))
        
    def draw_transition(self, x, y, name):
        self.canvas.create_rectangle(x-30, y-15, x+30, y+15, outline='black', width=2, fill='lightgreen')
        self.canvas.create_text(x, y, text=name, font=('Arial', 8))
    
    def draw_arc(self, start, end, weight):
        # Calculate direction vector
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        angle = atan2(dy, dx)
        
        # Adjust start and end points to shapes' edges
        start_adj = (
            start[0] + 30 * cos(angle),
            start[1] + 30 * sin(angle)
        )
        end_adj = (
            end[0] - 30 * cos(angle),
            end[1] - 30 * sin(angle)
        )
        
        # Draw the line
        line = self.canvas.create_line(
            start_adj[0], start_adj[1], 
            end_adj[0], end_adj[1], 
            arrow=tk.LAST, width=2
        )
        
        # Draw weight near the middle of the arc
        mid_x = (start_adj[0] + end_adj[0]) / 2
        mid_y = (start_adj[1] + end_adj[1]) / 2
        self.canvas.create_text(mid_x, mid_y, text=str(weight), font=('Arial', 8))
    
    def animate_token(self, start, end, callback=None):
        token = self.canvas.create_oval(0, 0, 10, 10, fill='red')
        self.canvas.move(token, start[0]-5, start[1]-5)
        
        dx = (end[0] - start[0]) / 20
        dy = (end[1] - start[1]) / 20
        
        def move():
            nonlocal token
            self.canvas.move(token, dx, dy)
            pos = self.canvas.coords(token)
            if (dx > 0 and pos[0] >= end[0]) or (dx < 0 and pos[0] <= end[0]):
                self.canvas.delete(token)
                if callback:
                    callback()
            else:
                self.root.after(50, move)
        
        move()
    
    def add_request(self):
        if self.petri_net.fire_transition("Send_Request"):
            self.status_label.config(text="Заявка добавлена в очередь")
            # Animate token from Users to Queue
            start = self.petri_net.place_coords["Users"]
            end = self.petri_net.transition_coords["Send_Request"]
            self.animate_token(start, end, lambda: self.animate_token(
                end,
                self.petri_net.place_coords["Queue"],
                self.draw_network
            ))
        else:
            self.status_label.config(text="Невозможно добавить заявку")
    
    def start_processing(self):
        if self.petri_net.fire_transition("Start_Processing"):
            self.status_label.config(text="Обработка начата")
            # Animate tokens from Queue and Free_WS to Busy_WS
            start_queue = self.petri_net.place_coords["Queue"]
            start_free = self.petri_net.place_coords["Free_WS"]
            transition = self.petri_net.transition_coords["Start_Processing"]
            end = self.petri_net.place_coords["Busy_WS"]
            
            self.animate_token(start_queue, transition, lambda: None)
            self.animate_token(start_free, transition, lambda: 
                self.animate_token(transition, end, self.draw_network)
            )
        else:
            self.status_label.config(text="Невозможно начать обработку (нет заявок или свободных станций)")
    
    def finish_processing(self):
        if self.petri_net.fire_transition("Finish_Processing"):
            self.status_label.config(text="Обработка завершена")
            # Animate tokens from Busy_WS to Free_WS and Processed
            start = self.petri_net.place_coords["Busy_WS"]
            transition = self.petri_net.transition_coords["Finish_Processing"]
            end_free = self.petri_net.place_coords["Free_WS"]
            end_processed = self.petri_net.place_coords["Processed"]
            
            self.animate_token(start, transition, lambda: (
                self.animate_token(transition, end_free, lambda: None),
                self.animate_token(transition, end_processed, self.draw_network)
            ))
        else:
            self.status_label.config(text="Невозможно завершить обработку (нет занятых станций)")
    
    def show_matrix(self):
        matrix, markings = self.petri_net.get_reachability_matrix()
        
        matrix_window = tk.Toplevel(self.root)
        matrix_window.title("Матрица достижимости")
        
        text = tk.Text(matrix_window, wrap=tk.NONE)
        text.pack(fill=tk.BOTH, expand=True)
        
        if matrix.size == 0:
            text.insert(tk.END, "Матрица достижимости пуста")
            return
            
        # Create header
        header = " " * 20
        for i in range(len(markings)):
            header += f"M{i} "
        text.insert(tk.END, header + "\n")
        
        # Add matrix rows
        for i in range(len(markings)):
            row = f"M{i}: {markings[i]}"[:20].ljust(20)
            for j in range(len(markings)):
                row += f"{matrix[i][j]}  "
            text.insert(tk.END, row + "\n")


def create_workstation_model(num_workstations=5):
    net = PetriNet()
    
    # Add places with coordinates for visualization
    net.add_place("Users", 100, 100, 100)
    net.add_place("Queue", 0, 300, 100)
    net.add_place("Free_WS", num_workstations, 300, 250)
    net.add_place("Busy_WS", 0, 500, 250)
    net.add_place("Processed", 0, 500, 100)
    
    # Add transitions with coordinates
    net.add_transition("Send_Request", 200, 100)
    net.add_transition("Start_Processing", 400, 175)
    net.add_transition("Finish_Processing", 400, 325)
    
    # Add arcs
    net.add_arc("Users", "Send_Request")
    net.add_arc("Send_Request", "Queue")
    
    net.add_arc("Queue", "Start_Processing")
    net.add_arc("Free_WS", "Start_Processing")
    net.add_arc("Start_Processing", "Busy_WS")
    
    net.add_arc("Busy_WS", "Finish_Processing")
    net.add_arc("Finish_Processing", "Free_WS")
    net.add_arc("Finish_Processing", "Processed")
    
    return net


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Эмулятор сети Петри - Рабочие станции")
    
    net = create_workstation_model(5)
    app = PetriNetVisualizer(root, net)
    
    root.mainloop()