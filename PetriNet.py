import tkinter as tk
from tkinter import messagebox
import numpy as np
from collections import defaultdict, deque
from math import atan2, cos, sin
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class PetriNet:
    def __init__(self):
        self.places = {}
        self.transitions = {}
        self.arcs = []
        self.markings_history = []
        self.place_coords = {}
        self.transition_coords = {}
        self.reachability_graph = None

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
            raise ValueError(f"Неизвестный источник или цель: {source} -> {target}")
            
    def fire_transition(self, transition_name):
        if transition_name not in self.transitions:
            raise ValueError(f"Неизвестная транзакция: {transition_name}")
            
        transition = self.transitions[transition_name]
        
        can_fire = all(self.places[place] >= weight for place, weight in transition['input'])
        if not can_fire:
            return False
            
        for place, weight in transition['input']:
            self.places[place] -= weight
            
        for place, weight in transition['output']:
            self.places[place] += weight
            
        self.markings_history.append(self.get_marking())
        return True
        
    def get_marking(self):
        return dict(self.places)
        
    def build_reachability_graph(self):
        initial_marking = self.get_marking()
        self.reachability_graph = {
            'nodes': {str(initial_marking): dict(initial_marking)},
            'edges': []
        }
        
        visited = set()
        queue = deque([(initial_marking, 0)])  # (marking, level)
        
        while queue and len(visited) < 50:  # Ограничение на 50 состояний
            current, level = queue.popleft()
            current_str = str(current)
            
            if current_str in visited:
                continue
                
            visited.add(current_str)
            
            # Проверяем все возможные переходы
            for transition in self.transitions:
                # Сохраняем текущее состояние
                temp_places = dict(self.places)
                self.places = dict(current)
                
                # Пробуем выполнить переход
                if self.fire_transition(transition):
                    new_marking = self.get_marking()
                    new_marking_str = str(new_marking)
                    
                    # Добавляем в граф достижимости
                    if new_marking_str not in self.reachability_graph['nodes']:
                        self.reachability_graph['nodes'][new_marking_str] = dict(new_marking)
                        if len(visited) < 50:  # Проверяем ограничение
                            queue.append((dict(new_marking), level + 1))
                    
                    # Добавляем ребро
                    self.reachability_graph['edges'].append({
                        'from': current_str,
                        'to': new_marking_str,
                        'transition': transition,
                        'level': level
                    })
                
                # Восстанавливаем состояние
                self.places = dict(temp_places)
        
        if len(visited) >= 50:
            print("Предупреждение: достигнуто максимальное количество состояний (50)")
    
    def get_reachability_matrix(self):
        if not self.reachability_graph:
            self.build_reachability_graph()
            
        nodes = list(self.reachability_graph['nodes'].keys())
        
        # Ограничиваем до первых 50 состояний
        nodes = nodes[:50]
        size = len(nodes)
        matrix = np.zeros((size, size), dtype=int)
        
        # Заполняем матрицу только для первых 50 состояний
        for edge in self.reachability_graph['edges']:
            if edge['from'] in nodes and edge['to'] in nodes:
                i = nodes.index(edge['from'])
                j = nodes.index(edge['to'])
                matrix[i][j] = 1
            
        return matrix, [self.reachability_graph['nodes'][node] for node in nodes]


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

        self.tree_btn = tk.Button(
            self.control_frame,
            text="Построить дерево достижимости",
            command=self.show_reachability_tree
        )
        self.tree_btn.pack(pady=10)
        
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
        fill_color = 'lightblue'
        if name == "Заявка в очереди":
            if tokens >= 3:
                fill_color = 'orange'
            if tokens >= 5:
                fill_color = 'red'
        
        radius = 45  # Увеличенный размер круга
        self.canvas.create_oval(
            x-radius, y-radius, 
            x+radius, y+radius, 
            outline='black', width=2, fill=fill_color
        )
        place_labels = {
            "Заявка в очереди": "P1",
            "Обработка заявки": "P2",
            "Свободные процессы": "P3",
            "Готовые заявки": "P4"
        }
        self.canvas.create_text(x, y, text=place_labels.get(name, ""), font=('Arial', 10, 'bold'))
        self.canvas.create_text(x, y+radius+15, text=name, font=('Arial', 8))
        self.canvas.create_text(x, y+radius+30, text=f"Токены: {tokens}", font=('Arial', 8))
        
    def draw_transition(self, x, y, name):
        width = 20
        height = 40
        self.canvas.create_rectangle(
            x - width, y - height,
            x + width, y + height,
            outline='black', width=2, fill='lightgreen'
        )
        # Внутри — T1, T2 и т.п.
        self.canvas.create_text(x, y, text=name, font=('Arial', 10, 'bold'))
        
        # Подпись под прямоугольником
        transition_labels = {
            "T1": "Добавить заявку",
            "T2": "Начать обработку",
            "T3": "Завершить обработку",
            "T4": "Выдать результат"
        }
        self.canvas.create_text(x, y + height + 15, text=transition_labels.get(name, ""), font=('Arial', 8))

    
    def draw_arc(self, start, end, weight):
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        angle = atan2(dy, dx)
        
        place_radius = 45 if start in self.petri_net.place_coords.values() else 0
        transition_offset = 40 if start in self.petri_net.transition_coords.values() else 0

        start_adj = (
            start[0] + place_radius * cos(angle),
            start[1] + place_radius * sin(angle)
        )
        end_adj = (
            end[0] - transition_offset * cos(angle),
            end[1] - transition_offset * sin(angle)
        )
        
        line = self.canvas.create_line(
            start_adj[0], start_adj[1], 
            end_adj[0], end_adj[1], 
            arrow=tk.LAST, width=2
        )
        
        mid_x = (start_adj[0] + end_adj[0]) / 2
        mid_y = (start_adj[1] + end_adj[1]) / 2
        self.canvas.create_text(mid_x, mid_y, text=str(weight), font=('Arial', 8))
    
    def animate_token(self, start, end, callback=None):
        token = self.canvas.create_oval(0, 0, 10, 10, fill='red')
        self.canvas.move(token, start[0]-5, start[1]-5)
        
        steps = 20
        dx = (end[0] - start[0]) / steps
        dy = (end[1] - start[1]) / steps
        
        def move(step=0):
            if step < steps:
                self.canvas.move(token, dx, dy)
                self.root.after(50, move, step+1)
            else:
                self.canvas.delete(token)
                if callback:
                    callback()
        
        move()
    
    def add_request(self):
        total_requests = (self.petri_net.places["Заявка в очереди"] + 
                         self.petri_net.places["Обработка заявки"])
        
        if total_requests >= 10:
            messagebox.showwarning(
                "Ошибка", 
                "Общее количество заявок в системе достигло максимума (10)"
            )
            self.status_label.config(text="Система перегружена (макс. 10 заявок)")
            return
            
        if self.petri_net.places["Заявка в очереди"] >= 5:
            messagebox.showwarning(
                "Ошибка", 
                "Очередь переполнена (максимум 5 заявок)"
            )
            self.status_label.config(text="Очередь переполнена (макс. 5 заявок)")
            return

        if self.petri_net.fire_transition("T1"):
            self.status_label.config(text="Заявка добавлена в очередь")
            transition_coords = self.petri_net.transition_coords["T1"]
            place_coords = self.petri_net.place_coords["Заявка в очереди"]
            self.animate_token(transition_coords, place_coords, self.draw_network)
        else:
            self.status_label.config(text="Невозможно добавить заявку")
    
    def start_processing(self):
        if (self.petri_net.places["Заявка в очереди"] > 0 and 
            self.petri_net.places["Свободные процессы"] > 0):
            
            if self.petri_net.fire_transition("T2"):
                self.status_label.config(text="Обработка начата")
                
                # Анимация движения заявки
                queue_coords = self.petri_net.place_coords["Заявка в очереди"]
                transition_coords = self.petri_net.transition_coords["T2"]
                processing_coords = self.petri_net.place_coords["Обработка заявки"]
                
                self.animate_token(queue_coords, transition_coords, 
                                lambda: self.animate_token(transition_coords, 
                                                        processing_coords,
                                                        self.draw_network))
                
                # Анимация движения свободного процесса
                free_coords = self.petri_net.place_coords["Свободные процессы"]
                busy_coords = (350, 250)  # Координаты "Система занята" (если есть)
                
                self.animate_token(free_coords, transition_coords,
                                lambda: self.animate_token(transition_coords,
                                                        processing_coords,
                                                        self.draw_network))
            else:
                self.status_label.config(text="Ошибка при запуске обработки")
        else:
            self.status_label.config(text="Нет заявок или свободных процессов")
    
    def finish_processing(self):
        if self.petri_net.fire_transition("T3"):
            self.status_label.config(text="Обработка завершена")

            transition_coords = self.petri_net.transition_coords["T3"]
            free_coords = self.petri_net.place_coords["Свободные процессы"]
            ready_coords = self.petri_net.place_coords["Готовые заявки"]
            
            # Анимация движения в свободные процессы
            self.animate_token(transition_coords, free_coords,
                            lambda: None)
            
            # Анимация движения в готовые заявки
            self.animate_token(transition_coords, ready_coords,
                            lambda: self.animate_token(
                                ready_coords,
                                self.petri_net.transition_coords["T4"],
                                self.draw_network
                            ))
        else:
            self.status_label.config(text="Ошибка при завершении обработки")

    
    def show_matrix(self):
        matrix, markings = self.petri_net.get_reachability_matrix()
        
        matrix_window = tk.Toplevel(self.root)
        matrix_window.title("Матрица достижимости")
        
        # Создаем фрейм с прокруткой
        frame = tk.Frame(matrix_window)
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Настройка прокрутки
        scrollbar = tk.Scrollbar(frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        text = tk.Text(
            frame, 
            wrap=tk.NONE,
            yscrollcommand=scrollbar.set,
            font=('Courier New', 8)  # Уменьшенный шрифт для компактности
        )
        text.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=text.yview)
        
        if matrix.size == 0:
            text.insert(tk.END, "Нет данных для построения матрицы достижимости")
            return
            
        # Сокращения для мест
        place_abbr = {
            "Заявка в очереди": "P1",
            "Обработка заявки": "P2",
            "Свободные процессы": "P3",
            "Готовые заявки": "P4"
        }
        
        # Формируем заголовок
        header = "Состояние".ljust(25)
        for i in range(min(50, len(markings))):  # Ограничение до 50 столбцов
            header += f"M{i}".center(8)
        text.insert(tk.END, header + "\n")
        
        # Формируем строки матрицы (не более 50)
        for i in range(min(50, len(markings))):
            # Создаем краткое описание состояния
            marking_desc = []
            for place, tokens in markings[i].items():
                if tokens > 0:
                    marking_desc.append(f"{place_abbr.get(place, place)}={tokens}")
            
            row = ", ".join(marking_desc)[:25].ljust(25)
            
            # Добавляем строку матрицы (не более 50 значений)
            for j in range(min(50, len(markings))):
                row += str(matrix[i][j]).center(8)
            text.insert(tk.END, row + "\n")
        
        # Добавляем пояснения
        if len(markings) > 50:
            text.insert(tk.END, f"\nПоказаны только первые 50 состояний из {len(markings)}\n")
        
        text.insert(tk.END, "\nСокращения мест:\n")
        for place, abbr in place_abbr.items():
            text.insert(tk.END, f"{abbr}: {place}\n")

    def show_reachability_tree(self):
        if not self.petri_net.reachability_graph:
            self.petri_net.build_reachability_graph()
        
        graph = self.petri_net.reachability_graph
        
        # Ограничиваем дерево 40 вершинами
        max_nodes = 40
        if len(graph['nodes']) > max_nodes:
            answer = messagebox.askyesno(
                "Большое дерево", 
                f"Дерево содержит {len(graph['nodes'])} состояний. Показать первые {max_nodes}?",
                parent=self.root
            )
            if not answer:
                return
        
        # Получаем начальную маркировку (корень дерева)
        initial_marking = str(self.petri_net.get_marking())
        
        # Собираем узлы в порядке BFS (первые 40)
        nodes = {initial_marking: graph['nodes'][initial_marking]}
        edges = []
        queue = deque([initial_marking])
        visited = set([initial_marking])
        
        while queue and len(nodes) < max_nodes:
            current = queue.popleft()
            
            for edge in graph['edges']:
                if edge['from'] == current and edge['to'] not in visited:
                    if len(nodes) >= max_nodes:
                        break
                    
                    nodes[edge['to']] = graph['nodes'][edge['to']]
                    edges.append(edge)
                    visited.add(edge['to'])
                    queue.append(edge['to'])
        
        tree_window = tk.Toplevel(self.root)
        tree_window.title("Дерево достижимости (первые 40 состояний)")
        tree_window.geometry("1200x800")
        
        # Создаем canvas с прокруткой
        canvas = tk.Canvas(tree_window, bg='white', scrollregion=(0, 0, 2000, 2000))
        vsb = tk.Scrollbar(tree_window, orient="vertical", command=canvas.yview)
        hsb = tk.Scrollbar(tree_window, orient="horizontal", command=canvas.xview)
        canvas.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        
        vsb.pack(side="right", fill="y")
        hsb.pack(side="bottom", fill="x")
        canvas.pack(side="left", fill="both", expand=True)
        
        # Параметры оформления
        oval_width = 140
        oval_height = 70
        level_height = 150
        h_padding = 60
        
        # Организация узлов по уровням (BFS)
        levels = defaultdict(list)
        node_positions = {}
        
        # Размещаем корневую вершину
        levels[0] = [initial_marking]
        node_positions[initial_marking] = (500, 50)
        
        # Распределяем остальные узлы по уровням
        for edge in edges:
            level = edge['level'] + 1
            if edge['to'] not in levels[level]:
                levels[level].append(edge['to'])
        
        # Рассчитываем позиции для каждого уровня
        for level, node_list in sorted(levels.items()):
            if level == 0:
                continue  # Корень уже размещен
                
            node_count = len(node_list)
            total_width = node_count * oval_width + (node_count - 1) * h_padding
            start_x = max(20, (1000 - total_width) // 2)
            
            for i, node in enumerate(node_list):
                x = start_x + i * (oval_width + h_padding)
                y = 50 + level * level_height
                node_positions[node] = (x, y)
        
        # Рисуем соединительные линии со стрелками
        for edge in edges:
            if edge['from'] in node_positions and edge['to'] in node_positions:
                from_x, from_y = node_positions[edge['from']]
                to_x, to_y = node_positions[edge['to']]
                
                # Корректируем координаты для плавных линий
                from_x += oval_width // 2
                from_y += oval_height
                to_x += oval_width // 2
                
                canvas.create_line(
                    from_x, from_y,
                    to_x, to_y,
                    arrow=tk.LAST, 
                    width=1.5,
                    fill='#555555',
                    arrowshape=(8, 10, 5)
                )
                
                # Подпись перехода
                mid_x = (from_x + to_x) // 2
                mid_y = (from_y + to_y) // 2
                canvas.create_text(
                    mid_x, mid_y - 10,
                    text=edge['transition'],
                    font=('Arial', 8, 'bold'),
                    fill='#333333'
                )
        
        # Рисуем овальные узлы
        for node, (x, y) in node_positions.items():
            marking = nodes[node]
            
            # Овальный узел с градиентной заливкой
            canvas.create_oval(
                x, y,
                x + oval_width, y + oval_height,
                fill='#E6F3FF',  # Светло-голубой
                outline='#0066CC',  # Синяя граница
                width=2
            )
            
            # Подпись узла
            label = self._format_marking(marking, short=True)
            canvas.create_text(
                x + oval_width//2, y + oval_height//2,
                text=label,
                font=('Arial', 8),
                width=oval_width-15,
                justify='center'
            )
        
        # Добавляем заголовок и информационную панель
        canvas.create_text(
            500, 20,
            text="Дерево достижимости сети Петри",
            font=('Arial', 12, 'bold'),
            anchor='n'
        )
        
        info_text = f"Всего состояний: {len(graph['nodes'])} | Показано: {len(nodes)}"
        if len(graph['nodes']) > max_nodes:
            info_text += f" (первые {max_nodes} по BFS)"
        
            canvas.create_text(
                10, 10,
                text=info_text,
                font=('Arial', 9),
                anchor='nw',
                fill='#666666'
            )
        
        # Обновляем область прокрутки
        canvas.update_idletasks()
        canvas.config(scrollregion=canvas.bbox("all"))

    def _format_marking(self, marking, short=False):
        place_abbr = {
            "Заявка в очереди": "P1",
            "Обработка заявки": "P2",
            "Свободные процессы": "P3",
            "Готовые заявки": "P4"
        }
        
        parts = []
        for place, tokens in marking.items():
            if tokens > 0:
                if short:
                    parts.append(f"{place_abbr.get(place, place)}={tokens}")
                else:
                    parts.append(f"{place}={tokens}")
        
        if short:
            return "\n".join(parts)
        return ", ".join(parts)
            
    def _build_tree_visualization(self, parent_frame, matrix, markings):
        G = nx.DiGraph()
        
        # Добавляем узлы и ребра
        for i in range(len(matrix)):
            G.add_node(f"M{i}", label=self._get_marking_label(markings[i]))
            for j in range(len(matrix)):
                if matrix[i][j] == 1:
                    G.add_edge(f"M{i}", f"M{j}")
        
        # Рисуем граф
        fig, ax = plt.subplots(figsize=(10, 8))
        pos = nx.spring_layout(G, seed=42)  # Для повторяемости расположения
        
        # Настраиваем отображение
        node_labels = nx.get_node_attributes(G, 'label')
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=1500, node_color='lightblue')
        nx.draw_networkx_edges(G, pos, ax=ax, arrows=True)
        nx.draw_networkx_labels(G, pos, labels=node_labels, ax=ax)
        
        # Встраиваем в Tkinter
        canvas = FigureCanvasTkAgg(fig, master=parent_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            
    def _get_marking_label(self, marking):
        # Создаем сокращенные подписи для узлов
        place_abbr = {
            "Заявка в очереди": "P1",
            "Обработка заявки": "P2",
            "Свободные процессы": "P3",
            "Готовые заявки": "P4"
        }
        
        parts = []
        for place, tokens in marking.items():
            if tokens > 0:
                parts.append(f"{place_abbr.get(place, place)}={tokens}")
        return "\n".join(parts)


def create_workstation_model(num_workstations=5):
    net = PetriNet()
    
    # Добавляем места с новыми координатами
    net.add_place("Заявка в очереди", 0, 150, 100)  # P1
    net.add_place("Обработка заявки", 0, 350, 100)  # P2
    net.add_place("Свободные процессы", num_workstations, 150, 250)  # P3
    net.add_place("Готовые заявки", 0, 250, 400)  # P5
    
    # Добавляем переходы с новыми координатами
    net.add_transition("T1", 50, 100)   # Добавить заявку
    net.add_transition("T2", 250, 100)  # Начать обработку
    net.add_transition("T3", 300, 250)  # Завершить обработку
    net.add_transition("T4", 400, 400)  # Выдать результат (смещено вниз)
    
    # Добавляем дуги
    net.add_arc("T1", "Заявка в очереди")  # T1 → P1
    net.add_arc("Заявка в очереди", "T2")  # P1 → T2
    net.add_arc("T2", "Обработка заявки")  # T2 → P2
    net.add_arc("Обработка заявки", "T3")  # P2 → T3
    net.add_arc("Свободные процессы", "T2")  # P3 → T2
    net.add_arc("T3", "Свободные процессы")  # T3 → P3
    net.add_arc("T3", "Готовые заявки")  # T3 → P5
    net.add_arc("Готовые заявки", "T4")  # P5 → T4
    
    return net


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Эмулятор сети Петри - Система обработки заявок")
    
    net = create_workstation_model(5)
    app = PetriNetVisualizer(root, net)
    
    root.mainloop()