import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import numpy as np
import json
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ---------------------- Funciones de utilidad ----------------------

def funcion_objetivo(x, y, z):
    return np.sin(x) + np.cos(y) + z

def cargar_datos_json(ruta):
    with open(ruta, "r") as f:
        datos = json.load(f)
    X = np.array([[d["x"], d["y"], d["z"]] for d in datos])
    y = np.array([funcion_objetivo(*fila) for fila in X])
    return X, y

# ---------------------- Clase de la aplicación ----------------------

class RNAApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Simulación de Red Neuronal - Taller Inteligencia Artificial")
        self.neuron_entries = []  # Para almacenar los campos de entrada de neuronas

        try:
            self.root.state('zoomed')
        except:
            self.root.geometry("1920x1080")

        self.setup_ui()

    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill="both", expand=True)

        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side="left", fill="y", padx=10)

        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side="right", fill="both", expand=True)

        ttk.Label(left_frame, text="Configuración del Modelo", font=("Arial", 14)).pack(pady=10)
        self.config_fields = {}

        def add_entry(label, default):
            frame = ttk.Frame(left_frame)
            frame.pack(fill="x", pady=2)
            ttk.Label(frame, text=label).pack(side="left")
            entry = ttk.Entry(frame)
            entry.insert(0, default)
            entry.pack(side="right", expand=True, fill="x")
            self.config_fields[label] = entry

        def add_combobox(label, options, default):
            frame = ttk.Frame(left_frame)
            frame.pack(fill="x", pady=2)
            ttk.Label(frame, text=label).pack(side="left")
            combo = ttk.Combobox(frame, values=options, state="readonly")
            combo.set(default)
            combo.pack(side="right", expand=True, fill="x")
            self.config_fields[label] = combo

        # Campo para número de capas
        add_entry("Número de capas ocultas", "3")
        
        # Botón para configurar neuronas por capa
        ttk.Button(left_frame, text="Configurar neuronas por capa", 
                  command=self.configurar_neuronas).pack(pady=5)
        
        # Frame para las entradas de neuronas por capa
        self.neuron_config_frame = ttk.Frame(left_frame)
        self.neuron_config_frame.pack(fill="x", pady=5)

        add_combobox("Función de activación", ["relu", "tanh", "logistic"], "tanh")
        add_combobox("Algoritmo de entrenamiento", ["adam", "sgd", "lbfgs"], "adam")
        add_entry("Tasa de aprendizaje", "0.001")
        add_entry("Iteraciones máximas", "150")
        add_entry("Tolerancia (error objetivo)", "1e-8")
        add_entry("Ruta del archivo JSON", "datos.json")

        ttk.Button(left_frame, text="Entrenar y Simular", command=self.entrenar_red).pack(pady=20)

        # --- Área de resultados ---
        self.output_text = scrolledtext.ScrolledText(right_frame, height=20, font=("Courier", 10))
        self.output_text.pack(fill="x", pady=10)

        # Frame para las gráficas (una al lado de la otra)
        graphs_frame = ttk.Frame(right_frame)
        graphs_frame.pack(fill="both", expand=True)

        # Gráfica de errores (izquierda)
        error_frame = ttk.Frame(graphs_frame)
        error_frame.pack(side="left", fill="both", expand=True, padx=5)

        self.figure = plt.Figure(figsize=(6, 4), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("Error por Patrón")
        self.ax.set_xlabel("Patrón")
        self.ax.set_ylabel("Error")

        self.canvas = FigureCanvasTkAgg(self.figure, master=error_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # Gráfica de performance (derecha)
        loss_frame = ttk.Frame(graphs_frame)
        loss_frame.pack(side="right", fill="both", expand=True, padx=5)

        self.loss_figure = plt.Figure(figsize=(6, 4), dpi=100)
        self.loss_ax = self.loss_figure.add_subplot(111)
        self.loss_ax.set_title("Performance de Entrenamiento (Loss Curve)")
        self.loss_ax.set_xlabel("Iteración")
        self.loss_ax.set_ylabel("Loss")

        self.loss_canvas = FigureCanvasTkAgg(self.loss_figure, master=loss_frame)
        self.loss_canvas.get_tk_widget().pack(fill="both", expand=True)

    def configurar_neuronas(self):
        # Limpiar frame de configuración previa
        for widget in self.neuron_config_frame.winfo_children():
            widget.destroy()
        
        self.neuron_entries = []  # Reiniciar la lista de entradas
        
        try:
            num_capas = int(self.config_fields["Número de capas ocultas"].get())
            if num_capas < 1:
                raise ValueError("Debe haber al menos 1 capa")
            
            ttk.Label(self.neuron_config_frame, text="Neuronas por capa:").pack(anchor="w")
            
            for i in range(num_capas - 1):  # Dejar la última capa fuera
                frame = ttk.Frame(self.neuron_config_frame)
                frame.pack(fill="x", pady=2)
                ttk.Label(frame, text=f"Capa {i+1}:").pack(side="left")
                entry = ttk.Entry(frame)
                entry.insert(0, "100")  # Valor por defecto
                entry.pack(side="right", expand=True, fill="x")
                self.neuron_entries.append(entry)
            
            # No se permite que el usuario configure la última capa, que debe tener el número de salidas
            ttk.Label(self.neuron_config_frame, text=f"Capa {num_capas}:").pack(side="left", anchor="w")
            ttk.Label(self.neuron_config_frame, text=f"{len(self.config_fields['Ruta del archivo JSON'].get())}").pack(side="left")
            
        except ValueError as e:
            messagebox.showerror("Error", f"Valor inválido: {str(e)}")


    def entrenar_red(self):
        self.output_text.delete(1.0, tk.END)
        self.ax.clear()
        self.loss_ax.clear()

        try:
            # Obtener configuración de capas
            if not self.neuron_entries:
                raise ValueError("Primero configure las neuronas por capa")
                
            capas = []
            for entry in self.neuron_entries[:-1]:  # No contar la última capa
                if entry:
                    neuronas = int(entry.get())
                    if neuronas < 1:
                        raise ValueError("Cada capa debe tener al menos 1 neurona")
                    capas.append(neuronas)

            # La última capa será igual al número de salidas
            capas.append(len(self.config_fields["Ruta del archivo JSON"].get()))  # Número de salidas
            
            activacion = self.config_fields["Función de activación"].get()
            solver = self.config_fields["Algoritmo de entrenamiento"].get()
            tasa = float(self.config_fields["Tasa de aprendizaje"].get())
            iteraciones = int(self.config_fields["Iteraciones máximas"].get())
            tolerancia = float(self.config_fields["Tolerancia (error objetivo)"].get())
            ruta = self.config_fields["Ruta del archivo JSON"].get()

            X, y = cargar_datos_json(ruta)

            modelo = MLPRegressor(
                hidden_layer_sizes=tuple(capas),
                activation=activacion,
                solver=solver,
                learning_rate_init=tasa,
                max_iter=iteraciones,
                tol=tolerancia,
                random_state=42
            )

            modelo.fit(X, y)
            predicciones = modelo.predict(X)
            errores = y - predicciones

            self.output_text.insert(tk.END, "\n")
            self.output_text.insert(tk.END, f"{'Real':>10} | {'Predicho':>10} | {'Error':>10}\n")
            self.output_text.insert(tk.END, "-" * 36 + "\n")
            for i in range(len(y)):
                self.output_text.insert(tk.END, f"{y[i]:10.4f} | {predicciones[i]:10.4f} | {errores[i]:10.4f}\n")

            # Gráfica de errores
            self.ax.plot(errores, marker='o', linestyle='-', color='blue')
            self.ax.set_title("Error por Patrón")
            self.ax.set_xlabel("Patrón")
            self.ax.set_ylabel("Error")
            self.ax.grid(True)
            self.canvas.draw()

            # Gráfica de loss
            self.loss_ax.plot(modelo.loss_curve_, color='green')
            self.loss_ax.set_title("Performance de Entrenamiento (Loss Curve)")
            self.loss_ax.set_xlabel("Iteración")
            self.loss_ax.set_ylabel("Loss")
            self.loss_ax.grid(True)
            self.loss_canvas.draw()

        except Exception as e:
            self.output_text.insert(tk.END, f"\n[ERROR] {str(e)}\n")


# ---------------------- Ejecutar ----------------------

if __name__ == "__main__":
    root = tk.Tk()
    app = RNAApp(root)
    root.mainloop()