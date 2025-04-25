import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import json
import numpy as np
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class Punto3App:
    def __init__(self, root):
        self.root = root
        self.root.title("Punto 3 - Red Neuronal Multisalida (S y C)")
        self.neuron_entries = []

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

        # Configuración
        ttk.Label(left_frame, text="Configuración del Modelo", font=("Arial", 14)).pack(pady=10)
        self.fields = {}

        def add_entry(label, default):
            frame = ttk.Frame(left_frame)
            frame.pack(fill="x", pady=2)
            ttk.Label(frame, text=label).pack(side="left")
            entry = ttk.Entry(frame)
            entry.insert(0, default)
            entry.pack(side="right", expand=True, fill="x")
            self.fields[label] = entry

        def add_combo(label, options, default):
            frame = ttk.Frame(left_frame)
            frame.pack(fill="x", pady=2)
            ttk.Label(frame, text=label).pack(side="left")
            combo = ttk.Combobox(frame, values=options, state="readonly")
            combo.set(default)
            combo.pack(side="right", expand=True, fill="x")
            self.fields[label] = combo

        add_entry("Número de capas ocultas", "1")
        ttk.Button(left_frame, text="Configurar neuronas por capa", command=self.configurar_neuronas).pack(pady=5)
        self.neuron_config_frame = ttk.Frame(left_frame)
        self.neuron_config_frame.pack(fill="x", pady=5)

        add_combo("Función de activación", ["relu", "tanh", "logistic"], "tanh")
        add_combo("Algoritmo de entrenamiento", ["adam", "sgd", "lbfgs"], "adam")
        add_entry("Tasa de aprendizaje", "0.01")
        add_entry("Iteraciones máximas", "1000")
        add_entry("Tolerancia (error objetivo)", "1e-6")
        add_entry("Ruta archivo JSON", "datos.json")

        ttk.Button(left_frame, text="Entrenar y Simular", command=self.entrenar_red).pack(pady=20)

        # Resultados
        self.output_text = scrolledtext.ScrolledText(right_frame, height=12, font=("Courier", 10))
        self.output_text.pack(fill="x", pady=10)

        graphs_frame = ttk.Frame(right_frame)
        graphs_frame.pack(fill="both", expand=True)

        # Error por patrón
        error_frame = ttk.Frame(graphs_frame)
        error_frame.pack(side="left", fill="both", expand=True, padx=5)

        self.error_fig = plt.Figure(figsize=(6, 4), dpi=100)
        self.error_ax = self.error_fig.add_subplot(111)
        self.error_canvas = FigureCanvasTkAgg(self.error_fig, master=error_frame)
        self.error_canvas.get_tk_widget().pack(fill="both", expand=True)

        # Gráfica de loss
        loss_frame = ttk.Frame(graphs_frame)
        loss_frame.pack(side="right", fill="both", expand=True, padx=5)

        self.loss_fig = plt.Figure(figsize=(6, 4), dpi=100)
        self.loss_ax = self.loss_fig.add_subplot(111)
        self.loss_canvas = FigureCanvasTkAgg(self.loss_fig, master=loss_frame)
        self.loss_canvas.get_tk_widget().pack(fill="both", expand=True)

    def configurar_neuronas(self):
        for widget in self.neuron_config_frame.winfo_children():
            widget.destroy()
        self.neuron_entries = []

        try:
            num_capas = int(self.fields["Número de capas ocultas"].get())
            if num_capas < 1:
                raise ValueError("Debe haber al menos una capa.")
            ttk.Label(self.neuron_config_frame, text="Neuronas por capa:").pack(anchor="w")
            for i in range(num_capas):
                frame = ttk.Frame(self.neuron_config_frame)
                frame.pack(fill="x", pady=2)
                ttk.Label(frame, text=f"Capa {i+1}:").pack(side="left")
                entry = ttk.Entry(frame)
                entry.insert(0, "5")
                entry.pack(side="right", expand=True, fill="x")
                self.neuron_entries.append(entry)
        except Exception as e:
            messagebox.showerror("Error", f"Error al configurar capas: {str(e)}")

    def entrenar_red(self):
        self.output_text.delete(1.0, tk.END)
        self.error_ax.clear()
        self.loss_ax.clear()

        try:
            capas = [int(entry.get()) for entry in self.neuron_entries]
            activacion = self.fields["Función de activación"].get()
            solver = self.fields["Algoritmo de entrenamiento"].get()
            tasa = float(self.fields["Tasa de aprendizaje"].get())
            iteraciones = int(self.fields["Iteraciones máximas"].get())
            tolerancia = float(self.fields["Tolerancia (error objetivo)"].get())
            ruta = self.fields["Ruta archivo JSON"].get()

            with open(ruta, "r") as f:
                datos = json.load(f)

            X = np.array([[d["x"], d["y"]] for d in datos])
            y = np.array([[d["s"], d["c"]] for d in datos])
            y_bin = (y + 1) // 2  # Convertir [-1,1] a [0,1]

            modelo = MLPClassifier(
                hidden_layer_sizes=tuple(capas),
                activation=activacion,
                solver=solver,
                learning_rate_init=tasa,
                max_iter=iteraciones,
                tol=tolerancia,
                random_state=42
            )

            modelo.fit(X, y_bin)
            y_pred_bin = modelo.predict(X)
            y_pred = y_pred_bin * 2 - 1  # Reconvertir a [-1,1]

            # Mostrar resultados
            self.output_text.insert(tk.END, f"{'X':>4} {'Y':>4} {'S_esp':>6} {'C_esp':>6} | {'S_pred':>6} {'C_pred':>6}\n")
            self.output_text.insert(tk.END, "-"*44 + "\n")
            for i in range(len(X)):
                self.output_text.insert(tk.END, f"{X[i][0]:>4} {X[i][1]:>4} {y[i][0]:>6} {y[i][1]:>6} | {y_pred[i][0]:>6} {y_pred[i][1]:>6}\n")

            # Gráfico de loss
            self.loss_ax.plot(modelo.loss_curve_, color='green')
            self.loss_ax.set_title("Loss Curve del Entrenamiento")
            self.loss_ax.set_xlabel("Iteración")
            self.loss_ax.set_ylabel("Loss")
            self.loss_ax.grid(True)
            self.loss_canvas.draw()

            # Gráfico de error por patrón (total error por salida)
            error_total = np.abs(y - y_pred).sum(axis=1)
            self.error_ax.plot(error_total, marker='o', color='red')
            self.error_ax.set_title("Error total por Patrón")
            self.error_ax.set_xlabel("Patrón")
            self.error_ax.set_ylabel("Error absoluto total")
            self.error_ax.grid(True)
            self.error_canvas.draw()

        except Exception as e:
            self.output_text.insert(tk.END, f"[ERROR] {str(e)}\n")

# ---------------------- Ejecutar ----------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = Punto3App(root)
    root.mainloop()