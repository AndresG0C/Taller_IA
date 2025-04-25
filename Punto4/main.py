import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import numpy as np
from sklearn.linear_model import Perceptron, SGDClassifier
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class AplicacionRedNeuronal:
    def __init__(self, root):
        self.root = root
        self.root.title("Red Neuronal para Control de Robot")

        # Pantalla completa 1920x1080
        try:
            self.root.state('zoomed')
            self.root.attributes('-fullscreen', True)
        except:
            self.root.geometry("1920x1080")

        # Datos de entrenamiento
        self.datos_entrenamiento = [
            [1, 1, 1, -1, -1, 0, 0],    # Detener
            [-1, 1, 1, -1, 1, 0, 1],    # Girar derecha
            [1, 1, -1, 1, -1, 1, 0],    # Girar izquierda
            [-1, -1, -1, 1, 1, 1, 1]    # Avanzar
        ]

        self.acciones = {
            (0, 0): "DETENER",
            (0, 1): "DERECHA",
            (1, 0): "IZQUIERDA",
            (1, 1): "AVANZAR"
        }

        self.entradas_neuronas = []
        self.historial_perdida = []
        self.historial_precision = []
        self.configurar_interfaz()

    def configurar_interfaz(self):
        marco_principal = ttk.Frame(self.root, padding=10)
        marco_principal.pack(fill="both", expand=True)

        marco_config = ttk.Frame(marco_principal, width=350)
        marco_config.pack(side="left", fill="y", padx=10)

        marco_resultados = ttk.Frame(marco_principal)
        marco_resultados.pack(side="right", fill="both", expand=True)

        ttk.Label(marco_config, text="Configuración del Modelo", font=("Arial", 14, "bold")).pack(pady=10)

        marco_tipo_modelo = ttk.Frame(marco_config)
        marco_tipo_modelo.pack(fill="x", pady=5)
        ttk.Label(marco_tipo_modelo, text="Tipo de Modelo:").pack(side="left")
        self.tipo_modelo = tk.StringVar(value="perceptron")
        ttk.Radiobutton(marco_tipo_modelo, text="Perceptrón", variable=self.tipo_modelo, value="perceptron", command=self.actualizar_interfaz).pack(side="left", padx=10)
        ttk.Radiobutton(marco_tipo_modelo, text="Adaline", variable=self.tipo_modelo, value="adaline", command=self.actualizar_interfaz).pack(side="left")
        ttk.Radiobutton(marco_tipo_modelo, text="MLP", variable=self.tipo_modelo, value="mlp", command=self.actualizar_interfaz).pack(side="left", padx=10)

        self.marco_capas = ttk.Frame(marco_config)
        self.marco_capas.pack(fill="x", pady=5)
        ttk.Label(self.marco_capas, text="Capas Ocultas:").pack(side="left")
        self.num_capas = tk.StringVar(value="1")
        ttk.Entry(self.marco_capas, textvariable=self.num_capas, width=5).pack(side="left", padx=5)
        ttk.Button(self.marco_capas, text="Configurar", command=self.configurar_neuronas).pack(side="left")

        self.marco_config_neuronas = ttk.Frame(marco_config)
        self.marco_config_neuronas.pack(fill="x", pady=5)

        ttk.Label(marco_config, text="Parámetros de Entrenamiento", font=("Arial", 12)).pack(pady=(15,5), anchor="w")

        self.campos = {}
        parametros = [
            ("Tasa Aprendizaje", "0.01"),
            ("Máx. Iteraciones", "1000"),
            ("Tolerancia", "1e-4"),
            ("Función Activación", "relu"),
            ("Algoritmo", "adam")
        ]

        for param, default in parametros:
            marco = ttk.Frame(marco_config)
            marco.pack(fill="x", pady=2)
            ttk.Label(marco, text=param+":").pack(side="left")
            if param in ["Función Activación", "Algoritmo"]:
                opciones = ["relu", "tanh", "logistic"] if param == "Función Activación" else ["adam", "sgd", "lbfgs"]
                combo = ttk.Combobox(marco, values=opciones, state="readonly")
                combo.set(default)
                combo.pack(side="right", expand=True, fill="x")
                self.campos[param] = combo
            else:
                entrada = ttk.Entry(marco)
                entrada.insert(0, default)
                entrada.pack(side="right", expand=True, fill="x")
                self.campos[param] = entrada

        ttk.Label(marco_config, text="Datos de Entrenamiento", font=("Arial", 12)).pack(pady=(15,5), anchor="w")
        texto_datos = scrolledtext.ScrolledText(marco_config, height=10, font=("Courier", 9))
        texto_datos.pack(fill="x", pady=5)
        texto_datos.insert(tk.END, "S1 S2 S3 M1 M2 | Acción\n")
        texto_datos.insert(tk.END, "----------------------\n")
        for dato in self.datos_entrenamiento:
            accion = self.acciones.get((dato[5], dato[6]), "DESCONOCIDA")
            texto_datos.insert(tk.END, f"{dato[0]:2} {dato[1]:2} {dato[2]:2} {dato[3]:2} {dato[4]:2} | {accion}\n")
        texto_datos.config(state="disabled")

        ttk.Button(marco_config, text="Entrenar Modelo", command=self.entrenar_modelo).pack(pady=20)

        self.texto_resultados = scrolledtext.ScrolledText(marco_resultados, height=15, font=("Courier", 10))
        self.texto_resultados.pack(fill="both", expand=True, pady=10)

        marco_graficos = ttk.Frame(marco_resultados)
        marco_graficos.pack(fill="both", expand=True)

        marco_precision = ttk.Frame(marco_graficos)
        marco_precision.pack(side="left", fill="both", expand=True, padx=5)
        self.fig_precision = plt.Figure(figsize=(6, 4), dpi=100)
        self.ejes_precision = self.fig_precision.add_subplot(111)
        self.lienzo_precision = FigureCanvasTkAgg(self.fig_precision, master=marco_precision)
        self.lienzo_precision.get_tk_widget().pack(fill="both", expand=True)

        marco_perdida = ttk.Frame(marco_graficos)
        marco_perdida.pack(side="right", fill="both", expand=True, padx=5)
        self.fig_perdida = plt.Figure(figsize=(6, 4), dpi=100)
        self.ejes_perdida = self.fig_perdida.add_subplot(111)
        self.lienzo_perdida = FigureCanvasTkAgg(self.fig_perdida, master=marco_perdida)
        self.lienzo_perdida.get_tk_widget().pack(fill="both", expand=True)

        self.actualizar_interfaz()

    def actualizar_interfaz(self):
        if self.tipo_modelo.get() == "mlp":
            self.marco_capas.pack(fill="x", pady=5)
            self.marco_config_neuronas.pack(fill="x", pady=5)
            self.configurar_neuronas()
        else:
            self.marco_capas.pack_forget()
            self.marco_config_neuronas.pack_forget()

    def configurar_neuronas(self):
        for widget in self.marco_config_neuronas.winfo_children():
            widget.destroy()
        self.entradas_neuronas = []
        try:
            num_capas = int(self.num_capas.get())
            if num_capas < 1:
                raise ValueError("Debe haber al menos 1 capa.")
            ttk.Label(self.marco_config_neuronas, text="Neuronas por Capa:").pack(anchor="w")
            for i in range(num_capas):
                marco = ttk.Frame(self.marco_config_neuronas)
                marco.pack(fill="x", pady=2)
                ttk.Label(marco, text=f"Capa {i+1}:").pack(side="left")
                entrada = ttk.Entry(marco)
                entrada.insert(0, "5")
                entrada.pack(side="right", expand=True, fill="x")
                self.entradas_neuronas.append(entrada)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def entrenar_modelo(self):
        self.texto_resultados.delete(1.0, tk.END)
        self.ejes_precision.clear()
        self.ejes_perdida.clear()
        self.historial_perdida.clear()
        self.historial_precision.clear()

        X = np.array([dato[:5] for dato in self.datos_entrenamiento])
        y = np.array([dato[5]*2 + dato[6] for dato in self.datos_entrenamiento])

        try:
            tasa = float(self.campos["Tasa Aprendizaje"].get())
            iteraciones = int(self.campos["Máx. Iteraciones"].get())
            tolerancia = float(self.campos["Tolerancia"].get())
            tipo = self.tipo_modelo.get()

            if tipo == "perceptron":
                modelo = Perceptron(eta0=tasa, max_iter=iteraciones, tol=tolerancia, random_state=42)
                nombre = "Perceptrón"
            elif tipo == "adaline":
                modelo = SGDClassifier(loss='squared_error', learning_rate='constant', eta0=tasa,
                                       max_iter=iteraciones, tol=tolerancia, penalty=None, random_state=42)
                nombre = "Adaline"
            else:
                capas = tuple(int(ent.get()) for ent in self.entradas_neuronas)
                activacion = self.campos["Función Activación"].get()
                algoritmo = self.campos["Algoritmo"].get()
                modelo = MLPClassifier(hidden_layer_sizes=capas, activation=activacion, solver=algoritmo,
                                       learning_rate_init=tasa, max_iter=iteraciones, tol=tolerancia, random_state=42)
                nombre = f"MLP ({len(capas)} capas)"

            modelo.fit(X, y)
            y_pred = modelo.predict(X)

            precision = np.mean(y == y_pred) * 100
            self.historial_precision.append(precision)

            self.texto_resultados.insert(tk.END, f"Resultados de {nombre}:\n\n")
            self.texto_resultados.insert(tk.END, f"Precisión: {precision:.2f}%\n")

            # Graficar precisión (línea verde)
            self.ejes_precision.plot(self.historial_precision, marker='o', color='green', label='Precisión')
            self.ejes_precision.set_ylim(0, 100)
            self.ejes_precision.set_title("Precisión del Modelo")
            self.ejes_precision.legend()

            # Graficar pérdida (línea roja si existe)
            if hasattr(modelo, 'loss_curve_'):
                self.ejes_perdida.plot(modelo.loss_curve_, color='red', linestyle='-', marker='x', label="Pérdida")
                self.ejes_perdida.set_title("Curva de Pérdida")
                self.ejes_perdida.legend()

            self.lienzo_precision.draw()
            self.lienzo_perdida.draw()

        except Exception as e:
            messagebox.showerror("Error", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = AplicacionRedNeuronal(root)
    root.mainloop()
