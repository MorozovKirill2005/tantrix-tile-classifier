import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from pathlib import Path
from processing import processing


class Window:
    def __init__(self, root):
        self.root = root
        self.root.title("Лабораторная работа 2: Обработка изображений")
        self.root.geometry("1000x600")

        self.create_menu()
        self.create_area_for_images()
    
    def create_menu(self, save: bool=False):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Меню", menu=file_menu)
        file_menu.add_command(label="Загрузить изображение", command=self.load_image)
        if save and self.group:
            file_menu.add_command(label="Сохранить обработанное изображение", command=self.save_result)
        file_menu.add_separator()
        file_menu.add_command(label="Выход", command=self.root.quit)

    def create_area_for_images(self):
        main_frame = tk.Frame(self.root)
        main_frame.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

        left_frame = tk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=5)
        left_frame_title = tk.Frame(left_frame)
        left_frame_title.pack(pady=5)
        self.left_title = tk.Label(left_frame_title, text="Исходное изображение")
        self.left_title.pack(expand=True, anchor='center')
        self.left = tk.Label(left_frame)
        self.left.pack(expand=True, fill=tk.BOTH, padx=5, pady=5)

        right_frame = tk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=5)
        right_frame_title = tk.Frame(right_frame)
        right_frame_title.pack(pady=5)
        self.right_title = tk.Label(right_frame_title, text="Результат работы алгоритма")
        self.right_title.pack(expand=True, anchor='center')
        self.right = tk.Label(right_frame)
        self.right.pack(expand=True, fill=tk.BOTH, padx=5, pady=5)

    def load_image(self):
        image_path = filedialog.askopenfilename(
            title="Выберите изображение",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        if not image_path:
            return
        try:
            self.left.config(image='', text='')
            self.right.config(image='', text='')
            self.right_title.config(text="")
            self.left.image = None
            self.right.image = None
            self.root.update()
            self.image = Image.open(image_path)
            filename = Path(image_path).name
            if filename[:7] == "Single_":
                self.group = False
            elif filename[:6] == "Group_":
                self.group = True
            else:
                raise ValueError("Исходный файл должен начинаться с префикса 'Single' или 'Group'")
            self.result = processing(self.image, self.group)
            self.display_result()
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить или обработать изображение:\n{str(e)}")

    def display_result(self):
        max_width, max_height = 400, 300

        img = self.image.copy()
        img.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(img)
        self.left.config(image=self.tk_image, text="")
        self.left.image = self.tk_image

        if self.group:
            img = self.result.copy()
            img.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
            self.tk_processed_image = ImageTk.PhotoImage(img)
            self.right.config(image=self.tk_processed_image, text="")
            self.right.image = self.tk_processed_image
            self.right_title.config(text=f"Изображение с номерами фишек")
        else:
            self.right.config(text=f"Номер фишки: {self.result}", font=("Arial", 28), fg="black")
            self.right_title.config(text="Результат распознавания")

        self.create_menu(save=True)

    def save_result(self):
        image_path = filedialog.asksaveasfilename(
            title="Сохранить обработанное изображение",
            defaultextension=".bmp",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        if not image_path:
            return
        try:
            self.result.save(image_path)
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось сохранить изображение:\n{str(e)}")


root = tk.Tk()
app = Window(root)
root.mainloop()
