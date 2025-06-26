import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
import os

class ImageProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ImagePro")
        self.root.geometry("1400x800")  
        self.root.configure(bg="#1c2526")  

        self.image = None
        self.image_path = None
        self.processed_image = None
        self.last_process = "Asli"
        self.output_dir = "output_images"
        os.makedirs(self.output_dir, exist_ok=True)

        # Definisi font dan gaya
        title_font = ("Helvetica", 16, "bold")
        label_font = ("Helvetica", 12, "bold")
        button_font = ("Helvetica", 10, "bold")
        btn_style = {
            "bg": "#0288d1",  # Warna biru modern
            "fg": "white",
            "font": button_font,
            "width": 22,
            "height": 2,
            "relief": "flat",
            "activebackground": "#0277bd",  # Warna lebih gelap saat diklik
            "activeforeground": "white"
        }

        # Kontainer utama
        main_frame = tk.Frame(root, bg="#1c2526")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        # Header
        header_frame = tk.Frame(main_frame, bg="#0288d1")
        header_frame.pack(fill=tk.X, pady=(0, 10))
        header_label = tk.Label(
            header_frame,
            text="ImagePro",
            font=title_font,
            bg="#0288d1",
            fg="white",
            pady=10
        )
        header_label.pack()

        # Kontainer konten
        content_frame = tk.Frame(main_frame, bg="#1c2526")
        content_frame.pack(fill=tk.BOTH, expand=True)

        # Frame kiri untuk kanvas
        left_frame = tk.Frame(content_frame, bg="#1c2526")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Frame kanan untuk tombol
        right_frame = tk.Frame(content_frame, bg="#1c2526", width=250)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))

        # Frame untuk kanvas
        canvas_frame = tk.Frame(left_frame, bg="#1c2526")
        canvas_frame.pack(pady=10)

        # Frame gambar asli
        input_frame = tk.Frame(canvas_frame, bg="#2e2e2e", bd=2, relief="groove")
        input_frame.pack(side=tk.LEFT, padx=15, pady=5)
        self.label_input = tk.Label(
            input_frame,
            text="Gambar Asli",
            font=label_font,
            bg="#2e2e2e",
            fg="#ffffff",
            pady=5
        )
        self.label_input.pack()
        self.canvas_input = tk.Canvas(
            input_frame,
            width=512,
            height=384,
            bg="#424242",
            highlightthickness=0
        )
        self.canvas_input.pack(pady=5)

        # Frame gambar hasil
        output_frame = tk.Frame(canvas_frame, bg="#2e2e2e", bd=2, relief="groove")
        output_frame.pack(side=tk.LEFT, padx=15, pady=5)
        self.label_output = tk.Label(
            output_frame,
            text="Hasil Gambar",
            font=label_font,
            bg="#2e2e2e",
            fg="#ffffff",
            pady=5
        )
        self.label_output.pack()
        self.canvas_output = tk.Canvas(
            output_frame,
            width=512,
            height=384,
            bg="#424242",
            highlightthickness=0
        )
        self.canvas_output.pack(pady=5)

        # Bilah status
        self.status_var = tk.StringVar()
        self.status_var.set("Status: Tidak ada gambar")
        status_bar = tk.Label(
            main_frame,
            textvariable=self.status_var,
            bg="#1c2526",
            fg="#b0bec5",
            font=("Helvetica", 10),
            anchor="w",
            padx=10
        )
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)

        # Daftar tombol dengan efek hover
        buttons = [
            ("Pilih Gambar", self.load_image),
            ("Konversi Grayscale", self.to_grayscale),
            ("Binerisasi", self.to_binary),
            ("Arithmetic (Add)", self.arithmetic_add),
            ("Logical (AND)", self.logical_and),
            ("Tampilkan Histogram", self.show_histogram),
            ("Sharpening Filter", self.apply_sharpening),
            ("Dilasi Morfologi", self.apply_dilation),
            ("Simpan Hasil", self.save_output),
            ("Hapus Gambar", self.clear_images)
        ]

        for text, command in buttons:
            btn = tk.Button(right_frame, text=text, command=command, **btn_style)
            btn.pack(pady=6, padx=10)
            # Efek hover
            btn.bind("<Enter>", lambda e, b=btn: b.configure(bg="#03a9f4"))
            btn.bind("<Leave>", lambda e, b=btn: b.configure(bg="#0288d1"))

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("File gambar", "*.jpg *.jpeg *.png")])
        if file_path:
            try:
                img = cv2.imread(file_path)
                if img is None or img.shape[0] == 0 or img.shape[1] == 0:
                    raise ValueError("Gambar tidak valid atau rusak.")
                self.image_path = file_path
                self.image = img
                self.display_image(self.image, self.canvas_input)
                self.processed_image = None
                self.last_process = "Asli"
                self.canvas_output.delete("all")
                self.status_var.set(f"Status: Gambar dimuat - {os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("Kesalahan", f"Gagal memuat gambar: {str(e)}")
                self.clear_images()

    def display_image(self, img, canvas):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 else img
        h, w = img_rgb.shape[:2]
        scale = min(512 / w, 384 / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img_rgb, (new_w, new_h))
        pil_img = Image.fromarray(resized)
        photo = ImageTk.PhotoImage(pil_img)
        canvas.create_image(256, 192, image=photo)
        canvas.image = photo

    def to_grayscale(self):
        if self.image is None:
            messagebox.showerror("Kesalahan", "Silakan muat gambar terlebih dahulu!")
            return
        self.processed_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.last_process = "Grayscale"
        self.display_image(self.processed_image, self.canvas_output)
        self.status_var.set("Status: Konversi ke Grayscale selesai")

    def to_binary(self):
        if self.image is None:
            messagebox.showerror("Kesalahan", "Silakan muat gambar terlebih dahulu!")
            return
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        _, self.processed_image = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        self.last_process = "Binerisasi"
        self.display_image(self.processed_image, self.canvas_output)
        self.status_var.set("Status: Binerisasi selesai")

    def arithmetic_add(self):
        if self.image is None:
            messagebox.showerror("Kesalahan", "Silakan muat gambar terlebih dahulu!")
            return
        constant = np.full_like(self.image, 50)
        self.processed_image = cv2.add(self.image, constant)
        self.last_process = "Arithmetic (Add)"
        self.display_image(self.processed_image, self.canvas_output)
        self.status_var.set("Status: Operasi Arithmetic (Add) selesai")

    def logical_and(self):
        if self.image is None:
            messagebox.showerror("Kesalahan", "Silakan muat gambar terlebih dahulu!")
            return
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        self.processed_image = cv2.bitwise_and(gray, binary)
        self.last_process = "Logical (AND)"
        self.display_image(self.processed_image, self.canvas_output)
        self.status_var.set("Status: Operasi Logical (AND) selesai")

    def show_histogram(self):
        if self.image is None and self.processed_image is None:
            messagebox.showerror("Kesalahan", "Silakan muat atau proses gambar terlebih dahulu!")
            return
        target_img = self.processed_image if self.processed_image is not None else self.image
        if len(target_img.shape) == 3:
            target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
        plt.figure(figsize=(8, 4))
        plt.hist(target_img.ravel(), 256, [0, 256], color='blue')
        plt.title(f"Histogram Gambar ({self.last_process})")
        plt.xlabel("Nilai Piksel")
        plt.ylabel("Jumlah Piksel")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        self.status_var.set(f"Status: Histogram untuk {self.last_process} ditampilkan")

    def apply_sharpening(self):
        if self.image is None:
            messagebox.showerror("Kesalahan", "Silakan muat gambar terlebih dahulu!")
            return
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        self.processed_image = cv2.filter2D(self.image, -1, kernel)
        self.last_process = "Sharpening Filter"
        self.display_image(self.processed_image, self.canvas_output)
        self.status_var.set("Status: Sharpening Filter selesai")

    def apply_dilation(self):
        if self.image is None:
            messagebox.showerror("Kesalahan", "Silakan muat gambar terlebih dahulu!")
            return
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        se1 = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        se2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilate1 = cv2.dilate(gray, se1, iterations=1)
        self.processed_image = cv2.dilate(dilate1, se2, iterations=1)
        self.last_process = "Dilasi Morfologi"
        self.display_image(self.processed_image, self.canvas_output)
        self.status_var.set("Status: Dilasi Morfologi selesai")

    def save_output(self):
        if self.processed_image is None:
            messagebox.showerror("Kesalahan", "Tidak ada gambar hasil untuk disimpan!")
            return
        filename = filedialog.asksaveasfilename(
            initialdir=self.output_dir,
            defaultextension=".png",
            filetypes=[("File PNG", "*.png"), ("File JPEG", "*.jpg")]
        )
        if filename:
            cv2.imwrite(filename, self.processed_image)
            messagebox.showinfo("Sukses", "Gambar berhasil disimpan!")
            self.status_var.set(f"Status: Gambar disimpan di {os.path.basename(filename)}")

    def clear_images(self):
        self.canvas_input.delete("all")
        self.canvas_output.delete("all")
        self.image = None
        self.processed_image = None
        self.image_path = None
        self.last_process = "Asli"
        self.status_var.set("Status: Tidak ada gambar")
        messagebox.showinfo("Gambar Dihapus", "Gambar asli dan hasil telah dihapus dari tampilan.")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessingApp(root)
    root.mainloop()