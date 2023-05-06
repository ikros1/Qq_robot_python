import math
import time
import tkinter as tk
import numpy as np
from PIL import Image, ImageTk


class EllipseApp:
    def __init__(self, face_arr):
        self.face_arr = face_arr
        self.root = tk.Tk()
        self.canvas = tk.Canvas(self.root, width=1000, height=784)
        self.canvas.pack()
        self.oval = self.canvas.create_oval(225, 225, 275, 275, fill='pink')
        self.index = 0
        self.max = len(face_arr)
        img = Image.open("img/ika.jpg")
        img = img.resize((1000, 784), Image.ANTIALIAS)
        self.bg = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.bg)

        # 每隔5毫秒调用一次self.change_oval_height()函数
        self.root.after(250, self.change_oval_height)

        self.root.mainloop()

    def change_oval_height(self):
        self.canvas.delete(self.oval)
        # hei = self.face_arr[self.index]/10
        hei = self.face_arr[self.index] / 7
        self.index += 1
        center = 303
        pas = math.floor(hei / 2)
        new_y1 = center - pas
        new_y2 = center + pas
        self.oval = self.canvas.create_oval(495, new_y1, 505, new_y2, fill='#F395B3', outline='#F395B3')
        # self.oval = self.canvas.create_oval(497, new_y1, 503, new_y2, fill='#F395B3', outline='#F395B3')
        if self.index < self.max:
            # 继续每隔5毫秒调用一次self.change_oval_height()函数
            self.root.after(21, self.change_oval_height)
        else:
            # 数组结束，调用stop()方法来关闭窗口
            self.stop()

    def stop(self):
        self.root.destroy()
