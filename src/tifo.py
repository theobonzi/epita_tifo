import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from PIL import Image, ImageTk
from filters import (apply_gray, apply_binary,
                    apply_histogram_equalization, apply_negative, apply_pixelize,
                    apply_sobel, apply_prewitt, apply_laplace,
                    blur_image_with_opencl, erode_image_with_opencl, dilate_image_with_opencl)

class App:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("CAN's Real-Time Video Stream App")
        self.window.geometry("1280x800")

        self.cap = cv2.VideoCapture(0)

        self.filter_status = {"gray": False,
                              "binary": False,
                              "histogram_eq": False,
                              "negative": False,
                              "pixelize": False,
                              "sobel": False,
                              "laplace": False,
                              "prewitt": False,
                              "blur": False,
                              "erode": False,
                              "dilate": False
}
                             
        self.filters = {"gray": apply_gray,
                        "binary": lambda img: apply_binary(img, self.threshold),
                        "histogram_eq": apply_histogram_equalization,
                        "negative": apply_negative,
                        "pixelize": lambda img: apply_pixelize(img, int(self.pixelize_value.get())),
                        "sobel": apply_sobel,
                        "laplace": apply_laplace,
                        "prewitt": apply_prewitt,
                        "blur": blur_image_with_opencl,
                        "erode": erode_image_with_opencl,
                        "dilate": dilate_image_with_opencl
}

        self.vline = None
        self.buttons = {}

        self.button_colors = {
            True: "green",
            False: "gray"
        }
        
        self.filter_list = [name for name, active in self.filter_status.items() if active]

        self.blur_value = tk.IntVar(value=15)
        self.pixelize_value = tk.IntVar(value=15)
        self.contrast_value = tk.IntVar(value=0)
        self.saturation_value = tk.IntVar(value=0)
        self.nb_color = tk.IntVar(value=10)

        self._create_widgets()
        
        self.update_image()
        self.update_histogram()
        self.window.mainloop()
        
        
    def apply_filters(self, frame, exclude_filters=None):
        if exclude_filters is None:
            exclude_filters = []
        
        for filter_name in self.filter_list:
            if self.filter_status[filter_name] and filter_name not in exclude_filters:
                filter_func = self.filters[filter_name]
                frame = filter_func(frame)
        return frame

    @property
    def threshold(self):
        if self.vline is not None:
            return self.vline.get_xdata()[0]
        else:
            return None
        
    def _create_widgets(self):
        top_frame = tk.Frame(self.window)
        top_frame.pack(side=tk.TOP)

        self.label_image = tk.Label(top_frame)
        self.label_image.pack(side=tk.RIGHT)

        filters_frame = tk.Frame(top_frame)
        filters_frame.pack(side=tk.LEFT, padx=10)

        for index, filter_name in enumerate(self.filter_status.keys()):
            if index % 7 == 0:
                filters_frame = tk.Frame(top_frame)
                filters_frame.pack(side=tk.LEFT, padx=10)

            button = tk.Button(filters_frame, text=f"{filter_name.capitalize()} Filter", 
                               command=lambda name=filter_name: self.toggle_filter(name), 
                               bg=self.button_colors[False], relief=tk.RAISED)
            button.pack(side=tk.TOP, fill=tk.X)
            self.buttons[filter_name] = button

            if (filter_name == "pixelize"):
                pixelize_scale = tk.Scale(filters_frame, from_=10, to=50, orient="horizontal", 
                                variable=self.pixelize_value, label="Pixelize value")
                pixelize_scale.pack(side=tk.TOP, fill=tk.X)
            elif (filter_name == "reduce_color"):
                reduce_color_scale = tk.Scale(filters_frame, from_=2, to=30, orient="horizontal", 
                                variable=self.nb_color, label="Number of color")
                reduce_color_scale.pack(side=tk.TOP, fill=tk.X)

        self.filter_listbox = tk.Listbox(filters_frame)
        for filter_name in self.filter_list:
            self.filter_listbox.insert(tk.END, filter_name)
        self.filter_listbox.pack(side=tk.TOP, fill=tk.X)

        up_button = tk.Button(filters_frame, text='Up', command=self.move_up)
        up_button.pack()
        down_button = tk.Button(filters_frame, text='Down', command=self.move_down)
        down_button.pack()

        bottom_frame = tk.Frame(self.window)
        bottom_frame.pack(side=tk.TOP, pady=10)

        self.fig = plt.Figure(figsize=(6, 8), dpi=100)
        self.plot = self.fig.add_subplot(1, 1, 1)

        self.canvas = FigureCanvasTkAgg(self.fig, master=bottom_frame)
        self.canvas.get_tk_widget().pack()
        self.canvas.mpl_connect('button_press_event', self.on_mouse_press)

    def move_up(self):
        selected = self.filter_listbox.curselection()
        if not selected:
            return
        index = selected[0]
        filter_name = self.filter_listbox.get(index)
        
        if filter_name not in self.filter_list:
            return
        if index == 0:
            return

        self.filter_listbox.delete(index)
        self.filter_listbox.insert(index - 1, filter_name)
        self.filter_listbox.selection_set(index - 1)

        self.filter_list.remove(filter_name)
        self.filter_list.insert(index - 1, filter_name)
    
    def move_down(self):
        selected = self.filter_listbox.curselection()
        if not selected:
            return
        index = selected[0]
        filter_name = self.filter_listbox.get(index)

        if filter_name not in self.filter_list:
            return
        if index == len(self.filter_list) - 1:
            return

        self.filter_listbox.delete(index)
        self.filter_listbox.insert(index + 1, filter_name)
        self.filter_listbox.selection_set(index + 1)

        self.filter_list.remove(filter_name)
        self.filter_list.insert(index + 1, filter_name)
    
    def toggle_filter(self, filter_name):
        self.filter_status[filter_name] = not self.filter_status[filter_name]

        new_status = self.filter_status[filter_name]
        self.buttons[filter_name].config(bg=self.button_colors[new_status])
        
        if self.filter_status[filter_name]:
            self.filter_listbox.insert(tk.END, filter_name)
            self.filter_list.append(filter_name)
        else: 
            index = self.filter_list.index(filter_name)
            self.filter_listbox.delete(index)
            self.filter_list.remove(filter_name)

    def update_image(self):
        _, frame = self.cap.read()
        frame = cv2.flip(frame, 1)
        frame = self.apply_filters(frame)

        frame_tk = self._convert_frame_to_tk(frame)

        self.label_image.config(image=frame_tk)
        self.label_image.image = frame_tk

        self.window.after(100, self.update_image)

    def _convert_frame_to_tk(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        frame_tk = ImageTk.PhotoImage(frame_pil)
        return frame_tk

    def on_mouse_press(self, event):
        if self.vline is not None:
            self.vline.set_xdata([event.xdata])
            self.canvas.draw()

    def update_histogram(self):
        _, frame = self.cap.read()

        frame_histogram = self.apply_filters(frame, exclude_filters=['binary'])

        channels = cv2.split(frame_histogram)
        colors = ('b', 'g', 'r')

        histograms = []
        for color, channel in zip(colors, channels):
            histogram = cv2.calcHist([channel], [0], None, [256], [0, 256])
            histograms.append(histogram)

        max_value = max([np.max(histogram) for histogram in histograms])

        self.plot.clear()

        for color, histogram in zip(colors, histograms):
            self.plot.plot(histogram/max_value, color=color)

        if self.vline is None:
            self.vline = self.plot.axvline(x=0, color='black', linestyle='--')
        else:
            self.vline.remove()
            self.vline = self.plot.axvline(x=self.vline.get_xdata()[0], color='black', linestyle='--')

        self.plot.set_title('Histogramme pour chaque canal de couleur')
        self.plot.set_xlabel('Intensité de couleur')
        self.plot.set_ylabel('Nombre de pixels')

        self.canvas.draw()

        self.window.after(100, self.update_histogram)
        
if __name__ == "__main__":
    App()
