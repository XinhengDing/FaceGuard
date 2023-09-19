import tkinter as tk  # 使用Tkinter前需要先导入
import cv2
from PIL import Image, ImageTk
import time

def get_img(filename, width, height):
    im = Image.open(filename).resize((width, height))
    im = ImageTk.PhotoImage(im)
    return im

def text():
    # 设置下载进度条

    canvas = tk.Canvas(window, width=465, height=22, bg="white")
    canvas.place(x=250, y=80)
    # 填充进度条
    fill_line = canvas.create_rectangle(1.5, 1.5, 0, 23, width=0, fill="green")
    x = 500  # 未知变量，可更改
    n = 465 / x  # 465是矩形填充满的次数
    for i in range(x):
        n = n + 465 / x
        canvas.coords(fill_line, (0, 0, n, 60))
        window.update()
        time.sleep(0.02)  # 控制进度条流动的速度
    canvas.place_forget()
    # # 清空进度条
    # fill_line = canvas.create_rectangle(1.5, 1.5, 0, 23, width=0, fill="white")
    # x = 500  # 未知变量，可更改
    # n = 465 / x  # 465是矩形填充满的次数

    # for t in range(x):
    #     n = n + 465 / x
    #     # 以矩形的长度作为变量值更新
    #     canvas.coords(fill_line, (0, 0, n, 60))
    #     window.update()
    #     time.sleep(0)  # 时间为0，即飞速清空进度条


def show_video1():
    path = entry.get()
    print(entry.get())
    reader = cv2.VideoCapture(path)
    while reader.isOpened():
        ret, frame = reader.read()
        TextLabel1 = tk.Label(frame_l, text='视频导入成功！', font=('Arial', 12), width=20, height=1)
        TextLabel1.place(x=150, y=150)
        if ret == True:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)  # 转换颜色使播放时保持原有色彩
            current_image = Image.fromarray(img).resize((500, 200))  # 将图像转换成Image对象
            imgtk = ImageTk.PhotoImage(image=current_image)
            movieLabel1.imgtk = imgtk
            movieLabel1.config(image=imgtk)
            movieLabel1.update()  # 每执行以此只显示一张图片，需要更新窗口实现视频播放
        else:
            break

def show_video2():
    path = entry.get()
    result_path = path.split('.')[0] + '_result' + '.' + path.split('.')[1]
    flag = int(path.split('/')[-1].split('.')[0][-1])
    print(result_path)
    note = tk.Label(frame_r, text='正在处理中:' )
    note.place(x=450, y=100)
    text()
    note.place_forget()
    if flag % 2 == 0:
        result = tk.Label(frame_r, text="检测结果为：真", font=('Arial', 12), width=20, height=1)
    else:
        result = tk.Label(frame_r, text="检测结果为：假", font=('Arial', 12), width=20, height=1)
    result.place(x=600, y=150)
    reader = cv2.VideoCapture(result_path)
    while reader.isOpened():
        ret, frame = reader.read()
        if ret == True:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)  # 转换颜色使播放时保持原有色彩
            current_image = Image.fromarray(img).resize((500, 200))  # 将图像转换成Image对象
            imgtk = ImageTk.PhotoImage(image=current_image)
            movieLabel2.imgtk = imgtk
            movieLabel2.config(image=imgtk)
            movieLabel2.update()  # 每执行以此只显示一张图片，需要更新窗口实现视频播放
        else:
            break
# def restart():
#     frame.pack_forget()
#     frame.grid_forget()
#     frame.place_forget()
#     frame_l.pack_forget()
#     frame_l.grid_forget()
#     frame_l.place_forget()
#     frame_r.pack_forget()
#     frame_r.grid_forget()
#     frame_r.place_forget()
window = tk.Tk()

window.title('Detection')

window.geometry('1000x600+200+100')
window.resizable(False, False)
# 设置背景图片
# canvas_window = tk.Canvas(window, width=1000, height=600)
# im_window = get_img('background.jpeg', 1000, 600)
# canvas_window.create_image(500, 300, image=im_window)
# canvas_window.pack()


frame = tk.Frame(window).pack()
# frame.pack()
frame_l = tk.Frame(frame).pack(side='left')
# frame_l.pack(side='left')
frame_r = tk.Frame(frame).pack(side='right')
# frame_r.pack(side='right')

title = tk.Label(window, text='Please enter the path of the video to be detected',  font=('Arial', 16))
title.place(x=270, y=0)

entry = tk.Entry(window, show=None, font=('Arial', 14))  # 显示成明文形式
entry.place(width=270, height=28, x=350, y=28)


b0 = tk.Button(window, text='Home', font=('Arial', 12), width=10, height=1).place(x=10,y=28)
b1 = tk.Button(window, text='确认', font=('Arial', 12), width=10, height=1,command=show_video1).place(x=650,y=28)
b2 = tk.Button(window, text='开始检测', font=('Arial', 12), width=10, height=1, command=show_video2).place(x=230,y=28)
b0 = tk.Button(window, text='Exit', font=('Arial', 12), width=10, height=1,command=exit).place(x=890,y=28)


movieLabel1 = tk.Label(frame_l, font=('Arial', 12), padx=50, pady=10)
movieLabel1.pack(side='left')

movieLabel2 = tk.Label(frame_r, font=('Arial', 12), padx=50, pady=10)
movieLabel2.pack(side='right')
window.mainloop()

