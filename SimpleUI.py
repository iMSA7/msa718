import os
import tkinter as tk
from PIL import Image, ImageTk

is_trained_tech = True
is_trained_sent = True


def run_tech_model():
    stock_symbol = entry.get()
    if stock_symbol:
        os.system(f"python TechnicalAnalysis/FNNTechnical.py {stock_symbol} no")
        # display on a new window
        new_window = tk.Toplevel(root)
        new_window.title("Technical Model Prediction")
        # get image
        image = ImageTk.PhotoImage(Image.open("technical.png"))
        # load image
        panel = tk.Label(new_window, image=image)
        panel.image = image
        panel.pack()


def run_sent_model():
    stock_symbol = entry.get()
    if stock_symbol:
        os.system(f"python SentimentAnalysis/FNNSentiment.py {stock_symbol} no")
        # display on a new window
        new_window = tk.Toplevel(root)
        new_window.title("Sentiment Model Prediction")
        # get image
        image = ImageTk.PhotoImage(Image.open("sentiment.png"))
        # load image
        panel = tk.Label(new_window, image=image)
        panel.image = image
        panel.pack()


def run_classifiers():
    os.system("python SentimentAnalysis/SentimentAnalysis.py")
    read_data = ''
    with open("classifiers", 'r') as f:
        read_data = f.read()
    window = tk.Tk()
    window.title("Classifiers Labelling")
    canvas = tk.Canvas(window, height=300, width=600, bg='#263D42')
    canvas.pack()

    frame = tk.Frame(window, bg='white')
    frame.place(relwidth=0.8, relheight=0.8, relx=0.1, rely=0.1)

    lbl = tk.Label(frame, text=read_data, font=("arial italic", 18))
    lbl.pack()
    os.system("rm classifiers")


def train_tech():
    global is_trained_tech
    try:
        os.system("python TechnicalAnalysis/FNNTechnical.py N.A yes")
        if not is_trained_tech:
            tech_model = tk.Button(frame2, text="Technical Analysis Model", padx=50, pady=10, fg='black', bg='#263D42',
                                   command=run_tech_model)
            tech_model.pack()
            is_trained_tech = True
    except Exception:
        print("Error training the technical analysis model")

    read_data = ''
    with open("Training_Technical", 'r') as f:
        read_data = f.read()
    window = tk.Tk()
    window.title("Technical Analysis Training Parameters")
    canvas = tk.Canvas(window, height=300, width=600, bg='#263D42')
    canvas.pack()

    frame = tk.Frame(window, bg='white')
    frame.place(relwidth=0.8, relheight=0.8, relx=0.1, rely=0.1)

    lbl = tk.Label(frame, text=read_data, font=("arial italic", 18))
    lbl.pack()
    os.system("rm Training_Technical")
    # display on a new window
    new_window = tk.Toplevel(root)
    new_window.title("Technical Model Loss")
    # get image
    image = ImageTk.PhotoImage(Image.open("tt.png"))
    # load image
    panel = tk.Label(new_window, image=image)
    panel.image = image
    panel.pack()


def train_sent():
    global is_trained_sent
    try:
        os.system(f"python SentimentAnalysis/FNNSentiment.py N.A yes")
        if not is_trained_sent:
            sent_model = tk.Button(frame2, text="Sentiment Analysis Model", padx=50, pady=10, fg='black', bg='#263D42',
                                   command=run_sent_model)
            sent_model.pack()
            is_trained_sent = True

    except Exception:
        print("Error training the sentiment analysis model")

    read_data = ''
    with open("Training_Sentiment", 'r') as f:
        read_data = f.read()
    window = tk.Tk()
    window.title("Sentiment Analysis Training Parameters")
    canvas = tk.Canvas(window, height=300, width=600, bg='#263D42')
    canvas.pack()

    frame = tk.Frame(window, bg='white')
    frame.place(relwidth=0.8, relheight=0.8, relx=0.1, rely=0.1)

    lbl = tk.Label(frame, text=read_data, font=("arial italic", 18))
    lbl.pack()
    os.system("rm Training_Sentiment")
    # display on a new window
    new_window = tk.Toplevel(root)
    new_window.title("Sentiment Model Loss")
    # get image
    image = ImageTk.PhotoImage(Image.open("st.png"))
    # load image
    panel = tk.Label(new_window, image=image)
    panel.image = image
    panel.pack()


root = tk.Tk()
root.title("Main")
canvas = tk.Canvas(root, height=700, width=700, bg='#263D42')
canvas.pack()

frame = tk.Frame(root, bg='gray')
frame.place(relwidth=0.8, relheight=0.8, relx=0.1, rely=0.1)

lbl = tk.Label(frame, text="Enter Stock Symbol", font=("arial italic", 12))
lbl.pack()
entry = tk.Entry(frame)
entry.pack()

frame2 = tk.Frame(frame, bg='gray')
frame2.place(relwidth=0.5, relheight=0.4, relx=0.25, rely=0.2)

label = tk.Label(frame2, text="Choose the Model to Predict:", bg='gray', font=("arial italic", 18))
label.pack()

try:
    with open('TechnicalAnalysis/techModel.sav') as f:
        print("technical model is trained!")
except IOError:
    is_trained_tech = False

try:
    with open('SentimentAnalysis/sentModel.sav') as f:
        print("sentiment model is trained!")
except IOError:
    is_trained_sent = False

if is_trained_tech:
    tech_model = tk.Button(frame2, text="Technical Analysis Model", padx=50, pady=10, fg='black', bg='#263D42', command=run_tech_model)
    tech_model.pack()

if is_trained_sent:
    sent_model = tk.Button(frame2, text="Sentiment Analysis Model", padx=50, pady=10, fg='black', bg='#263D42', command=run_sent_model)
    sent_model.pack()

sent_classifiers = tk.Button(frame2, text="Candidate ML Classifiers", padx=50, pady=10, fg='black', bg='#263D42', command=run_classifiers)
sent_classifiers.pack()

frame3 = tk.Frame(frame, bg='gray')
frame3.place(relwidth=0.5, relheight=0.4, relx=0.25, rely=0.5)
label2 = tk.Label(frame3, text="Choose the Model to Train:", bg='gray', font=("arial italic", 18))
label2.pack()

train_tech_model = tk.Button(frame3, text="Technical Analysis Model", padx=50, pady=10, fg='black', bg='#263D42', command=train_tech)
train_tech_model.pack()

train_sent_model = tk.Button(frame3, text="Sentiment Analysis Model", padx=50, pady=10, fg='black', bg='#263D42', command=train_sent)
train_sent_model.pack()

root.mainloop()
