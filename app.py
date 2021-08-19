from tkinter import *
from chat import get_reponse, bot_name

# Global variables
BACKGROUND = '#295b5e'
TEXT_COLOUR = '#ffffff'
FONT = 'Courier 14'
FONT_BOLD = "Courier 13 bold"

class ChatBotApplication:
    def __init__(self):
        self.window = Tk()
        self.setup_chatbot_window()

    def run(self):
        self.window.mainloop()

    def setup_chatbot_window(self):
        self.window.title("ChatBot")
        self.window.resizable(width=False, height=False)
        self.window.configure(width=700, height=600, bg=BACKGROUND)

        head_label = Label(self.window, bg=BACKGROUND, fg=TEXT_COLOUR, 
            text="Welcome", font=FONT_BOLD, pady=10)
        head_label.place(relwidth=1)

        line = Label(self.window, width=450, bg=BACKGROUND)
        line.place(relwidth=1, rely=0.07, relheight=0.012)

        # Text widget
        self.text_widget = Text(self.window, width=20, height=2, bg=BACKGROUND, fg=TEXT_COLOUR,
            font=FONT, padx=5, pady=5)
        self.text_widget.place(relheight=0.745, relwidth=1, rely=0.08)
        self.text_widget.configure(cursor="arrow", state=DISABLED)

        # Scroll bar
        scroll_bar = Scrollbar(self.text_widget)
        scroll_bar.place(relheight=1, relx=0.974)
        scroll_bar.configure(command=self.text_widget.yview)

        # Bottom label widget
        bottom_label = Label(self.window, bg=BACKGROUND, height=80)
        bottom_label.place(relwidth=1, rely=0.825)

        # Message entry box
        self.msg_entry = Entry(bottom_label, bg=BACKGROUND, fg=TEXT_COLOUR, font=FONT)
        self.msg_entry.place(relwidth=0.74, relheight=0.06, rely=0.008, relx=0.011)
        self.msg_entry.focus()
        self.msg_entry.bind("<Return>", self.on_enter)

        # Send button
        send_button = Button(bottom_label, text="Send", font=FONT_BOLD, width=20, bg=BACKGROUND,
            command=lambda: self.on_enter(None))
        send_button.place(relx=0.77, rely=0.008, relheight=0.06, relwidth=0.22)

    def on_enter(self, event):
        msg = self.msg_entry.get()
        self.insert_message(msg, "You")

    def insert_message(self, msg, sender):
        if not msg:
            Return

        self.msg_entry.delete(0, END)
        msg1 = f"{sender}: {msg}\n\n"
        self.text_widget.configure(state=NORMAL)
        self.text_widget.insert(END, msg1)
        self.text_widget.configure(state=DISABLED)
        
        msg2 = f"{bot_name}: {get_reponse(msg)}\n\n"
        self.text_widget.configure(state=NORMAL)
        self.text_widget.insert(END, msg2)
        self.text_widget.configure(state=DISABLED)

        self.text_widget.see(END)

if __name__ == "__main__":
    app = ChatBotApplication()
    app.run()