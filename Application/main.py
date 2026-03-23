import customtkinter as ctk
from database import init_db
from app import FocusApp

def main():
    init_db()

    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")

    app = FocusApp()
    app.mainloop()

if __name__ == "__main__":
    main()