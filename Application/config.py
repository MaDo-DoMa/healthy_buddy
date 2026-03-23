import os

DB_PATH = os.path.join(os.path.expanduser("~"), ".focus_assistant.db")

COLORS = {
    "bg":        "#0f0f13",
    "panel":     "#16161e",
    "border":    "#2a2a3a",
    "accent":    "#e8643c",
    "accent2":   "#4c9be8",
    "green":     "#3ecf74",
    "text":      "#e8e8f0",
    "muted":     "#6b6b80",
    "done_text": "#3a3a4a",
}

PHASES = {
    "work":     ("FOCUS", COLORS["accent"], 25*60),
    "short":    ("PRZERWA", COLORS["green"], 5*60),
    "long":     ("DŁUGA PRZERWA", COLORS["accent2"], 15*60),
}