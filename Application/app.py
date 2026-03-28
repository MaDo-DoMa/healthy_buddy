import customtkinter as ctk
import database as db
from config import COLORS, PHASES
from camera import CameraFeed


class FocusApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Focus Assistant")
        self.geometry("820x780")
        self.minsize(700, 680)
        self.configure(fg_color=COLORS["bg"])

        self._phase = "work"
        self._running = False
        self._after_id = None
        self._pomodoros = 0
        self._cycle = 0

        self._mins = {
            "work": int(db.db_load_setting("min_work", 25)),
            "short": int(db.db_load_setting("min_short", 5)),
            "long": int(db.db_load_setting("min_long", 15)),
        }
        self._seconds_left = self._mins["work"] * 60

        self._task_widgets = {}
        self._build_ui()
        self._load_tasks()
        self.protocol("WM_DELETE_WINDOW", self._on_close)  # ✅ moved here

    def _build_ui(self):
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=2)
        self.grid_rowconfigure(0, weight=1)

        # --- LEFT PANEL: TIMER ---
        left = ctk.CTkFrame(self, fg_color=COLORS["panel"], corner_radius=18,
                            border_width=1, border_color=COLORS["border"])
        left.grid(row=0, column=0, padx=(18, 9), pady=18, sticky="nsew")
        left.grid_columnconfigure(0, weight=1)
        left.grid_rowconfigure(4, weight=1)

        self.phase_label = ctk.CTkLabel(left, text="FOCUS", font=("Courier New", 13, "bold"),
                                        text_color=COLORS["accent"])
        self.phase_label.grid(row=0, column=0, pady=(28, 0))

        self.timer_label = ctk.CTkLabel(left, text=self._fmt(self._seconds_left),
                                        font=("Courier New", 72, "bold"), text_color=COLORS["text"])
        self.timer_label.grid(row=1, column=0, pady=(4, 0))

        # Dots
        self.dots_frame = ctk.CTkFrame(left, fg_color="transparent")
        self.dots_frame.grid(row=2, column=0, pady=6)
        self._dot_labels = []
        for i in range(4):
            d = ctk.CTkLabel(self.dots_frame, text="●", font=("Arial", 14), text_color=COLORS["border"])
            d.grid(row=0, column=i, padx=5)
            self._dot_labels.append(d)

        # Buttons
        btn_frame = ctk.CTkFrame(left, fg_color="transparent")
        btn_frame.grid(row=3, column=0, pady=18)

        self.start_btn = ctk.CTkButton(btn_frame, text="▶ START", fg_color=COLORS["accent"],
                                       command=self._toggle_timer, corner_radius=10, width=130, height=42)
        self.start_btn.grid(row=0, column=0, padx=8)
        ctk.CTkButton(btn_frame, text="↺ RESET", fg_color=COLORS["border"],
                      command=self._reset_timer, width=110, height=42).grid(row=0, column=1, padx=8)
        ctk.CTkButton(btn_frame, text="⏭ SKIP", fg_color=COLORS["border"],
                      command=self._skip_phase, width=110, height=42).grid(row=0, column=2, padx=8)

        # Settings
        settings_frame = ctk.CTkFrame(left, fg_color=COLORS["bg"], corner_radius=12,
                                      border_width=1, border_color=COLORS["border"])
        settings_frame.grid(row=4, column=0, padx=24, pady=(0, 12), sticky="sew")
        settings_frame.grid_columnconfigure((0, 1, 2), weight=1)

        self._min_vars = {}
        for col, (lbl, key) in enumerate(
                [("FOCUS (min)", "work"), ("PRZERWA (min)", "short"), ("DŁUGA (min)", "long")]):
            ctk.CTkLabel(settings_frame, text=lbl, font=("Courier New", 10),
                         text_color=COLORS["muted"]).grid(row=0, column=col, pady=(10, 2))
            var = ctk.StringVar(value=str(self._mins[key]))
            self._min_vars[key] = var
            e = ctk.CTkEntry(settings_frame, textvariable=var, width=55, justify="center",
                             fg_color=COLORS["panel"])
            e.grid(row=1, column=col, padx=12, pady=(0, 12))
            e.bind("<Return>", lambda event, k=key: self._apply_settings(k))
            e.bind("<FocusOut>", lambda event, k=key: self._apply_settings(k))

        self.pom_label = ctk.CTkLabel(left, text="Dzisiaj: 0 sesji", font=("Courier New", 11),
                                      text_color=COLORS["muted"])
        self.pom_label.grid(row=5, column=0, pady=(4, 8))

        # Camera thumbnail  ✅
        self.cam_label = ctk.CTkLabel(left, text="no camera", width=220, height=140,
                                      fg_color=COLORS["bg"], corner_radius=10)
        self.cam_label.grid(row=6, column=0, pady=(0, 18))

        self._camera = CameraFeed(self.cam_label)
        if not self._camera.start():
            self.cam_label.configure(text="⚠ camera unavailable")

        # --- RIGHT PANEL: TO-DO ---
        right = ctk.CTkFrame(self, fg_color=COLORS["panel"], corner_radius=18,
                             border_width=1, border_color=COLORS["border"])
        right.grid(row=0, column=1, padx=(9, 18), pady=18, sticky="nsew")
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(2, weight=1)

        ctk.CTkLabel(right, text="TO-DO", font=("Courier New", 13, "bold"),
                     text_color=COLORS["accent2"]).grid(row=0, column=0, pady=(22, 8))

        input_frame = ctk.CTkFrame(right, fg_color="transparent")
        input_frame.grid(row=1, column=0, padx=16, pady=(0, 10), sticky="ew")
        input_frame.grid_columnconfigure(0, weight=1)

        self.task_entry = ctk.CTkEntry(input_frame, placeholder_text="Nowe zadanie...", fg_color=COLORS["bg"])
        self.task_entry.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self.task_entry.bind("<Return>", lambda e: self._add_task())

        ctk.CTkButton(input_frame, text="+", width=36, command=self._add_task,
                      fg_color=COLORS["accent2"]).grid(row=0, column=1)

        self.task_scroll = ctk.CTkScrollableFrame(right, fg_color="transparent")
        self.task_scroll.grid(row=2, column=0, padx=10, pady=(0, 16), sticky="nsew")
        self.task_scroll.grid_columnconfigure(0, weight=1)

        ctk.CTkButton(right, text="🗑 Usuń ukończone", font=("Courier New", 11), fg_color="transparent",
                      border_width=1, border_color=COLORS["border"],
                      command=self._delete_done).grid(row=3, column=0, padx=16, pady=(0, 16), sticky="ew")

    # --- LOGIC ---
    def _fmt(self, secs):
        m, s = divmod(secs, 60)
        return f"{m:02d}:{s:02d}"

    def _toggle_timer(self):
        if self._running:
            self._running = False
            self.start_btn.configure(text="▶ START", fg_color=COLORS["accent"])
            if self._after_id: self.after_cancel(self._after_id)
        else:
            self._running = True
            self.start_btn.configure(text="⏸ PAUSE", fg_color="#7a3a2a")
            self._tick()

    def _tick(self):
        if not self._running: return
        if self._seconds_left > 0:
            self._seconds_left -= 1
            self.timer_label.configure(text=self._fmt(self._seconds_left))
            self._after_id = self.after(1000, self._tick)
        else:
            self._phase_done()

    def _phase_done(self):
        self._running = False
        self.start_btn.configure(text="▶ START", fg_color=COLORS["accent"])
        if self._phase == "work":
            self._pomodoros += 1
            self._cycle += 1
            self._update_dots()
            self.pom_label.configure(text=f"Dzisiaj: {self._pomodoros} sesji")
            self._set_phase("long" if self._cycle >= 4 else "short")
            if self._cycle >= 4: self._cycle = 0
        else:
            self._set_phase("work")
        self._flash_title()

    def _set_phase(self, phase):
        self._phase = phase
        name, color, _ = PHASES[phase]
        mins = self._mins.get(phase, PHASES[phase][2] // 60)
        self._seconds_left = mins * 60
        self.timer_label.configure(text=self._fmt(self._seconds_left))
        self.phase_label.configure(text=name, text_color=color)
        self.start_btn.configure(fg_color=color)

    def _reset_timer(self):
        self._running = False
        if self._after_id: self.after_cancel(self._after_id)
        self.start_btn.configure(text="▶ START", fg_color=COLORS["accent"])
        self._seconds_left = self._mins[self._phase] * 60
        self.timer_label.configure(text=self._fmt(self._seconds_left))

    def _skip_phase(self):
        self._phase_done()

    def _update_dots(self):
        idx = (self._cycle - 1) % 4
        for i, d in enumerate(self._dot_labels):
            d.configure(text_color=COLORS["accent"] if i <= idx and self._cycle > 0 else COLORS["border"])

    def _flash_title(self):
        self.title(f"✅ {PHASES[self._phase][0]} – Focus Assistant")
        self.after(3000, lambda: self.title("Focus Assistant"))

    def _apply_settings(self, key):
        try:
            val = max(1, min(120, int(self._min_vars[key].get())))
            self._mins[key] = val
            db.db_save_setting(f"min_{key}", val)
            self._min_vars[key].set(str(val))
            if self._phase == key and not self._running:
                self._seconds_left = val * 60
                self.timer_label.configure(text=self._fmt(self._seconds_left))
        except ValueError:
            self._min_vars[key].set(str(self._mins[key]))

    def _load_tasks(self):
        for task_id, text, done in db.db_load_tasks():
            self._render_task(task_id, text, bool(done))

    def _add_task(self):
        text = self.task_entry.get().strip()
        if text:
            task_id = db.db_add_task(text)
            self._render_task(task_id, text, False, insert_top=True)
            self.task_entry.delete(0, "end")

    def _render_task(self, task_id, text, done, insert_top=False):
        row = ctk.CTkFrame(self.task_scroll, fg_color=COLORS["bg"], corner_radius=8,
                           border_width=1, border_color=COLORS["border"])
        row.grid_columnconfigure(1, weight=1)
        var = ctk.BooleanVar(value=done)

        def on_toggle(tid=task_id, v=var):
            db.db_toggle_task(tid, int(v.get()))
            lbl.configure(text_color=COLORS["done_text"] if v.get() else COLORS["text"])

        chk = ctk.CTkCheckBox(row, variable=var, text="", width=28,
                              fg_color=COLORS["accent"], command=on_toggle)
        chk.grid(row=0, column=0, padx=(10, 4), pady=8)

        lbl = ctk.CTkLabel(row, text=text, font=("Courier New", 12),
                           text_color=COLORS["done_text"] if done else COLORS["text"], anchor="w")
        lbl.grid(row=0, column=1, sticky="ew", pady=8)

        ctk.CTkButton(row, text="✕", width=26, fg_color="transparent", text_color=COLORS["muted"],
                      command=lambda tid=task_id: self._delete_task(tid)).grid(row=0, column=2, padx=(4, 8))

        if insert_top:
            row.grid(row=0, column=0, sticky="ew", padx=4, pady=3)
            self._regrid_tasks()
        else:
            row.grid(row=len(self._task_widgets), column=0, sticky="ew", padx=4, pady=3)

        self._task_widgets[task_id] = {"frame": row, "var": var, "label": lbl}

    def _delete_task(self, task_id):
        db.db_delete_task(task_id)
        self._task_widgets[task_id]["frame"].destroy()
        del self._task_widgets[task_id]
        self._regrid_tasks()

    def _delete_done(self):
        for tid in [tid for tid, w in self._task_widgets.items() if w["var"].get()]:
            self._delete_task(tid)

    def _regrid_tasks(self):
        for i, w in enumerate(self._task_widgets.values()):
            w["frame"].grid(row=i, column=0, sticky="ew", padx=4, pady=3)

    def _on_close(self):  # ✅ properly inside class
        self._camera.stop()
        self.destroy()