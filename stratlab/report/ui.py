"""Enhanced interactive UI for viewing backtesting results."""

import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from datetime import datetime
import pandas as pd


class BacktestUI:
    """Professional UI for switching between multiple backtest visualizations."""

    def __init__(self, figures: list[tuple[str, Figure]], metrics_data: dict = None) -> None:
        """
        Initialize the backtest UI.

        Args:
            figures: List of (name, Figure) tuples to display
            metrics_data: Optional dict with summary metrics
        """
        self.figures = figures
        self.metrics_data = metrics_data or {}
        self.current_idx = 0

        # Create root window
        self.root = tk.Tk()
        self.root.title("Backtest Results Viewer")
        self.root.geometry("1400x900")

        # Create main layout
        self._create_layout()
        self._bind_keys()
        self._update_view()

    def _create_layout(self) -> None:
        """Create the main UI layout with canvas and controls."""
        # Top info bar
        info_frame = tk.Frame(self.root, bg="#f0f0f0", height=40)
        info_frame.pack(fill=tk.X, padx=0, pady=0)
        info_frame.pack_propagate(False)

        self.title_label = tk.Label(
            info_frame,
            text="",
            font=("Arial", 14, "bold"),
            bg="#f0f0f0",
        )
        self.title_label.pack(side=tk.LEFT, padx=20, pady=8)

        self.counter_label = tk.Label(
            info_frame,
            text="",
            font=("Arial", 10),
            bg="#f0f0f0",
        )
        self.counter_label.pack(side=tk.RIGHT, padx=20, pady=8)

        # Canvas frame for matplotlib
        canvas_frame = tk.Frame(self.root)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.canvas = FigureCanvasTkAgg(
            self.figures[0][1],
            master=canvas_frame,
        )
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Toolbar
        toolbar_frame = tk.Frame(self.root)
        toolbar_frame.pack(fill=tk.X, padx=10)
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()

        # Navigation controls
        nav_frame = tk.Frame(self.root, bg="#f0f0f0", height=60)
        nav_frame.pack(fill=tk.X, padx=0, pady=0)
        nav_frame.pack_propagate(False)

        # Left controls
        left_frame = tk.Frame(nav_frame, bg="#f0f0f0")
        left_frame.pack(side=tk.LEFT, padx=15, pady=10)

        btn_prev = tk.Button(
            left_frame,
            text="◀ Previous",
            command=self._prev,
            width=12,
            height=1,
            font=("Arial", 10),
        )
        btn_prev.pack(side=tk.LEFT, padx=5)

        btn_next = tk.Button(
            left_frame,
            text="Next ▶",
            command=self._next,
            width=12,
            height=1,
            font=("Arial", 10),
        )
        btn_next.pack(side=tk.LEFT, padx=5)

        # Center controls
        center_frame = tk.Frame(nav_frame, bg="#f0f0f0")
        center_frame.pack(side=tk.LEFT, expand=True, padx=15, pady=10)

        tk.Label(
            center_frame,
            text="Jump to:",
            bg="#f0f0f0",
            font=("Arial", 10),
        ).pack(side=tk.LEFT, padx=5)

        self.figure_combo = ttk.Combobox(
            center_frame,
            values=[f[0] for f in self.figures],
            state="readonly",
            width=30,
            font=("Arial", 10),
        )
        self.figure_combo.pack(side=tk.LEFT, padx=5)
        self.figure_combo.bind("<<ComboboxSelected>>", self._on_combo_select)

        # Right controls
        right_frame = tk.Frame(nav_frame, bg="#f0f0f0")
        right_frame.pack(side=tk.RIGHT, padx=15, pady=10)

        btn_info = tk.Button(
            right_frame,
            text="ℹ Info",
            command=self._show_info,
            width=8,
            height=1,
            font=("Arial", 10),
        )
        btn_info.pack(side=tk.LEFT, padx=5)

        btn_quit = tk.Button(
            right_frame,
            text="✕ Quit",
            command=self.root.destroy,
            width=8,
            height=1,
            font=("Arial", 10),
        )
        btn_quit.pack(side=tk.LEFT, padx=5)

    def _bind_keys(self) -> None:
        """Bind keyboard shortcuts."""
        self.root.bind("<Left>", lambda e: self._prev())
        self.root.bind("<Right>", lambda e: self._next())
        self.root.bind("<q>", lambda e: self.root.destroy())
        self.root.bind("<Escape>", lambda e: self.root.destroy())
        self.root.bind("<Home>", lambda e: self._first())
        self.root.bind("<End>", lambda e: self._last())

    def _get_label(self) -> str:
        """Get the label for the current figure."""
        name = self.figures[self.current_idx][0]
        return f"{name} ({self.current_idx + 1}/{len(self.figures)})"

    def _update_view(self) -> None:
        """Update the displayed figure and labels."""
        self.canvas.figure = self.figures[self.current_idx][1]
        self.canvas.draw()

        label = self._get_label()
        self.title_label.config(text=self.figures[self.current_idx][0])
        self.counter_label.config(text=f"{self.current_idx + 1} / {len(self.figures)}")
        self.figure_combo.set(self.figures[self.current_idx][0])

    def _prev(self) -> None:
        """Go to previous figure."""
        self.current_idx = (self.current_idx - 1) % len(self.figures)
        self._update_view()

    def _next(self) -> None:
        """Go to next figure."""
        self.current_idx = (self.current_idx + 1) % len(self.figures)
        self._update_view()

    def _first(self) -> None:
        """Go to first figure."""
        self.current_idx = 0
        self._update_view()

    def _last(self) -> None:
        """Go to last figure."""
        self.current_idx = len(self.figures) - 1
        self._update_view()

    def _on_combo_select(self, event) -> None:
        """Handle combobox selection."""
        selected = self.figure_combo.get()
        for idx, (name, _) in enumerate(self.figures):
            if name == selected:
                self.current_idx = idx
                self._update_view()
                break

    def _show_info(self) -> None:
        """Show metrics info dialog."""
        if not self.metrics_data:
            messagebox.showinfo("Info", "No metrics data available.")
            return

        info_text = "Backtest Metrics Summary\n" + "=" * 40 + "\n\n"
        for key, value in self.metrics_data.items():
            if isinstance(value, float):
                if "return" in key.lower() or "drawdown" in key.lower():
                    info_text += f"{key}: {value:.2%}\n"
                else:
                    info_text += f"{key}: {value:.4f}\n"
            else:
                info_text += f"{key}: {value}\n"

        messagebox.showinfo("Backtest Metrics", info_text)

    def show(self) -> None:
        """Display the UI window."""
        self.root.mainloop()
