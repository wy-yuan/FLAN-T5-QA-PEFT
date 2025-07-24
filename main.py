import tkinter as tk
from tkinter import ttk, font
import tkinter.scrolledtext as scrolledtext
from datetime import datetime
import threading
import time
import re
from model_evaluate import *

class ModernChatbotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("AI assistant")
        self.root.geometry("450x700")
        self.root.minsize(350, 500)

        # Configure style
        self.setup_styles()

        # Set dark theme colors
        self.bg_color = "#feffef"
        self.secondary_bg = "#d6d6d6"
        self.accent_color = "#00b4d8"
        self.text_color = "#292929"
        self.user_msg_bg = "#d2e9be"
        self.bot_msg_bg = "#e6e6e6"

        self.root.configure(bg=self.bg_color)

        # Create main container
        self.create_header()
        self.create_chat_area()
        self.create_input_area()

        # Add welcome message
        self.add_message("bot", "Hello! I'm your AI assistant. How can I help you today?")

        # Focus on input
        self.message_input.focus_set()

        # Bind window resize event
        self.root.bind("<Configure>", self.on_window_resize)

    def setup_styles(self):
        """Configure ttk styles for modern look"""
        self.style = ttk.Style()
        self.style.theme_use('clam')

        # Configure custom fonts
        self.header_font = font.Font(family="Helvetica", size=16, weight="bold")
        self.chat_font = font.Font(family="Helvetica", size=11)
        self.input_font = font.Font(family="Helvetica", size=12)

    def create_header(self):
        """Create modern header with title and status"""
        header_frame = tk.Frame(self.root, bg=self.secondary_bg, height=60)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)

        # Title
        title_label = tk.Label(
            header_frame,
            text="AI assistant",
            font=self.header_font,
            bg=self.secondary_bg,
            fg=self.text_color
        )
        title_label.pack(side=tk.LEFT, padx=20, pady=15)

        # Status indicator
        self.status_frame = tk.Frame(header_frame, bg=self.secondary_bg)
        self.status_frame.pack(side=tk.RIGHT, padx=20, pady=20)

        self.status_dot = tk.Canvas(
            self.status_frame,
            width=10,
            height=10,
            bg=self.secondary_bg,
            highlightthickness=0
        )
        self.status_dot.pack(side=tk.LEFT, padx=(0, 5))
        self.status_dot.create_oval(0, 0, 10, 10, fill="#005c00", outline="")

        self.status_label = tk.Label(
            self.status_frame,
            text="Online",
            font=("Helvetica", 10),
            bg=self.secondary_bg,
            fg="#006d00"
        )
        self.status_label.pack(side=tk.LEFT)

    def create_chat_area(self):
        """Create the main chat display area"""
        # Chat container
        chat_container = tk.Frame(self.root, bg=self.bg_color)
        chat_container.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)

        # Create canvas and scrollbar for chat messages
        self.canvas = tk.Canvas(chat_container, bg=self.bg_color, highlightthickness=0)
        scrollbar = ttk.Scrollbar(chat_container, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas, bg=self.bg_color)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)

        # Pack canvas and scrollbar
        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Mouse wheel scrolling
        self.canvas.bind_all("<MouseWheel>", lambda e: self.canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"))

    def create_input_area(self):
        """Create modern input area with text field and send button"""
        input_container = tk.Frame(self.root, bg=self.secondary_bg, height=80)
        input_container.pack(fill=tk.X, side=tk.BOTTOM)
        input_container.pack_propagate(False)

        input_frame = tk.Frame(input_container, bg=self.secondary_bg)
        input_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=15)

        # Create rounded input field appearance
        input_bg = tk.Frame(input_frame, bg="#ffffff", highlightbackground="#ececec", highlightthickness=2)
        input_bg.pack(fill=tk.X, side=tk.LEFT, expand=True, padx=(0, 10))

        self.message_input = tk.Text(
            input_bg,
            height=2,
            font=self.input_font,
            bg="#ffffff",
            fg=self.text_color,
            insertbackground=self.text_color,
            relief=tk.FLAT,
            wrap=tk.WORD,
            padx=10,
            pady=8
        )
        self.message_input.pack(fill=tk.BOTH, expand=True)
        self.message_input.bind("<Return>", self.send_message)
        self.message_input.bind("<Shift-Return>", lambda e: None)

        # Create modern send button
        self.send_button = tk.Button(
            input_frame,
            text="Send",
            font=("Helvetica", 12, "bold"),
            bg=self.accent_color,
            fg="white",
            activebackground="#0096c7",
            activeforeground="white",
            relief=tk.FLAT,
            padx=20,
            pady=10,
            cursor="hand2",
            command=lambda: self.send_message(None)
        )
        self.send_button.pack(side=tk.RIGHT)

        # Placeholder text
        self.add_placeholder()
        self.message_input.bind("<FocusIn>", self.remove_placeholder)
        self.message_input.bind("<FocusOut>", self.add_placeholder)

    def add_placeholder(self, event=None):
        """Add placeholder text to input field"""
        if self.message_input.get("1.0", "end-1c") == "":
            self.message_input.insert("1.0", "Type a message...")
            self.message_input.config(fg="#808080")

    def remove_placeholder(self, event=None):
        """Remove placeholder text from input field"""
        if self.message_input.get("1.0", "end-1c") == "Type a message...":
            self.message_input.delete("1.0", tk.END)
            self.message_input.config(fg=self.text_color)

    def create_rounded_rectangle(self, canvas, x1, y1, x2, y2, radius=15, **kwargs):
        """Create a rounded rectangle on canvas"""
        points = []
        for x, y in [(x1, y1 + radius), (x1, y1), (x1 + radius, y1),
                     (x2 - radius, y1), (x2, y1), (x2, y1 + radius),
                     (x2, y2 - radius), (x2, y2), (x2 - radius, y2),
                     (x1 + radius, y2), (x1, y2), (x1, y2 - radius)]:
            points.extend([x, y])
        return canvas.create_polygon(points, smooth=True, **kwargs)
    
    def create_avatar(self, parent, sender, size=35):
        """Create avatar for user or bot"""
        avatar_frame = tk.Frame(parent, bg=self.bg_color, width=size, height=size)
        avatar_frame.pack_propagate(False)
        
        avatar_canvas = tk.Canvas(
            avatar_frame, 
            width=size, 
            height=size, 
            bg=self.bg_color, 
            highlightthickness=0
        )
        avatar_canvas.pack()
        
        if sender == "user":
            # User avatar - circle with initial
            avatar_canvas.create_oval(
                2, 2, size-2, size-2,
                fill="#b5ce70",
                outline="#5B693B",
                width=2
            )
            avatar_canvas.create_text(
                size//2, size//2,
                text="'.'",
                font=("Helvetica", size//2, "bold"),
                fill="black"
            )
        else:
            # Bot avatar - rounded square with bot icon
            self.create_rounded_rectangle(
                avatar_canvas,
                2, 2, size-2, size-2,
                radius=8,
                fill="#b1b1b1",
                outline="#404040",
                width=2
            )
            # Simple bot icon (two dots for eyes)
            eye_size = 3
            eye_y = size//2 - 3
            avatar_canvas.create_oval(
                size//2 - 7, eye_y,
                size//2 - 7 + eye_size, eye_y + eye_size,
                fill="#004400", outline=""
            )
            avatar_canvas.create_oval(
                size//2 + 4, eye_y,
                size//2 + 4 + eye_size, eye_y + eye_size,
                fill="#004400", outline=""
            )
            # Bot mouth
            avatar_canvas.create_arc(
                size//2 - 8, size//2,
                size//2 + 8, size//2 + 8,
                start=0, extent=-180,
                style=tk.ARC,
                outline="#004400",
                width=2
            )
        
        return avatar_frame
    
    def create_message_bubble(self, sender, message, timestamp):
        """Create a modern message bubble with avatar"""
        # Main message row container
        row_container = tk.Frame(self.scrollable_frame, bg=self.bg_color)
        row_container.pack(fill=tk.X, padx=15, pady=8)
        
        # Inner container for avatar and message
        inner_container = tk.Frame(row_container, bg=self.bg_color)
        
        # Determine alignment and colors based on sender
        if sender == "user":
            bg_color = self.user_msg_bg
            fg_color = "black"
            inner_container.pack(anchor=tk.E)
            avatar_side = tk.RIGHT
            bubble_side = tk.LEFT
            time_anchor = tk.E
        else:
            bg_color = self.bot_msg_bg
            fg_color = self.text_color
            inner_container.pack(anchor=tk.W)
            avatar_side = tk.LEFT
            bubble_side = tk.RIGHT
            time_anchor = tk.W
        
        # Create avatar
        avatar = self.create_avatar(inner_container, sender)
        avatar.pack(side=avatar_side, padx=(5, 5))
        
        # Message content container
        content_container = tk.Frame(inner_container, bg=self.bg_color)
        content_container.pack(side=bubble_side)
        
        # Create canvas for rounded bubble
        bubble_canvas = tk.Canvas(
            content_container,
            bg=self.bg_color,
            highlightthickness=0
        )
        bubble_canvas.pack()
        
        # Create message label first to get dimensions
        temp_label = tk.Label(
            content_container,
            text=message,
            font=self.chat_font,
            wraplength=280,
            justify=tk.LEFT
        )
        temp_label.update_idletasks()
        
        # Calculate bubble dimensions
        padding = 20
        text_width = temp_label.winfo_reqwidth() + padding * 2
        text_height = temp_label.winfo_reqheight() + padding * 1.5
        
        # Configure canvas size
        bubble_canvas.config(width=text_width, height=text_height)
        
        # Draw rounded rectangle
        self.create_rounded_rectangle(
            bubble_canvas,
            5, 5,
            text_width - 5, text_height - 5,
            radius=15,
            fill=bg_color,
            outline=""
        )
        
        # Add shadow effect for user messages
        if sender == "user":
            self.create_rounded_rectangle(
                bubble_canvas,
                7, 7,
                text_width - 3, text_height - 3,
                radius=15,
                fill="#a5c284",
                outline=""
            )
            self.create_rounded_rectangle(
                bubble_canvas,
                5, 5,
                text_width - 5, text_height - 5,
                radius=15,
                fill=bg_color,
                outline=""
            )
        
        # Place text on canvas
        bubble_canvas.create_text(
            text_width // 2,
            text_height // 2,
            text=message,
            font=self.chat_font,
            fill=fg_color,
            width=280,
            anchor=tk.CENTER
        )
        
        # Destroy temporary label
        temp_label.destroy()
        
        # Timestamp below the message
        time_container = tk.Frame(row_container, bg=self.bg_color)
        time_container.pack(fill=tk.X)
        
        time_label = tk.Label(
            time_container,
            text=timestamp,
            font=("Helvetica", 8),
            bg=self.bg_color,
            fg="#808080"
        )
        
        # Adjust timestamp position based on sender
        if sender == "user":
            time_label.pack(anchor=time_anchor, padx=(0, 45))
        else:
            time_label.pack(anchor=time_anchor, padx=(45, 0))
        
        # Animate entry
        self.animate_message_entry(row_container)

    def animate_message_entry(self, widget):
        """Simple fade-in animation for messages"""
        widget.update_idletasks()
        # Scroll to bottom
        self.canvas.yview_moveto(1.0)

    def add_message(self, sender, message):
        """Add a message to the chat"""
        timestamp = datetime.now().strftime("%H:%M")
        self.create_message_bubble(sender, message, timestamp)

    def send_message(self, event):
        """Send user message and get bot response"""
        # Get message text
        message = self.message_input.get("1.0", "end-1c").strip()

        # Ignore if empty or placeholder
        if message and message != "Type a message...":
            # Add user message
            self.add_message("user", message)

            # Clear input
            self.message_input.delete("1.0", tk.END)

            # Update status to typing
            self.update_status("typing")

            # Disable input while processing
            self.message_input.config(state=tk.DISABLED)
            self.send_button.config(state=tk.DISABLED)

            # Get bot response in separate thread
            threading.Thread(target=self.get_bot_response, args=(message,), daemon=True).start()

        return "break"  # Prevent default Return behavior

    def update_status(self, status):
        """Update the status indicator"""
        if status == "typing":
            self.status_dot.delete("all")
            self.status_dot.create_oval(0, 0, 10, 10, fill="#9c6600", outline="")
            self.status_label.config(text="Typing...", fg="#9c6600")
        else:
            self.status_dot.delete("all")
            self.status_dot.create_oval(0, 0, 10, 10, fill="#005800", outline="")
            self.status_label.config(text="Online", fg="#005800")

    def get_bot_response(self, user_message):
        """Generate bot response with typing simulation"""
        # Simulate typing delay
        time.sleep(1.5)

        # Generate response
        response = self.generate_response(user_message)

        # Update UI in main thread
        self.root.after(0, self.add_message, "bot", response)
        self.root.after(0, self.update_status, "online")
        self.root.after(0, self.enable_input)

    def enable_input(self):
        """Re-enable input after bot response"""
        self.message_input.config(state=tk.NORMAL)
        self.send_button.config(state=tk.NORMAL)
        self.message_input.focus_set()

    def generate_response(self, user_input):
        """Enhanced response generation with more patterns"""
        user_input_lower = user_input.lower()

        # Greeting patterns
        if re.search(r'\b(hi|hello|hey|greetings)\b', user_input_lower):
            return "Hey there! I'm here to help. What's on your mind?"

        # How are you patterns
        elif re.search(r'how are you|how\'s it going|what\'s up', user_input_lower):
            return "I'm functioning perfectly! Thanks for asking. How can I assist you today?"

        # Help patterns
        elif re.search(r'\b(help|assist|support)\b', user_input_lower):
            return "I'm here to help! I can chat about various topics, answer questions, or just have a friendly conversation. What would you like to talk about?"

        # Goodbye patterns
        elif re.search(r'\b(bye|goodbye|see you|farewell)\b', user_input_lower):
            return "Take care! Feel free to come back anytime you need assistance. Have a wonderful day!"

        # Thank you patterns
        elif re.search(r'\b(thank|thanks|appreciate)\b', user_input_lower):
            return "You're very welcome! Is there anything else I can help you with?"

        # Default response
        else:
            return get_answer(user_input_lower, model_version="original") # either "original" or "peft"

    def on_window_resize(self, event):
        """Handle window resize events"""
        # Update wraplength for message bubbles based on window width
        new_wraplength = min(int(self.root.winfo_width() * 0.7), 400)
        for widget in self.scrollable_frame.winfo_children():
            for child in widget.winfo_children():
                if isinstance(child, tk.Frame):
                    for label in child.winfo_children():
                        if isinstance(label, tk.Label) and label.cget("wraplength") > 0:
                            label.config(wraplength=new_wraplength)


# Create and run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = ModernChatbotGUI(root)
    root.mainloop()