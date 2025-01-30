import tkinter as tk
from tkinter import ttk, messagebox, font
from ttkthemes import ThemedTk
import joblib
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import nltk
from PIL import Image, ImageTk
import time

class ModernSentimentGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Sentiment Analyzer")
        self.root.geometry("1960x900")

        # Set theme
        self.style = ttk.Style()
        self.style.theme_use('clam')

        # Configure colors
        self.configure_styles()

        # Load models
        self.load_models()

        # Initialize lemmatizer
        self.lemmatizer = WordNetLemmatizer()

        # Setup GUI
        self.create_notebook_interface()

        # Add animation variables
        self.analyzing = False
        self.progress = 0

    def configure_styles(self):
        # Define color scheme - Light theme with blue accents
        self.colors = {
            'bg': '#FFFFFF',           # White background
            'fg': '#333333',           # Dark gray text
            'accent': '#1E90FF',       # Dodger blue accent
            'accent_light': '#E6F3FF', # Light blue for highlights
            'error': '#FF4444',        # Red for negative
            'success': '#00A67E',      # Green for positive
            'warning': '#FF9800',      # Orange for neutral
            'border': '#DDDDDD'        # Light gray for borders
        }

        # Configure root window
        self.root.configure(bg=self.colors['bg'])

        # Configure styles
        self.style.configure('Main.TFrame', background=self.colors['bg'])
        self.style.configure('Main.TLabel',
                             background=self.colors['bg'],
                             foreground=self.colors['fg'],
                             font=('Helvetica', 10))
        self.style.configure('Header.TLabel',
                             background=self.colors['bg'],
                             foreground=self.colors['accent'],
                             font=('Helvetica', 16, 'bold'))
        self.style.configure('Result.TLabel',
                             background=self.colors['bg'],
                             foreground=self.colors['success'],
                             font=('Helvetica', 12, 'bold'))

        # Configure Combobox style
        self.style.configure('TCombobox',
                             fieldbackground=self.colors['bg'],
                             background=self.colors['accent'],
                             foreground=self.colors['fg'])

        # Configure Button style
        self.style.configure('Accent.TButton',
                             background=self.colors['accent'],
                             foreground=self.colors['bg'],
                             padding=10)

        # Configure Notebook style
        self.style.configure('TNotebook',
                             background=self.colors['bg'],
                             foreground=self.colors['fg'])
        self.style.configure('TNotebook.Tab',
                             background=self.colors['bg'],
                             foreground=self.colors['fg'],
                             padding=[10, 5])

        # Configure Progressbar style
        self.style.configure("TProgressbar",
                             troughcolor=self.colors['border'],
                             background=self.colors['accent'],
                             borderwidth=0,
                             thickness=10)

    def load_models(self):
        try:
            self.vectorizer = joblib.load('tfidf_vectorizer.pkl')
            self.nb_model = joblib.load('nb_model.pkl')
            self.svm_model = joblib.load('svm_model.pkl')
            self.lr_model = joblib.load('lr_model.pkl')
        except FileNotFoundError as e:
            messagebox.showerror("Error", f"Could not load model files: {str(e)}")
            self.root.destroy()
            return

    def create_notebook_interface(self):
        # Create main notebook with custom styling
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)

        # Create frames for each tab
        self.analyzer_frame = ttk.Frame(self.notebook, style='Main.TFrame')
        self.stats_frame = ttk.Frame(self.notebook, style='Main.TFrame')
        self.help_frame = ttk.Frame(self.notebook, style='Main.TFrame')

        # Add frames to notebook
        self.notebook.add(self.analyzer_frame, text='Analyzer')
        self.notebook.add(self.stats_frame, text='Statistics')
        self.notebook.add(self.help_frame, text='Help')

        # Setup each frame
        self.setup_analyzer_frame()
        self.setup_stats_frame()
        self.setup_help_frame()

    def setup_analyzer_frame(self):
        # Title
        title = ttk.Label(self.analyzer_frame,
                          text="Sentiment Analysis",
                          style='Header.TLabel')
        title.pack(pady=20)

        # Create main content frame with border
        content_frame = ttk.Frame(self.analyzer_frame, style='Main.TFrame')
        content_frame.pack(fill='both', expand=True, padx=20)

        # Model selection with improved styling
        model_frame = ttk.Frame(content_frame, style='Main.TFrame')
        model_frame.pack(fill='x', pady=10)

        ttk.Label(model_frame,
                  text="🤖 Select Model:",
                  style='Main.TLabel').pack(side='left')

        self.model_var = tk.StringVar(value="Naive Bayes")
        model_combo = ttk.Combobox(model_frame,
                                   textvariable=self.model_var,
                                   values=['Naive Bayes', 'SVM', 'Logistic Regression'],
                                   state='readonly')
        model_combo.pack(side='left', padx=(10, 0), fill='x', expand=True)

        # Topic and Source selection with light blue background
        options_frame = ttk.Frame(content_frame, style='Main.TFrame')
        options_frame.pack(fill='x', pady=10)

        # Topic
        topic_frame = ttk.Frame(options_frame, style='Main.TFrame')
        topic_frame.pack(side='left', fill='x', expand=True)

        ttk.Label(topic_frame,
                  text="📑 Topic:",
                  style='Main.TLabel').pack(side='left')

        self.topic_var = tk.StringVar(value="Politics")
        topic_combo = ttk.Combobox(topic_frame,
                                   textvariable=self.topic_var,
                                   values=['Politics', 'Entertainment', 'Sports', 'Business'],
                                   state='readonly')
        topic_combo.pack(side='left', padx=(10, 20), fill='x', expand=True)

        # Source
        source_frame = ttk.Frame(options_frame, style='Main.TFrame')
        source_frame.pack(side='left', fill='x', expand=True)

        ttk.Label(source_frame,
                  text="📱 Source:",
                  style='Main.TLabel').pack(side='left')

        self.source_var = tk.StringVar(value="Twitter")
        source_combo = ttk.Combobox(source_frame,
                                    textvariable=self.source_var,
                                    values=['Twitter', 'Facebook', 'Instagram'],
                                    state='readonly')
        source_combo.pack(side='left', padx=(10, 0), fill='x', expand=True)

        # Text input with light border
        ttk.Label(content_frame,
                  text="✍️ Enter Text to Analyze:",
                  style='Main.TLabel').pack(pady=(20, 5))

        self.text_input = tk.Text(content_frame,
                                  height=5,
                                  bg=self.colors['bg'],
                                  fg=self.colors['fg'],
                                  insertbackground=self.colors['accent'],
                                  relief='solid',
                                  borderwidth=1,
                                  font=('Helvetica', 11))
        self.text_input.pack(fill='x', pady=5)

        # Progress bar with blue accent
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(content_frame,
                                            mode='determinate',
                                            variable=self.progress_var,
                                            style="TProgressbar")
        self.progress_bar.pack(fill='x', pady=10)
        self.progress_bar.pack_forget()

        # Analyze button with blue accent
        self.analyze_button = ttk.Button(content_frame,
                                         text="🔍 Analyze Sentiment",
                                         style='Accent.TButton',
                                         command=self.animate_analysis)
        self.analyze_button.pack(pady=20)

        # Result frame
        self.result_frame = ttk.Frame(content_frame, style='Main.TFrame')
        self.result_frame.pack(fill='x', pady=10)

        self.result_var = tk.StringVar()
        self.result_label = ttk.Label(self.result_frame,
                                      textvariable=self.result_var,
                                      style='Result.TLabel')
        self.result_label.pack()

    def setup_stats_frame(self):
        ttk.Label(self.stats_frame,
                  text="Model Performance Statistics",
                  style='Header.TLabel').pack(pady=20)

        stats_content = ttk.Frame(self.stats_frame, style='Main.TFrame')
        stats_content.pack(fill='both', expand=True, padx=20)

        models = {
            'Naive Bayes': 0.85,
            'SVM': 0.87,
            'Logistic Regression': 0.86
        }

        for model, accuracy in models.items():
            model_frame = ttk.Frame(stats_content, style='Main.TFrame')
            model_frame.pack(fill='x', pady=10)

            ttk.Label(model_frame,
                      text=f"{model}:",
                      style='Main.TLabel').pack(side='left')

            progress = ttk.Progressbar(model_frame,
                                       length=200,
                                       mode='determinate',
                                       style="TProgressbar")
            progress.pack(side='left', padx=10)
            progress['value'] = accuracy * 100

            ttk.Label(model_frame,
                      text=f"{accuracy:.2%}",
                      style='Main.TLabel').pack(side='left')

    def setup_help_frame(self):
        ttk.Label(self.help_frame,
                  text="How to Use",
                  style='Header.TLabel').pack(pady=20)

        help_text = """
        1. Select a Model:
           - Naive Bayes: Fast and efficient
           - SVM: More accurate but slower
           - Logistic Regression: Balanced approach
        
        2. Choose Topic and Source:
           - These help contextualize the analysis
        
        3. Enter Text:
           - Type or paste the text you want to analyze
        
        4. Click Analyze:
           - The system will process and show the sentiment
        
        Tips:
        - Longer texts provide more accurate results
        - Clean text works better than messy text
        - All models are trained on English text
        """

        text_widget = tk.Text(self.help_frame,
                              wrap='word',
                              height=15,
                              bg=self.colors['bg'],
                              fg=self.colors['fg'],
                              font=('Helvetica', 11),
                              relief='solid',
                              borderwidth=1)
        text_widget.pack(fill='both', expand=True, padx=20, pady=10)
        text_widget.insert('1.0', help_text)
        text_widget.configure(state='disabled')

    def animate_analysis(self):
        if not self.analyzing:
            self.analyzing = True
            self.progress_bar.pack(fill='x', pady=10)
            self.progress_var.set(0)
            self.analyze_button.configure(state='disabled')
            self.root.after(50, self.update_progress)

    def update_progress(self):
        if self.progress_var.get() < 100:
            self.progress_var.set(self.progress_var.get() + 2)
            self.root.after(20, self.update_progress)
        else:
            self.progress_bar.pack_forget()
            self.analyzing = False
            self.analyze_button.configure(state='normal')
            self.predict_sentiment()

    def pre_process_text(self, text):
        text = text.lower()
        text = ''.join([char for char in text if char not in string.punctuation])
        stop_words = set(stopwords.words('english'))
        text = ' '.join([word for word in text.split() if word not in stop_words])
        text = ' '.join([self.lemmatizer.lemmatize(word) for word in text.split()])
        return text

    def predict_sentiment(self):
        text = self.text_input.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("Warning", "Please enter some text to analyze.")
            return

        try:
            processed_text = self.pre_process_text(text)
            text_vectorized = self.vectorizer.transform([processed_text]).toarray()

            topic_mapping = {'Politics': 0, 'Entertainment': 1, 'Sports': 2, 'Business': 3}
            source_mapping = {'Twitter': 0, 'Facebook': 1, 'Instagram': 2}

            topic_encoded = topic_mapping[self.topic_var.get()]
            source_encoded = source_mapping[self.source_var.get()]

            features = np.hstack((text_vectorized, [[source_encoded, topic_encoded]]))

            model_mapping = {
                "Naive Bayes": self.nb_model,
                "SVM": self.svm_model,
                "Logistic Regression": self.lr_model
            }

            selected_model = model_mapping[self.model_var.get()]
            prediction = selected_model.predict(features)

            sentiment_mapping = {0: "Negative", 1: "Neutral", 2: "Positive"}
            sentiment = sentiment_mapping[prediction[0]]

            # Update result with color coding
            color_mapping = {
                "Positive": self.colors['success'],
                "Neutral": self.colors['warning'],
                "Negative": self.colors['error']
            }

            self.result_label.configure(foreground=color_mapping[sentiment])
            self.result_var.set(f"Predicted Sentiment: {sentiment}")

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during prediction: {str(e)}")

if __name__ == "__main__":
    root = ThemedTk(theme="clam")
    app = ModernSentimentGUI(root)
    root.mainloop()
