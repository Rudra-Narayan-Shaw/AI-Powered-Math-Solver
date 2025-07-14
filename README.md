# AI-Powered-Math-Solver

**A next-generation, AI-driven math assistant that interprets, solves, and explains handwritten and printed equations—empowering students with stepwise insights and visualizations.**

> *Developed during the IDEAS, ISI Kolkata Internship under the mentorship of Srijit Mukherjee.*

**🌐 [Live Demo on Hugging Face Spaces](https://huggingface.co/spaces/rU-ShawJI-07/EqToSol_V4)**

## 🚀 Overview

**AI-Powered-Math-Solver** bridges the gap between flexible math input (typed, scanned, or handwritten) and deep conceptual understanding. By fusing advanced OCR, symbolic computation, and large language models, it delivers:

- Stepwise, human-like explanations
- Graphical visualizations
- Practice problem generation
- Intuitive, modern UI

## ✨ Key Features

- **Hybrid Input:** Accepts both typed math and image uploads (handwritten or printed).
- **Cutting-Edge OCR:** Leverages Pix2Text and other models for robust recognition.
- **Symbolic Computation:** Solves linear and polynomial equations via SymPy.
- **Stepwise Explanations:** Breaks down each step with natural language and LaTeX.

- **Interactive Visualizations:** Plots equations, roots, and solution steps.
- **Practice Generator:** Creates similar problems for self-study.
- **Web App:** Clean, responsive interface built with Gradio.

## 🗂️ Project Structure

| File/Folder                      | Description                                                                 |
|----------------------------------|-----------------------------------------------------------------------------|
| `README.md`                      | Project overview, setup, usage, and credits (this file).                    |
| `requirements.txt`               | Python dependencies (gradio, sympy, numpy, opencv-python, etc.).            |
| `app.py`                         | Main application script; integrates all modules and runs the web app.       |
| `image.py`                       | Image input processing and OCR integration (Pix2Text, Tesseract, MathPix).  |
| `linear.py`                      | Logic for solving linear equations (matrix ops, symbolic methods).          |
| `polynomial.py`                  | Polynomial equation solving and stepwise formatting.                        |
| `.gitignore`                     | Specifies files/folders to be ignored by Git.                               |
| `LICENSE`                        | Project license (e.g., MIT, Apache 2.0).                                    |
| `AIMathSolver(1).PDF`            | Group documentation (MS Word, submitted to IDEAS, ISI Kolkata).             |
| `AI-Powered_Equation_Solver.pdf` | Self-documentation (LaTeX, Overleaf).                                       |

## 🛠️ Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
git clone https://github.com/YOUR-USERNAME/AI-Powered-Math-Solver.git
cd AI-Powered-Math-Solver
pip install -r requirements.txt
```

### Run the Application

```bash
python app.py
```

The application will launch in your browser with a Gradio interface.

## 💡 How to Use

- **Typed Input:** Enter equations directly in the text box.
- **Image Input:** Upload a photo or scan of handwritten/printed equations.
- **View Solutions:** Explore stepwise explanations, symbolic steps, and interactive graphs.
- **Practice:** Generate new, similar problems for extra learning.

## 🧩 Example Inputs & Outputs

### Example 1: Typed Input

**Input:**  
`2x + 3y = 7, 4x - y = 5`

**Output:**  
- Stepwise symbolic solution for the system
- Natural language explanation for each step
- Graph showing intersection point

### Example 2: Image Input

**Input:**  
Upload a photo of a handwritten quadratic equation.

**Output:**  
- OCR extracts the equation
- Stepwise solution with LaTeX formatting
- Explanation referencing relevant theorems
- Plot of the quadratic with roots highlighted

## 📄 Documentation

- **Group Report:** `AIMathSolver(1).PDF`
- **Self Documentation:** `AI-Powered_Equation_Solver.pdf`
- **Demo:** [Hugging Face Space](https://huggingface.co/spaces/rU-ShawJI-07/EqToSol_V4)

## 👥 Meet the Team

- **Aritrajit Roy**
- **Rudra Narayan Shaw**
- **Rwiddhit Chatterjee**
- **Swapnomon Murari**
- **Vedika Anand Thakur**

**Mentor:** Srijit Mukherjee  
**Institution:** IDEAS – Institute of Data Engineering, Analytics and Science Foundation, ISI Kolkata

## 🌐 Social Media & Connecting Links

| Platform/Version         | Link                                                                                                  |
|--------------------------|-------------------------------------------------------------------------------------------------------|
| **Main Demo**            | [Hugging Face Space (Rudra’s Version)](https://huggingface.co/spaces/rU-ShawJI-07/EqToSol_V4)         |
| **Aritrajit’s Version**  | [Hugging Face Space](https://huggingface.co/spaces/Aroy1997/poly_oracle)                              |
| **Swapnomon’s Version**  | [Hugging Face Space](https://huggingface.co/spaces/MasteredUltraInstinct/FinishedProject)             |
| **Vedika’s Version**     | [GitHub Repository](https://github.com/VedikaThakur/Solvify/tree/main)                                |
| **Rwiddhit’s Version**   | [Hugging Face Space](https://www.linkedin.com/in/srijit-mukherjee/)                                   |
| **Mentor**               | [LinkedIN](https://huggingface.co/spaces/red-cq/polynomialOCR)                                        |
| **Institution:**         | [Google](https://www.ideas-tih.org/)                                                                  |


*For questions, suggestions, or collaborations, please open an issue or submit a pull request. You can also connect with team members via their respective project links above.*

## 📜 License

Licensed under the terms specified in the `LICENSE` file.

> *For questions, suggestions, or contributions, please open an issue or submit a pull request. Happy solving!*

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/72691087/a412c8b8-6da3-45f5-b641-c094463494cb/AIMathSolver-1.PDF
[2] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/72691087/49b4af50-6d25-4f31-a316-d6ee8997e742/AI-Powered_Equation_Solver.pdf
