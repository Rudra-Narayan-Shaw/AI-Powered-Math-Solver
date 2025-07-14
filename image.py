import gradio as gr
import sympy as sp
from pix2text import Pix2Text
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import re
import io
import logging

# Configure logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define symbolic variables
x, y = sp.symbols('x y')

# Initialize Pix2Text model globally
try:
    p2t_model = Pix2Text.from_config()
    logger.info("Pix2Text model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load Pix2Text model: {e}")
    p2t_model = None

def clean_latex_expression(latex_str):
    """Clean and normalize LaTeX expression for SymPy parsing"""
    if not latex_str:
        return ""
    
    latex_str = latex_str.strip()
    latex_str = re.sub(r'^\$\$|\$\$$', '', latex_str)  # Remove $$ delimiters
    latex_str = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', latex_str)  # Remove LaTeX commands
    latex_str = re.sub(r'\\{2,}', r'\\', latex_str)  # Fix multiple backslashes
    latex_str = re.sub(r'\s+', ' ', latex_str)  # Normalize whitespace
    latex_str = re.sub(r'\^{([^}]+)}', r'**\1', latex_str)  # Convert x^{n} to x**n
    latex_str = re.sub(r'(\d*\.?\d+)\s*([xy])', r'\1*\2', latex_str)  # Add multiplication: 1.0x -> 1.0*x
    latex_str = re.sub(r'\s*([+\-*/=])\s*', r'\1', latex_str)  # Remove spaces around operators
    if '=' in latex_str:
        left, right = latex_str.split('=')
        latex_str = f"{left} - ({right})"  # Move right-hand side to left
    return latex_str.strip()

def parse_equation_type(latex_str):
    """Determine if the equation is polynomial (single-variable) or linear system (two-variable)"""
    try:
        cleaned = clean_latex_expression(latex_str)
        if not cleaned:
            return 'polynomial'

        # Check for two-variable system
        if 'y' in cleaned and 'x' in cleaned:
            if '\\\\' in latex_str or '\n' in latex_str or len(re.split(r'\\\\|\n|;', latex_str)) >= 2:
                return 'linear_system'
            return 'linear'  # Single equation with x and y

        # Check for single-variable polynomial
        try:
            expr = sp.sympify(cleaned.split('-')[0] if '-' in cleaned else cleaned)
            if x in expr.free_symbols and y not in expr.free_symbols:
                degree = sp.degree(expr, x)
                return 'polynomial' if degree > 0 else 'linear'
            elif x not in expr.free_symbols and y in expr.free_symbols:
                return 'polynomial'  # Treat as polynomial in y if x is absent
            else:
                return 'polynomial'  # Default to polynomial if no clear variables
        except:
            if 'x**' in cleaned or '^' in latex_str:
                return 'polynomial'
            return 'polynomial'  # Fallback to polynomial
    except Exception as e:
        logger.error(f"Error determining equation type: {e}")
        return 'polynomial'

def extract_polynomial_coefficients(latex_str):
    """Extract polynomial coefficients from LaTeX string"""
    try:
        cleaned = clean_latex_expression(latex_str)
        if '-' in cleaned:
            cleaned = cleaned.split('-')[0].strip()  # Use left side for polynomial

        expr = sp.sympify(cleaned, evaluate=False)
        if x not in expr.free_symbols and y not in expr.free_symbols:
            raise ValueError("No variable (x or y) found in expression")
        
        variable = x if x in expr.free_symbols else y
        degree = sp.degree(expr, variable)
        if degree < 1 or degree > 8:
            raise ValueError(f"Polynomial degree {degree} is out of supported range (1-8)")

        poly = sp.Poly(expr, variable)
        coeffs = [float(poly.coeff_monomial(variable**i)) for i in range(degree, -1, -1)]
        
        return {
            "type": "polynomial",
            "degree": degree,
            "coeffs": " ".join(map(str, coeffs)),
            "latex": latex_str,
            "success": True,
            "variable": str(variable)
        }
    except Exception as e:
        logger.error(f"Error extracting polynomial coefficients: {e}")
        return {
            "type": "polynomial",
            "degree": 2,
            "coeffs": "1 0 0",
            "latex": latex_str,
            "success": False,
            "error": str(e),
            "variable": "x"
        }

def extract_linear_system_coefficients(latex_str):
    """Extract linear system coefficients from LaTeX string"""
    try:
        cleaned = clean_latex_expression(latex_str)
        equations = re.split(r'\\\\|\n|;', latex_str)
        if len(equations) < 2:
            equations = re.split(r'(?<=[0-9])\s*(?=[+-]?\s*[0-9]*[xy])', cleaned)
        
        if len(equations) < 2 or 'y' not in cleaned or 'x' not in cleaned:
            raise ValueError("Could not find two equations or two variables (x, y) in system")

        eq1_str = equations[0].strip()
        eq2_str = equations[1].strip()
        
        def parse_linear_eq(eq_str):
            if '-' not in eq_str:
                raise ValueError("No equals sign (converted to '-') found")
            left, right = eq_str.split('-')
            expr = sp.sympify(left) - sp.sympify(right or '0')
            a = float(expr.coeff(x, 1)) if expr.coeff(x, 1) else 0
            b = float(expr.coeff(y, 1)) if expr.coeff(y, 1) else 0
            c = float(-expr.as_coefficients_dict()[1]) if 1 in expr.as_coefficients_dict() else 0
            return f"{a} {b} {c}"
        
        eq1_coeffs = parse_linear_eq(eq1_str)
        eq2_coeffs = parse_linear_eq(eq2_str)
        
        return {
            "type": "linear",
            "eq1_coeffs": eq1_coeffs,
            "eq2_coeffs": eq2_coeffs,
            "latex": latex_str,
            "success": True
        }
    except Exception as e:
        logger.error(f"Error extracting linear system coefficients: {e}")
        return {
            "type": "linear",
            "eq1_coeffs": "1 1 3",
            "eq2_coeffs": "1 -1 1",
            "latex": latex_str,
            "success": False,
            "error": str(e)
        }

def extract_equation_from_image(image_file):
    """Extract equation from image using Pix2Text"""
    try:
        if p2t_model is None:
            return {
                "type": "error",
                "latex": "Pix2Text model not loaded. Please check installation.",
                "success": False
            }
        
        if image_file is None:
            return {
                "type": "error",
                "latex": "No image file provided.",
                "success": False
            }
        
        if isinstance(image_file, str):
            image = Image.open(image_file)
        else:
            image = Image.open(image_file.name)
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        logger.info(f"Processing image of size: {image.size}")
        
        result = p2t_model.recognize_text_formula(image)
        if not result or result.strip() == "":
            return {
                "type": "error",
                "latex": "No text or formulas detected in the image.",
                "success": False
            }
        
        logger.info(f"Extracted text: {result}")
        
        eq_type = parse_equation_type(result)
        if eq_type == 'polynomial':
            return extract_polynomial_coefficients(result)
        elif eq_type == 'linear_system':
            return extract_linear_system_coefficients(result)
        else:
            return {
                "type": "error",
                "latex": f"Unsupported equation type detected: {eq_type}",
                "success": False
            }
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return {
            "type": "error",
            "latex": f"Error processing image: {str(e)}",
            "success": False
        }

def solve_polynomial(degree, coeff_string, real_only):
    """Solve polynomial equation"""
    try:
        coeffs = list(map(float, coeff_string.strip().split()))
        if len(coeffs) != degree + 1:
            return f"⚠️ Please enter exactly {degree + 1} coefficients.", None, None

        poly = sum([coeffs[i] * x**(degree - i) for i in range(degree + 1)])
        simplified = sp.simplify(poly)
        factored = sp.factor(simplified)
        roots = sp.solve(sp.Eq(simplified, 0), x)

        if real_only:
            roots = [r for r in roots if sp.im(r) == 0]

        roots_output = "$$\n" + "\\ ".join(
            [f"r_{{{i}}} = {sp.latex(sp.nsimplify(r, rational=True))}" for i, r in enumerate(roots, 1)]
        ) + "\n$$"

        steps_output = f"""
### Polynomial Expression
$$ {sp.latex(poly)} = 0 $$
### Simplified
$$ {sp.latex(simplified)} = 0 $$
### Factored
$$ {sp.latex(factored)} = 0 $$
### Roots {'(Only Real)' if real_only else '(All Roots)'}
{roots_output}
        """

        x_vals = np.linspace(-10, 10, 400)
        y_vals = np.polyval(coeffs, x_vals)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(x_vals, y_vals, label="Polynomial", color="blue")
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axvline(0, color='black', linewidth=0.5)
        ax.grid(True)
        ax.set_title("Graph of the Polynomial")
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        ax.legend()

        return steps_output, fig, ""
    except Exception as e:
        return f"❌ Error: {e}", None, ""

def solve_linear_system_from_coeffs(eq1_str, eq2_str):
    """Solve linear system"""
    try:
        coeffs1 = list(map(float, eq1_str.strip().split()))
        coeffs2 = list(map(float, eq2_str.strip().split()))

        if len(coeffs1) != 3 or len(coeffs2) != 3:
            return "⚠️ Please enter exactly 3 coefficients for each equation.", None, None, None

        a1, b1, c1 = coeffs1
        a2, b2, c2 = coeffs2

        eq1 = sp.Eq(a1 * x + b1 * y, c1)
        eq2 = sp.Eq(a2 * x + b2 * y, c2)

        sol = sp.solve([eq1, eq2], (x, y), dict=True)
        if not sol:
            return "❌ No unique solution.", None, None, None

        solution = sol[0]
        eq_latex = f"$$ {sp.latex(eq1)} \\ {sp.latex(eq2)} $$"

        steps = rf"""
### Step-by-step Solution
1. **Original Equations:**
   $$ {sp.latex(eq1)} $$
   $$ {sp.latex(eq2)} $$
2. **Standard Form:** Already provided.
3. **Solve using SymPy `solve`:** Internally applies substitution/elimination.
4. **Solve for `x` and `y`:**
   $$ x = {sp.latex(solution[x])}, \quad y = {sp.latex(solution[y])} $$
5. **Verification:** Substitute back into both equations."""

        x_vals = np.linspace(-10, 10, 400)
        f1 = sp.solve(eq1, y)
        f2 = sp.solve(eq2, y)

        fig, ax = plt.subplots()
        if f1:
            f1_func = sp.lambdify(x, f1[0], modules='numpy')
            ax.plot(x_vals, f1_func(x_vals), label=sp.latex(eq1))
        if f2:
            f2_func = sp.lambdify(x, f2[0], modules='numpy')
            ax.plot(x_vals, f2_func(x_vals), label=sp.latex(eq2))

        ax.plot(solution[x], solution[y], 'ro', label=f"Solution ({solution[x]}, {solution[y]})")
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axvline(0, color='black', linewidth=0.5)
        ax.legend()
        ax.set_title("Graph of the Linear System")
        ax.grid(True)

        return eq_latex, steps, fig, ""
    except Exception as e:
        return f"❌ Error: {e}", None, None, None

def solve_extracted_equation(eq_data, real_only):
    """Route to appropriate solver based on equation type"""
    if eq_data["type"] == "polynomial":
        return solve_polynomial(eq_data["degree"], eq_data["coeffs"], real_only)
    elif eq_data["type"] == "linear":
        return "❌ Single linear equation not supported. Please upload a system of equations.", None, ""
    elif eq_data["type"] == "linear_system":
        return solve_linear_system_from_coeffs(eq_data["eq1_coeffs"], eq_data["eq2_coeffs"])
    else:
        return "❌ Unknown equation type", None, ""

def image_tab():
    """Create the Image Upload Solver tab"""
    with gr.Tab("Image Upload Solver"):
        gr.Markdown("## Solve Equations from Image")
        
        with gr.Row():
            image_input = gr.File(
                label="Upload Question Image",
                file_types=[".pdf", ".png", ".jpg", ".jpeg"],
                file_count="single"
            )
            image_upload_btn = gr.Button("Process Image")
        
        gr.Markdown("**Supported Formats:** .pdf, .png, .jpg, .jpeg")

        with gr.Row():
            real_image_checkbox = gr.Checkbox(label="Show Only Real Roots (for Polynomials)", value=False)
            preview_image_btn = gr.Button("Preview Equation")

        image_equation_display = gr.Markdown()
        
        with gr.Row():
            confirm_image_btn = gr.Button("Display Solution", visible=False)
            edit_image_btn = gr.Button("Make Changes Manually", visible=False)

        edit_latex_input = gr.Textbox(label="Edit LaTeX Equation", visible=False, lines=3)
        save_edit_btn = gr.Button("Save Changes", visible=False)

        image_steps_md = gr.Markdown()
        image_plot_output = gr.Plot()
        extracted_eq_state = gr.State()

        def handle_image_upload(image_file):
            """Handle image upload and initial processing"""
            if image_file is None:
                return "", None, "", None, None
            
            try:
                eq_data = extract_equation_from_image(image_file)
                if eq_data["success"]:
                    return "", eq_data, "", None, None
                else:
                    return "", eq_data, "", None, None
            except Exception as e:
                return "", None, "", None, None

        image_upload_btn.click(
            fn=handle_image_upload,
            inputs=[image_input],
            outputs=[image_equation_display, extracted_eq_state, image_steps_md, 
                    image_plot_output, edit_latex_input]
        )

        def preview_image_equation(eq_data, real_only):
            """Preview the extracted equation"""
            if eq_data is None:
                return ("⚠️ No equation data available. Please upload and process an image first.", 
                       gr.update(visible=False), gr.update(visible=False), "", None)
            
            if eq_data["type"] == "error":
                return (eq_data["latex"], gr.update(visible=False), gr.update(visible=False), "", None)
            
            if eq_data["type"] == "polynomial":
                eq_type_display = "Polynomial Equation"
            elif eq_data["type"] == "linear_system":
                eq_type_display = "Linear System"
            else:
                eq_type_display = "Unknown Equation Type"

            preview_text = f"""
### ✅ Confirm {eq_type_display}
**Extracted LaTeX:** {eq_data['latex']}
            """
            
            return (preview_text, gr.update(visible=True), gr.update(visible=True), "", None)

        preview_image_btn.click(
            fn=preview_image_equation,
            inputs=[extracted_eq_state, real_image_checkbox],
            outputs=[image_equation_display, confirm_image_btn, edit_image_btn, 
                    image_steps_md, image_plot_output]
        )

        def confirm_image_solution(eq_data, real_only):
            """Confirm and solve the extracted equation"""
            if eq_data is None or eq_data["type"] == "error":
                return "⚠️ No valid equation to solve.", None, ""
            
            try:
                steps, plot, error = solve_extracted_equation(eq_data, real_only)
                return steps, plot, ""
            except Exception as e:
                return f"❌ Error solving equation: {str(e)}", None, ""

        confirm_image_btn.click(
            fn=confirm_image_solution,
            inputs=[extracted_eq_state, real_image_checkbox],
            outputs=[image_steps_md, image_plot_output, image_equation_display]
        )

        def enable_manual_edit(eq_data):
            """Enable manual editing of the equation"""
            if eq_data is None:
                latex_value = "No equation to edit. Please upload an image first."
            elif eq_data["type"] == "error":
                latex_value = "Error in extraction. Please enter your equation manually."
            else:
                latex_value = eq_data.get("latex", "")
            
            return (gr.update(visible=True, value=latex_value), 
                   gr.update(visible=True), 
                   gr.update(visible=False), 
                   gr.update(visible=False))

        edit_image_btn.click(
            fn=enable_manual_edit,
            inputs=[extracted_eq_state],
            outputs=[edit_latex_input, save_edit_btn, confirm_image_btn, edit_image_btn]
        )

        def save_manual_changes(latex_input, real_only):
            """Save manual changes and solve"""
            try:
                if not latex_input or latex_input.strip() == "":
                    return "⚠️ Please enter a valid equation.", None, ""
                
                eq_type = parse_equation_type(latex_input)
                if eq_type == 'polynomial':
                    eq_data = extract_polynomial_coefficients(latex_input)
                    steps, plot, error = solve_polynomial(eq_data["degree"], eq_data["coeffs"], real_only)
                elif eq_type == 'linear_system':
                    eq_data = extract_linear_system_coefficients(latex_input)
                    eq_latex, steps, plot, error = solve_linear_system_from_coeffs(
                        eq_data["eq1_coeffs"], eq_data["eq2_coeffs"])
                else:
                    return "❌ Unsupported equation type", None, ""
                
                return steps, plot, ""
            except Exception as e:
                return f"❌ Error parsing manual input: {str(e)}", None, ""

        save_edit_btn.click(
            fn=save_manual_changes,
            inputs=[edit_latex_input, real_image_checkbox],
            outputs=[image_steps_md, image_plot_output, image_equation_display]
        )

    return (image_input, image_upload_btn, real_image_checkbox, preview_image_btn,
            image_equation_display, confirm_image_btn, edit_image_btn, edit_latex_input,
            save_edit_btn, image_steps_md, image_plot_output, extracted_eq_state)
