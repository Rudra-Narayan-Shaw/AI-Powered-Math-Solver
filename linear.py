import gradio as gr
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import random

x, y = sp.symbols('x y')

def generate_linear_template():
    return "$$ a_1x + b_1y = c_1 \\ a_2x + b_2y = c_2 $$"

def load_linear_example():
    examples = [
        ("1 -4 -2", "5 1 9"),
        ("2 1 8", "1 -1 2"),
        ("3 2 12", "1 1 5"),
        ("4 -1 3", "2 3 6"),
        ("1 2 10", "3 -1 5")
    ]
    return random.choice(examples)

def solve_linear_system_from_coeffs(eq1_str, eq2_str):
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

def linear_tab():
    with gr.Tab("Linear System Solver"):
        gr.Markdown("## Solve 2x2 Linear System")
        linear_template = gr.Markdown(value=generate_linear_template())

        with gr.Row():
            linear_eq1_input = gr.Textbox(label="Equation 1 Coefficients (a1 b1 c1)", placeholder="e.g. 2 1 8")
            linear_eq2_input = gr.Textbox(label="Equation 2 Coefficients (a2 b2 c2)", placeholder="e.g. 1 -1 2")

        linear_example_btn = gr.Button("Load Example")
        preview_button = gr.Button("Preview Equations")

        linear_equation_display = gr.Markdown()
        with gr.Row():
            confirm_btn = gr.Button("Display Solution", visible=False)
            cancel_btn = gr.Button("Make Changes in Equation", visible=False)

        linear_steps_md = gr.Markdown()
        linear_plot = gr.Plot()
        linear_error = gr.Textbox(visible=False)

        def update_example():
            eq1, eq2 = load_linear_example()
            return eq1, eq2

        linear_example_btn.click(fn=update_example, inputs=[], outputs=[linear_eq1_input, linear_eq2_input])

        def preview_equations(eq1_str, eq2_str):
            try:
                coeffs1 = list(map(float, eq1_str.strip().split()))
                coeffs2 = list(map(float, eq2_str.strip().split()))
                if len(coeffs1) != 3 or len(coeffs2) != 3:
                    return "⚠️ Please enter exactly 3 coefficients for each equation.", gr.update(visible=False), gr.update(visible=False)
                a1, b1, c1 = coeffs1
                a2, b2, c2 = coeffs2
                eq1 = sp.Eq(a1 * x + b1 * y, c1)
                eq2 = sp.Eq(a2 * x + b2 * y, c2)
                eq_latex = f"### ✅ Confirm Equations\n\n$$ {sp.latex(eq1)} \\\\ {sp.latex(eq2)} $$"
                return eq_latex, gr.update(visible=True), gr.update(visible=True)
            except Exception as e:
                return f"❌ Error parsing equations: {e}", gr.update(visible=False), gr.update(visible=False)

        preview_button.click(
            fn=preview_equations,
            inputs=[linear_eq1_input, linear_eq2_input],
            outputs=[linear_equation_display, confirm_btn, cancel_btn]
        )

        cancel_btn.click(
            fn=lambda: (gr.update(visible=False), gr.update(visible=False), "", None, None),
            outputs=[confirm_btn, cancel_btn, linear_equation_display, linear_steps_md, linear_plot]
        )

        confirm_btn.click(
            fn=solve_linear_system_from_coeffs,
            inputs=[linear_eq1_input, linear_eq2_input],
            outputs=[linear_equation_display, linear_steps_md, linear_plot, linear_error]
        )
