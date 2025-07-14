import gradio as gr
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# Define symbolic variable for polynomial operations
x = sp.symbols('x')

# Function to generate a LaTeX template for a polynomial based on its degree
def generate_polynomial_template(degree):
    terms = [f"a_{{{i}}}x^{degree - i}" for i in range(degree)]
    terms.append(f"a_{{{degree}}}")
    return "$$" + " + ".join(terms) + " = 0$$"

# Function to load example coefficients for a given polynomial degree
def load_poly_example(degree):
    examples = {
        1: "3 9",
        2: "1 -3 2",
        3: "1 -6 11 -6",
        4: "1 0 -5 0 4",
        5: "1 -9 3 8 1 8",
        6: "1 -9 3 8 1 8 3",
        7: "1 -9 3 8 1 8 6 2",
        8: "1 -9 3 8 1 8 2 3 7"
    }
    return examples.get(degree, "")

# Function to solve the polynomial equation and generate a graph
def solve_polynomial(degree, coeff_string, real_only):
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

# Function to create the Polynomial Solver tab with Gradio components
def polynomial_tab():
    with gr.Tab("Polynomial Solver"):
        # Row for displaying the equation template and real roots checkbox
        with gr.Row():
            template_display = gr.Markdown(value=generate_polynomial_template(2))
            real_checkbox = gr.Checkbox(label="Show Only Real Roots", value=False)

        # Row for selecting the polynomial degree and entering coefficients
        with gr.Row():
            degree_slider = gr.Slider(1, 8, value=2, step=1, label="Select Degree of Polynomial Equation")
            coeff_input = gr.Textbox(label="Enter Coefficients (space-separated)", placeholder="e.g. 1 -3 2")

        # Row for example and preview buttons
        with gr.Row():
            example_btn = gr.Button("Load Example")
            preview_poly_button = gr.Button("Preview Equation")

        # Row for displaying the confirmed equation
        with gr.Row():
            poly_equation_display = gr.Markdown()

        # Row for confirm and cancel buttons
        with gr.Row():
            confirm_poly_btn = gr.Button("Display Solution", visible=False)
            cancel_poly_btn = gr.Button("Make Changes in Equation", visible=False)

        # Markdown component to display step-by-step solution
        steps_md = gr.Markdown()

        # Plot component to display the polynomial graph
        plot_output = gr.Plot()

        # Textbox to display errors (initially hidden)
        error_box = gr.Textbox(visible=False)

        # Function to preview the polynomial equation based on user input
        def preview_polynomial(degree, coeff_string, real_only):
            try:
                coeffs = list(map(float, coeff_string.strip().split()))
                if len(coeffs) != degree + 1:
                    return f"⚠️ Please enter exactly {degree + 1} coefficients.", gr.update(visible=False), gr.update(visible=False), "", None
                poly = sum([coeffs[i] * x**(degree - i) for i in range(degree + 1)])
                eq_latex = f"### ✅ Confirm Polynomial\n\n$$ {sp.latex(poly)} = 0 $$"
                return eq_latex, gr.update(visible=True), gr.update(visible=True), "", None
            except Exception as e:
                return f"❌ Error parsing coefficients: {e}", gr.update(visible=False), gr.update(visible=False), "", None

        # Event handler for preview button click
        preview_poly_button.click(
            fn=preview_polynomial,
            inputs=[degree_slider, coeff_input, real_checkbox],
            outputs=[poly_equation_display, confirm_poly_btn, cancel_poly_btn, steps_md, plot_output]
        )

        # Function to handle cancellation of the preview
        def cancel_poly():
            return gr.update(visible=False), gr.update(visible=False), "", "", None

        # Event handler for cancel button click
        cancel_poly_btn.click(
            fn=cancel_poly,
            inputs=[],
            outputs=[confirm_poly_btn, cancel_poly_btn, poly_equation_display, steps_md, plot_output]
        )

        # Event handler for confirm button click to solve and display results
        confirm_poly_btn.click(
            fn=solve_polynomial,
            inputs=[degree_slider, coeff_input, real_checkbox],
            outputs=[steps_md, plot_output, error_box]
        )

        # Event handler to update the template when the degree slider changes
        degree_slider.change(fn=generate_polynomial_template, inputs=degree_slider, outputs=template_display)

        # Event handler to load an example when the example button is clicked
        example_btn.click(fn=load_poly_example, inputs=degree_slider, outputs=coeff_input)

        # Initialize the template display with the default degree (2) on load
        template_display.value = generate_polynomial_template(2)

    return template_display, real_checkbox, degree_slider, coeff_input, example_btn, preview_poly_button, poly_equation_display, confirm_poly_btn, cancel_poly_btn, steps_md, plot_output, error_box
