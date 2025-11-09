# ==========================================================
# math_engine.py — Advanced Mathematical & Computational Intelligence (Fixed)
# ==========================================================

import re
from typing import Dict, Any
import numpy as np

# Lazy imports for heavy dependencies
sympy = None
scipy_stats = None
scipy_optimize = None


def _init_math_deps():
    """Initialize math dependencies lazily"""
    global sympy, scipy_stats, scipy_optimize
    try:
        import sympy as sp
        sympy = sp
    except ImportError:
        print("⚠️ sympy not installed. Run: pip install sympy")

    try:
        from scipy import stats, optimize
        scipy_stats = stats
        scipy_optimize = optimize
    except ImportError:
        print("⚠️ scipy not installed. Run: pip install scipy")


class MathEngine:
    """
    Advanced mathematical problem solver with symbolic computation,
    statistics, calculus, and financial calculations.
    """

    def __init__(self):
        _init_math_deps()
        self.symbols_cache = {}

    # ==========================================================
    # 🧩 Expression Normalization
    # ==========================================================
    def _normalize_expression(self, expr_str: str) -> str:
        """
        Convert human-style math (x^2 + 3x + 5, sin^2(x)) to Sympy syntax.
        """
        expr_str = expr_str.strip()

        # Replace ^ with **
        expr_str = expr_str.replace("^", "**")

        # Fix sin^2(x), cos^2(x), etc. → (sin(x))**2
        expr_str = re.sub(r"(\b(?:sin|cos|tan|log|exp|sqrt))\*\*(\d+)\((.*?)\)",
                          r"(\1(\3))**\2", expr_str)

        # Add * between number and variable (3x → 3*x)
        expr_str = re.sub(r"(\d)([a-zA-Z])", r"\1*\2", expr_str)

        # Add * between variable and variable (xy → x*y)
        # Add * between variables only if not part of known functions like sin, cos, tan, log, exp
        # Add * between alphabetic variables, but don't break known math function names like sin, cos, tan, log, etc.
        # Add * between variables, but avoid breaking known function names
        def safe_insert_mul(match):
            a, b = match.group(1), match.group(2)
            # Skip known math function prefixes
            known_funcs = ["sin", "cos", "tan", "log", "exp", "sqrt"]
            # If 'a' is part of a known function (like 's' in 'sin'), skip
            for func in known_funcs:
                if (a + b).startswith(func[:2]) or func.startswith(a + b):
                    return a + b
            return a + "*" + b

        expr_str = re.sub(r"([a-zA-Z])([a-zA-Z])", safe_insert_mul, expr_str)

        # Add * between ) and variable (like (x+1)(x-1) → (x+1)*(x-1))
        expr_str = re.sub(r"\)\s*(?=[a-zA-Z\(])", r")*", expr_str)
        # Replace pi and e with symbolic constants
        expr_str = re.sub(r'\bpi\b', 'pi', expr_str, flags=re.IGNORECASE)
        expr_str = re.sub(r'\be\b', 'E', expr_str)

        return expr_str.strip()

    # ==========================================================
    # 🔹 Main Query Router
    # ==========================================================
    def solve_query(self, query: str) -> Dict[str, Any]:
        query_lower = query.lower()

        if any(kw in query_lower for kw in ['derivative', 'differentiate', 'diff']):
            return self._solve_derivative(query)
        elif any(kw in query_lower for kw in ['integral', 'integrate']):
            return self._solve_integral(query)
        elif any(kw in query_lower for kw in ['solve', 'equation', 'solve for']):
            return self._solve_equation(query)
        elif any(kw in query_lower for kw in ['mean', 'median', 'std', 'variance', 'statistics']):
            return self._calculate_statistics(query)
        elif any(kw in query_lower for kw in ['matrix', 'eigenvalue', 'determinant']):
            return self._matrix_operations(query)
        elif any(kw in query_lower for kw in ['compound interest', 'roi', 'npv', 'irr']):
            return self._financial_calculation(query)
        elif any(kw in query_lower for kw in ['convert', 'unit']):
            return self._unit_conversion(query)
        else:
            return self._symbolic_compute(query)

    # ==========================================================
    # 🧮 Derivatives
    # ==========================================================
    def _solve_derivative(self, query: str) -> Dict[str, Any]:
        if sympy is None:
            return {"error": "sympy required"}

        try:
            expr_match = re.search(r'derivative of (.+?)(?:with respect to|$)', query, re.I)
            if not expr_match:
                expr_match = re.search(r'differentiate (.+?)(?:with respect to|$)', query, re.I)
            if not expr_match:
                return {"error": "Could not parse expression"}

            expr_str = self._normalize_expression(expr_match.group(1))
            var_match = re.search(r'with respect to (\w+)', query, re.I)
            var = sympy.Symbol(var_match.group(1) if var_match else 'x')

            expr = sympy.sympify(expr_str)
            derivative = sympy.diff(expr, var)

            return {
                "success": True,
                "original": str(expr),
                "derivative": str(derivative),
                "simplified": str(sympy.simplify(derivative)),
                "latex": sympy.latex(derivative),
                "variable": str(var)
            }
        except Exception as e:
            return {"error": f"Differentiation failed: {str(e)}"}

    # ==========================================================
    # ∫ Integrals
    # ==========================================================
    def _solve_integral(self, query: str) -> Dict[str, Any]:
        if sympy is None:
            return {"error": "sympy required"}

        try:
            expr_match = re.search(r'integral of (.+?)(?:with respect to|from|$)', query, re.I)
            if not expr_match:
                return {"error": "Could not parse expression"}

            expr_str = self._normalize_expression(expr_match.group(1))

            # ✅ Directly map constants to Sympy objects instead of string substitutions
            constants = {"pi": sympy.pi, "PI": sympy.pi, "e": sympy.E, "E": sympy.E}

            var_match = re.search(r'with respect to (\w+)', query, re.I)
            var = sympy.Symbol(var_match.group(1) if var_match else 'x')
            limits_match = re.search(r'from (.+?) to (.+)', query, re.I)

            # Convert string to expression with proper constants
            expr = sympy.sympify(expr_str, locals=constants)

            if limits_match:
                lower_raw, upper_raw = limits_match.group(1).strip(), limits_match.group(2).strip()
                lower = sympy.sympify(lower_raw, locals=constants)
                upper = sympy.sympify(upper_raw, locals=constants)
                integral = sympy.integrate(expr, (var, lower, upper))
                result_type = "definite"
            else:
                integral = sympy.integrate(expr, var)
                result_type = "indefinite"

            return {
                "success": True,
                "original": str(expr),
                "integral": str(integral),
                "type": result_type,
                "variable": str(var),
                "latex": sympy.latex(integral)
            }
        except Exception as e:
            return {"error": f"Integration failed: {str(e)}"}

    # ==========================================================
    # ⚙️ Equation Solving
    # ==========================================================
    def _solve_equation(self, query: str) -> Dict[str, Any]:
        if sympy is None:
            return {"error": "sympy required"}

        try:
            eq_match = re.search(r'solve (.+?)(?:for|$)', query, re.I)
            if not eq_match:
                return {"error": "Could not parse equation"}

            eq_str = self._normalize_expression(eq_match.group(1))
            var_match = re.search(r'for (\w+)', query, re.I)
            var = sympy.Symbol(var_match.group(1) if var_match else 'x')

            if '=' in eq_str:
                lhs, rhs = eq_str.split('=')
                eq = sympy.sympify(lhs) - sympy.sympify(rhs)
            else:
                eq = sympy.sympify(eq_str)

            solutions = sympy.solve(eq, var)

            return {
                "success": True,
                "equation": str(eq) + " = 0",
                "variable": str(var),
                "solutions": [str(sol) for sol in solutions],
                "count": len(solutions)
            }
        except Exception as e:
            return {"error": f"Equation solving failed: {str(e)}"}

    # ==========================================================
    # 📊 Statistics
    # ==========================================================
    def _calculate_statistics(self, query: str) -> Dict[str, Any]:
        try:
            numbers = [float(x) for x in re.findall(r'-?\d+\.?\d*', query)]
            if not numbers:
                return {"error": "No numbers found"}

            data = np.array(numbers)
            return {
                "success": True,
                "data": numbers,
                "count": len(numbers),
                "mean": float(np.mean(data)),
                "median": float(np.median(data)),
                "std_dev": float(np.std(data)),
                "variance": float(np.var(data)),
                "min": float(np.min(data)),
                "max": float(np.max(data)),
                "quartiles": {
                    "q1": float(np.percentile(data, 25)),
                    "q2": float(np.percentile(data, 50)),
                    "q3": float(np.percentile(data, 75))
                }
            }
        except Exception as e:
            return {"error": f"Statistics failed: {str(e)}"}

    # ==========================================================
    # 🧮 Matrix Operations
    # ==========================================================
    def _matrix_operations(self, query: str) -> Dict[str, Any]:
        if sympy is None:
            return {"error": "sympy required"}

        try:
            matrix_match = re.search(r'\[\[.+?\]\]', query)
            if not matrix_match:
                return {"error": "Matrix not found. Format: [[1,2],[3,4]]"}

            matrix_data = eval(matrix_match.group(0))
            M = sympy.Matrix(matrix_data)

            result = {
                "success": True,
                "matrix": str(M),
                "shape": M.shape,
                "determinant": str(M.det()) if M.is_square else "N/A"
            }

            if M.is_square:
                result["trace"] = str(M.trace())
                try:
                    result["eigenvalues"] = {str(k): v for k, v in M.eigenvals().items()}
                except Exception:
                    result["eigenvalues"] = "Could not compute"
            return result
        except Exception as e:
            return {"error": f"Matrix failed: {str(e)}"}

    # ==========================================================
    # 💰 Financial + Unit Conversions
    # ==========================================================
    def _financial_calculation(self, query: str) -> Dict[str, Any]:
        try:
            query_lower = query.lower()
            if 'compound interest' in query_lower:
                nums = [float(x) for x in re.findall(r'\d+\.?\d*', query)]
                if len(nums) >= 3:
                    p, r, t = nums[:3]
                    r = r / 100 if r > 1 else r
                    a = p * (1 + r) ** t
                    return {
                        "success": True,
                        "type": "compound_interest",
                        "principal": p,
                        "rate": r * 100,
                        "time": t,
                        "final_amount": round(a, 2),
                        "interest": round(a - p, 2)
                    }
            return {"error": "Could not parse financial query"}
        except Exception as e:
            return {"error": f"Financial failed: {str(e)}"}

    def _unit_conversion(self, query: str) -> Dict[str, Any]:
        conversions = {('km', 'miles'): 0.621371, ('miles', 'km'): 1.60934}
        try:
            match = re.search(r'(\d+\.?\d*)\s*(\w+)\s*(?:to|in)\s*(\w+)', query)
            if not match:
                return {"error": "Invalid conversion query"}
            value, f, t = float(match.group(1)), match.group(2).lower(), match.group(3).lower()
            if (f, t) in conversions:
                res = value * conversions[(f, t)]
                return {"success": True, "from": f"{value} {f}", "to": f"{res:.2f} {t}"}
            return {"error": f"No conversion from {f} to {t}"}
        except Exception as e:
            return {"error": f"Conversion failed: {str(e)}"}

    # ==========================================================
    # 🧠 General Symbolic Compute
    # ==========================================================
    def _symbolic_compute(self, query: str) -> Dict[str, Any]:
        if sympy is None:
            return {"error": "sympy required"}
        try:
            # Remove leading command words like "simplify", "evaluate", etc.
            expr_str = query.lower()
            for kw in ["simplify", "evaluate", "calculate", "compute", "find"]:
                expr_str = re.sub(rf"\b{kw}\b", "", expr_str, flags=re.IGNORECASE)

            expr_str = self._normalize_expression(expr_str)

            # Map constants
            constants = {"pi": sympy.pi, "PI": sympy.pi, "e": sympy.E, "E": sympy.E}

            expr = sympy.sympify(expr_str.strip(), locals=constants)
            evaluated = expr.evalf()

            return {
                "success": True,
                "expression": str(expr),
                "evaluated": str(evaluated),
                "simplified": str(sympy.simplify(expr)),
                "latex": sympy.latex(expr)
            }
        except Exception as e:
            return {"error": f"Could not evaluate expression: {str(e)}"}


# ==========================================================
# 🧪 Usage Example
# ==========================================================
if __name__ == "__main__":
    engine = MathEngine()
    tests = [
        "derivative of x^2 + 3x + 5 with respect to x",
        "derivative of sin^2(x) with respect to x",
        "integral of sin(x) + x^2 from 0 to pi",
        "solve x^2 - 5x + 6 = 0 for x",
        "convert 100 km to miles",
        "statistics 10, 20, 30, 40, 50"
    ]
    for test in tests:
        print(f"\n{'='*60}\nQuery: {test}")
        print(engine.solve_query(test))
