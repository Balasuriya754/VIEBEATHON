# ==========================================================
# test_math_engine_extended.py — Validation for MathEngine
# ==========================================================
import sys
from pathlib import Path
from pprint import pprint
from termcolor import colored

ROOT = Path(__file__).parent
RAG_CORE = ROOT / "OfflineRAG_Pro" / "rag_core"
sys.path.insert(0, str(RAG_CORE))

print("🔍 Verifying extended MathEngine capabilities...\n")

try:
    from math_engine import MathEngine
except Exception as e:
    print("❌ Failed to import MathEngine:", e)
    sys.exit(1)

engine = MathEngine()

tests = [
    # Calculus
    "derivative of x^3 + 5x^2 - 6x + 7 with respect to x",
    "derivative of sin(x)*cos(x) with respect to x",
    "integral of x^2 + 3x from 0 to 2",
    "integral of exp(x) with respect to x",
    "integral of sin(x)*cos(x) with respect to x",

    # Algebra / Equations
    "solve 2*x + 6 = 0 for x",
    "solve x^2 - 9 = 0 for x",
    "solve sin(x) = 0 for x",
    "solve 3*x + 2*y - 5 = 0 for x",

    # Statistics
    "statistics 2, 4, 6, 8, 10, 12",
    "mean and variance of 5, 10, 15, 20, 25",

    # Matrices
    "determinant of [[1,2],[3,4]]",
    "matrix [[2,1],[1,2]] eigenvalues",
    "matrix [[4, 7], [2, 6]] trace and determinant",

    # Finance / Conversion
    "compound interest 1000 5 2",
    "convert 10 km to miles",
    "convert 50 miles to km",

    # General symbolic
    "simplify (x^2 - y^2)/(x-y)",
    "evaluate sin(pi/2)",
    "calculate exp(1)",
]

total = len(tests)
success = 0

for i, q in enumerate(tests, 1):
    print(colored(f"\n-----------------------------------------------", "cyan"))
    print(colored(f"🧩 Test {i}/{total}: {q}", "yellow"))
    try:
        result = engine.solve_query(q)
        if result.get("success"):
            success += 1
            print(colored("✅ Success", "green"))
        else:
            print(colored("⚠️ Partial / Failed:", "red"), result.get("error"))
        pprint(result)
    except Exception as e:
        print(colored("❌ Exception:", "red"), str(e))

print(colored(f"\n🏁 Completed {success}/{total} successful tests ({(success/total)*100:.1f}%).", "green" if success == total else "yellow"))
