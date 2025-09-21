import streamlit as st
import numpy as np
from scipy import integrate
import pandas as pd
from typing import Tuple, Union, List
import math  # ✅ ADDED FOR factorial
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Moments Calculator",
    page_icon="🎲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS (keep your beautiful styling for calculator)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    .main {
        font-family: 'Inter', sans-serif;
    }
    /* Animated gradient background */
    .main-header {
        background: linear-gradient(-45deg, #667eea, #764ba2, #f093fb, #f5576c);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
        padding: 3rem 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.4);
        position: relative;
        overflow: hidden;
    }
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 20px;
    }
    .main-header * {
        position: relative;
        z-index: 1;
    }
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    .main-header h1 {
        margin: 0;
        font-size: 3rem;
        font-weight: 700;
        text-shadow: 0 2px 10px rgba(0,0,0,0.3);
        background: linear-gradient(45deg, #ffffff, #f0f0f0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .main-header p {
        margin: 1rem 0 0 0;
        font-size: 1.3rem;
        opacity: 0.95;
        font-weight: 400;
        text-shadow: 0 1px 5px rgba(0,0,0,0.2);
    }
    /* Glassmorphism inputs */
    .stSelectbox > div > div > div {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 12px;
        color: #2d3436;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .stSelectbox > div > div > div:hover {
        background: rgba(255, 255, 255, 0.95);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
    }
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stNumberInput > div > div > input {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 12px;
        color: #2d3436;
        font-size: 14px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus,
    .stNumberInput > div > div > input:focus {
        background: rgba(255, 255, 255, 1);
        border: 2px solid #667eea;
        box-shadow: 0 0 20px rgba(102, 126, 234, 0.3);
        transform: scale(1.02);
    }
    .stTextArea > div > div > textarea {
        font-family: 'SF Mono', 'Monaco', 'Inconsolata', 'Roboto Mono', monospace;
    }
    /* Animated metric cards */
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin: 0.25rem 0;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    .metric-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.5s;
    }
    .metric-container:hover {
        transform: translateY(-2px) scale(1.02);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    .metric-container:hover::before {
        left: 100%;
    }
    .metric-label {
        font-size: 0.8rem;
        font-weight: 500;
        margin-bottom: 0.4rem;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 1.2rem;
        font-weight: 700;
        margin: 0;
        font-family: 'SF Mono', monospace;
    }
    /* Enhanced message boxes with icons */
    .error-box, .success-box, .info-box, .warning-box {
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        font-weight: 500;
        position: relative;
        backdrop-filter: blur(10px);
        border: 1px solid;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .error-box {
        background: linear-gradient(135deg, rgba(255, 71, 87, 0.1), rgba(255, 71, 87, 0.05));
        border-color: #ff4757;
        color: #d63031;
    }
    .success-box {
        background: linear-gradient(135deg, rgba(46, 213, 115, 0.1), rgba(46, 213, 115, 0.05));
        border-color: #2ed573;
        color: #00b894;
    }
    .info-box {
        background: linear-gradient(135deg, rgba(116, 185, 255, 0.1), rgba(116, 185, 255, 0.05));
        border-color: #74b9ff;
        color: #0984e3;
    }
    .warning-box {
        background: linear-gradient(135deg, rgba(255, 167, 38, 0.1), rgba(255, 167, 38, 0.05));
        border-color: #ffa726;
        color: #ef6c00;
    }
    /* Section headers with underline animation */
    .section-header {
        color: white !important;
        font-size: 1.4rem;
        font-weight: 700 !important;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        position: relative;
        display: inline-block;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    .section-header::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 0;
        width: 0;
        height: 3px;
        background: linear-gradient(90deg, #667eea, #764ba2);
        transition: width 0.3s ease;
        border-radius: 2px;
    }
    .section-header:hover::after {
        width: 100%;
    }
    /* Enhanced dataframe styling */
    .dataframe {
        font-size: 14px;
        border-radius: 12px;
        overflow: hidden;
        border: none;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        background: white;
    }
    .dataframe th {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        font-weight: 600;
        padding: 15px 12px;
        border: none;
    }
    .dataframe td {
        padding: 12px;
        border-bottom: 1px solid #f1f3f4;
        transition: background-color 0.2s ease;
    }
    .dataframe tr:hover td {
        background-color: #f8f9fa;
    }
    /* Sidebar enhancements */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    /* Custom expander styling */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    .streamlit-expanderHeader:hover {
        background: rgba(102, 126, 234, 0.1);
        transform: translateX(5px);
    }
    /* Loading animation */
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    .loading {
        animation: pulse 2s infinite;
    }
    /* Progress indicators */
    .progress-container {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 4px;
        margin: 1rem 0;
    }
    .progress-bar {
        background: linear-gradient(90deg, #667eea, #764ba2);
        border-radius: 16px;
        height: 8px;
        transition: width 0.3s ease;
    }
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #667eea, #764ba2);
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #5a6fd8, #6a42a0);
    }
    /* Responsive grid for moments */
    .moments-grid {
        display: grid;
        gap: 0.5rem;
        margin: 1rem 0;
    }
    .moments-grid-2 { grid-template-columns: repeat(2, 1fr); }
    .moments-grid-3 { grid-template-columns: repeat(3, 1fr); }
    .moments-grid-4 { grid-template-columns: repeat(4, 1fr); }
    .moments-grid-5 { grid-template-columns: repeat(5, 1fr); }
</style>
""", unsafe_allow_html=True)

# Helper classes
class InfiniteSeriesHandler:
    @staticmethod
    def detect_series_pattern(values: List[float]) -> Tuple[str, dict]:
        if len(values) < 2:
            return "unknown", {}
        diffs = [values[i+1] - values[i] for i in range(len(values)-1)]
        if len(set(diffs)) == 1:
            return "arithmetic", {"start": values[0], "diff": diffs[0]}
        if all(v != 0 for v in values):
            ratios = [values[i+1] / values[i] for i in range(len(values)-1)]
            if all(abs(ratios[i] - ratios[0]) < 1e-15 for i in range(len(ratios))):
                return "geometric", {"start": values[0], "ratio": ratios[0]}
        return "custom", {"values": values}

    @staticmethod
    def generate_extended_series(pattern_type: str, params: dict, max_terms: int = 100) -> List[float]:
        if pattern_type == "arithmetic":
            start, diff = params["start"], params["diff"]
            return [start + i * diff for i in range(max_terms)]
        elif pattern_type == "geometric":
            start, ratio = params["start"], params["ratio"]
            return [start * (ratio ** i) for i in range(max_terms)]
        else:
            base_values = params.get("values", [0])
            if len(base_values) >= 2:
                diff = base_values[-1] - base_values[-2]
                extended = base_values.copy()
                while len(extended) < max_terms:
                    extended.append(extended[-1] + diff)
                return extended
            else:
                return [base_values[0] + i for i in range(max_terms)]

    @staticmethod
    def estimate_infinite_sum(func_str: str, values: List[float], pattern_type: str, params: dict) -> Tuple[float, bool, str]:
        safe_dict = {
            'x': 0, 'factorial': math.factorial, 'sqrt': np.sqrt,  # ✅ FIXED
            'exp': np.exp, 'log': np.log, 'sin': np.sin, 'cos': np.cos,
            'tan': np.tan, 'pi': np.pi, 'e': np.e
        }
        partial_sum = 0
        terms = []
        try:
            for i, x_val in enumerate(values[:50]):
                safe_dict['x'] = x_val
                term = eval(func_str, {"__builtins__": {}}, safe_dict)
                terms.append(term)
                partial_sum += term

            if len(terms) >= 20:
                recent_terms = terms[-10:]
                if all(abs(t) < 1e-15 for t in recent_terms):
                    return partial_sum, True, f"Series converges (terms → 0, sum ≈ {partial_sum:.15f})"

                if len(terms) >= 25 and all(abs(t) > 1e-20 for t in terms[-15:]):
                    ratios = [abs(terms[i+1]/terms[i]) for i in range(len(terms)-10, len(terms)-1)]
                    avg_ratio = sum(ratios) / len(ratios)
                    if avg_ratio < 0.9:
                        remaining_estimate = terms[-1] * avg_ratio / (1 - avg_ratio)
                        estimated_total = partial_sum + remaining_estimate
                        return estimated_total, True, f"Series converges (ratio test, sum ≈ {estimated_total:.15f})"
                    elif avg_ratio > 1.1:
                        return partial_sum, False, "Series appears to diverge (ratio test)"

                if len(terms) >= 15:
                    signs = [1 if t >= 0 else -1 for t in terms[-15:]]
                    if len(set(signs)) == 2:
                        abs_terms = [abs(t) for t in terms[-10:]]
                        if all(abs_terms[i] >= abs_terms[i+1] for i in range(len(abs_terms)-1)):
                            return partial_sum, True, f"Alternating series converges (sum ≈ {partial_sum:.15f})"

            return partial_sum, False, f"Convergence uncertain (partial sum of {len(terms)} terms: {partial_sum:.15f})"

        except Exception as e:
            return 0, False, f"Error in series evaluation: {str(e)}"

class EnhancedProbabilityValidator:
    @staticmethod
    def validate_drv_probabilities(func_str: str, range_values: List[float], is_infinite: bool = False) -> Tuple[bool, str, float, dict]:
        # ✅ Initialize analysis BEFORE try block
        analysis = {"terms_computed": 0, "convergence_info": "", "series_type": "finite"}
        
        try:
            safe_dict = {
                'x': 0, 'factorial': math.factorial, 'sqrt': np.sqrt,  # ✅ FIXED
                'exp': np.exp, 'log': np.log, 'sin': np.sin, 'cos': np.cos,
                'tan': np.tan, 'pi': np.pi, 'e': np.e
            }
            negative_probs = []

            if is_infinite:
                pattern_type, params = InfiniteSeriesHandler.detect_series_pattern(range_values[:10])
                total_prob, converges, convergence_msg = InfiniteSeriesHandler.estimate_infinite_sum(
                    func_str, range_values, pattern_type, params
                )
                
                for x_val in range_values[:50]:
                    safe_dict['x'] = x_val
                    prob = eval(func_str, {"__builtins__": {}}, safe_dict)
                    if prob < 0:
                        negative_probs.append(x_val)
                
                analysis.update({
                    "terms_computed": min(50, len(range_values)),
                    "convergence_info": convergence_msg,
                    "series_type": "infinite",
                    "pattern_type": pattern_type,
                    "converges": converges
                })
                tolerance = 0.05 if not converges else 0.01
            else:
                total_prob = 0
                for x_val in range_values:
                    safe_dict['x'] = x_val
                    prob = eval(func_str, {"__builtins__": {}}, safe_dict)
                    if prob < 0:
                        negative_probs.append(x_val)
                    total_prob += prob
                
                analysis.update({
                    "terms_computed": len(range_values),
                    "convergence_info": "Finite sum computed exactly",
                    "series_type": "finite"
                })
                tolerance = 1e-15

            if negative_probs:
                return False, f"Negative probabilities detected at x = {negative_probs[:3]}" + ("..." if len(negative_probs) > 3 else ""), total_prob, analysis

            if abs(total_prob - 1.0) > tolerance:
                return False, f"Probabilities sum to {total_prob:.15f}, should be 1.0", total_prob, analysis

            return True, f"Valid probability function (sum = {total_prob:.15f})", total_prob, analysis

        except Exception as e:
            return False, f"Error evaluating function: {str(e)}", 0, analysis

    @staticmethod
    def validate_crv_pdf(func_str: str, range_bounds: Tuple[float, float]) -> Tuple[bool, str, float]:
        try:
            safe_dict = {
                'sqrt': np.sqrt, 'exp': np.exp, 'log': np.log,
                'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
                'pi': np.pi, 'e': np.e
            }

            def pdf_func(x):
                safe_dict['x'] = x
                try:
                    result = eval(func_str, {"__builtins__": {}}, safe_dict)
                    return max(0, result)
                except:
                    return 0

            lower, upper = range_bounds
            
            if np.isinf(lower) and np.isinf(upper):
                test_points = np.concatenate([
                    np.linspace(-100, -1, 20),
                    np.linspace(-1, 1, 40),
                    np.linspace(1, 100, 20)
                ])
            elif np.isinf(lower):
                test_points = np.concatenate([
                    np.linspace(upper-200, upper-1, 30),
                    np.linspace(upper-1, upper, 20)
                ])
            elif np.isinf(upper):
                test_points = np.concatenate([
                    np.linspace(lower, lower+1, 20),
                    np.linspace(lower+1, lower+200, 30)
                ])
            else:
                test_points = np.linspace(lower, upper, 100)

            original_func = lambda x: eval(func_str, {"__builtins__": {}}, {**safe_dict, 'x': x})
            
            for point in test_points:
                try:
                    val = original_func(point)
                    if val < -1e-15:
                        return False, f"Negative PDF value {val:.15f} at x = {point:.6f}", 0
                except:
                    continue

            try:
                integral_result, error = integrate.quad(
                    pdf_func, lower, upper,
                    limit=500, epsabs=1e-15, epsrel=1e-15
                )
            except:
                integral_result, error = integrate.quad(pdf_func, lower, upper, limit=100)

            if abs(integral_result - 1.0) > 1e-4:
                return False, f"PDF integrates to {integral_result:.15f} ± {error:.2e}, not 1.0", integral_result

            return True, f"Valid PDF (integral = {integral_result:.15f} ± {error:.2e})", integral_result

        except Exception as e:
            return False, f"Error evaluating PDF: {str(e)}", 0

class EnhancedMomentCalculator:
    @staticmethod
    def calculate_drv_moment_infinite(func_str: str, range_values: List[float], r: int, a: float, max_iter: int = 10**6, tol: float = 1e-12) -> Tuple[float, dict]:
        safe_dict = {
            'factorial': math.factorial, 'sqrt': np.sqrt,  # ✅ FIXED
            'exp': np.exp, 'log': np.log, 'sin': np.sin,
            'cos': np.cos, 'tan': np.tan, 'pi': np.pi, 'e': np.e
        }

        def pmf(x):
            safe_dict['x'] = x
            try:
                return eval(func_str, {"__builtins__": {}}, safe_dict)
            except:
                return 0

        try:
            moment = rth_moment(pmf, "infinite", r, a, tol, max_iter)
            cumulative = 0.0
            terms_used = 0
            for x in range(max_iter):
                p = pmf(x)
                cumulative += p
                terms_used += 1
                if 1 - cumulative < tol:
                    break
            
            analysis = {
                "converged": True,
                "terms_used": terms_used,
                "convergence_info": f"Converged after {terms_used} terms using rth_moment algorithm"
            }
            return moment, analysis

        except Exception as e:
            moment = 0
            terms = []
            analysis = {"converged": False, "terms_used": 0, "convergence_info": ""}
            
            for i, x_val in enumerate(range_values[:200]):
                safe_dict['x'] = x_val
                prob = eval(func_str, {"__builtins__": {}}, safe_dict)
                term = ((x_val - a) ** r) * prob
                terms.append(term)
                moment += term
                
                if i >= 30 and i % 10 == 0:
                    recent_terms = terms[-10:]
                    if all(abs(t) < 1e-18 for t in recent_terms):
                        analysis.update({
                            "converged": True,
                            "terms_used": i + 1,
                            "convergence_info": f"Converged after {i+1} terms (terms → 0)"
                        })
                        break
                    
                    if len(terms) >= 50:
                        recent_sums = [sum(terms[:j]) for j in range(len(terms)-20, len(terms))]
                        sum_diffs = [abs(recent_sums[i+1] - recent_sums[i]) for i in range(len(recent_sums)-1)]
                        if all(d < 1e-15 for d in sum_diffs):
                            analysis.update({
                                "converged": True,
                                "terms_used": i + 1,
                                "convergence_info": f"Converged after {i+1} terms (partial sums stabilized)"
                            })
                            break

            if not analysis["converged"]:
                analysis.update({
                    "converged": False,
                    "terms_used": len(terms),
                    "convergence_info": f"Used {len(terms)} terms, convergence uncertain. Error: {str(e)}"
                })

            return moment, analysis

    @staticmethod
    def calculate_drv_moment(func_str: str, range_values: List[float], r: int, a: float, is_infinite: bool = False, max_iter: int = 10**6, tol: float = 1e-12) -> Tuple[float, dict]:
        if is_infinite:
            return EnhancedMomentCalculator.calculate_drv_moment_infinite(func_str, range_values, r, a, max_iter, tol)
        
        safe_dict = {
            'factorial': math.factorial, 'sqrt': np.sqrt,  # ✅ FIXED
            'exp': np.exp, 'log': np.log, 'sin': np.sin,
            'cos': np.cos, 'tan': np.tan, 'pi': np.pi, 'e': np.e
        }

        def pmf(x):
            safe_dict['x'] = x
            try:
                return eval(func_str, {"__builtins__": {}}, safe_dict)
            except:
                return 0

        try:
            moment = rth_moment(pmf, range_values, r, a, tol, max_iter)
            analysis = {
                "converged": True,
                "terms_used": len(range_values),
                "convergence_info": f"Exact calculation with {len(range_values)} terms using rth_moment algorithm"
            }
            return moment, analysis

        except Exception as e:
            moment = 0
            for x_val in range_values:
                safe_dict['x'] = x_val
                prob = eval(func_str, {"__builtins__": {}}, safe_dict)
                moment += ((x_val - a) ** r) * prob
            
            analysis = {
                "converged": True,
                "terms_used": len(range_values),
                "convergence_info": f"Exact calculation with {len(range_values)} terms (fallback)"
            }
            return moment, analysis

    @staticmethod
    def calculate_crv_moment(func_str: str, range_bounds: Tuple[float, float], r: int, a: float) -> float:
        safe_dict = {
            'sqrt': np.sqrt, 'exp': np.exp, 'log': np.log,
            'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
            'pi': np.pi, 'e': np.e
        }

        def integrand(x):
            safe_dict['x'] = x
            try:
                pdf_val = eval(func_str, {"__builtins__": {}}, safe_dict)
                return ((x - a) ** r) * pdf_val
            except:
                return 0

        try:
            moment, error = integrate.quad(
                integrand, range_bounds[0], range_bounds[1],
                limit=500, epsabs=1e-15, epsrel=1e-15
            )
            return moment
        except:
            moment, _ = integrate.quad(integrand, range_bounds[0], range_bounds[1], limit=100)
            return moment

# Implement the rth_moment function
def rth_moment(pmf, support, r, c=0, tol=1e-12, max_iter=10**6):
    moment = 0.0
    if support == "infinite":
        x = 0
        cumulative = 0.0
        while x < max_iter:
            p = pmf(x)
            if p < 0: 
                raise ValueError("PMF cannot be negative")
            moment += (x - c)**r * p
            cumulative += p
            if 1 - cumulative < tol:
                break
            x += 1
        else:
            raise RuntimeError("Maximum iterations reached. Tail too heavy?")
    else:
        for x in support:
            p = pmf(x)
            if p < 0: 
                raise ValueError("PMF cannot be negative")
            moment += (x - c)**r * p
    return moment

def parse_range_input(range_input: str) -> Tuple[List[float], bool, str]:
    range_input = range_input.strip()
    is_infinite = False
    pattern_info = ""
    
    if range_input.endswith("...") or range_input.endswith("…"):
        is_infinite = True
        base_values_str = range_input.replace("...", "").replace("…", "").strip()
        base_values = [float(x.strip()) for x in base_values_str.split(',') if x.strip()]
        
        pattern_type, params = InfiniteSeriesHandler.detect_series_pattern(base_values)
        
        if pattern_type == "arithmetic":
            pattern_info = f"Arithmetic sequence: start={params['start']}, step={params['diff']}"
            extended_values = InfiniteSeriesHandler.generate_extended_series(pattern_type, params, 200)
        elif pattern_type == "geometric":
            pattern_info = f"Geometric sequence: start={params['start']}, ratio={params['ratio']}"
            extended_values = InfiniteSeriesHandler.generate_extended_series(pattern_type, params, 200)
        else:
            pattern_info = "Custom pattern detected"
            extended_values = InfiniteSeriesHandler.generate_extended_series(pattern_type, params, 200)
        
        return extended_values, is_infinite, pattern_info
    
    values = [float(x.strip()) for x in range_input.split(',')]
    pattern_info = f"Finite sequence with {len(values)} values"
    return sorted(values), is_infinite, pattern_info

def parse_continuous_bound(bound_str: str) -> float:
    bound_str = bound_str.strip().lower()
    infinity_variants = ['inf', 'infinity', '∞', '+inf', '+infinity', '+∞']
    neg_infinity_variants = ['-inf', '-infinity', '-∞']
    
    if bound_str in infinity_variants:
        return np.inf
    elif bound_str in neg_infinity_variants:
        return -np.inf
    else:
        return float(bound_str)

# Unicode subscript mapping
SUBSCRIPT_MAP = {
    0: '₀', 1: '₁', 2: '₂', 3: '₃', 4: '₄',
    5: '₅', 6: '₆', 7: '₇', 8: '₈', 9: '₉'
}

def to_subscript(n):
    """Convert number to subscript string"""
    return ''.join(SUBSCRIPT_MAP.get(int(d), str(d)) for d in str(n))

# ✨ NEW: Streamlit-Native Landing Page (No HTML rendering issues!)
def show_landing_page():
    # Main header (keep your beautiful animated header)
    st.markdown("""
    <div class="main-header">
        <h1>Moments Calculator</h1>
        <p>Calculate statistical moments for discrete and continuous random variables</p>
    </div>
    """, unsafe_allow_html=True)
    
    # What are Moments? — using st.info for guaranteed rendering
    st.info("""
    **📘 What are Moments?**
    
    Moments describe the shape and characteristics of a probability distribution.
    The **r-th moment about point a** is defined as:
    
    ```
    E[(X - a)^r]
    ```
    
    • **1st moment about 0** = Mean  
    • **2nd moment about mean** = Variance  
    • **3rd moment about mean** = Skewness  
    • **4th moment about mean** = Kurtosis
    """)
    
    # Features — using st.success for visual appeal
    st.success("""
    **⚡ Features**
    
    • Support for **Discrete** and **Continuous** RVs    
    • **Adjustable precision** for calculation & display  
    • **Real-time validation** and error guidance  
    """)
    
    # How to Use — using st.warning for instructional tone
    st.warning("""
    **🎯 How to Use**
    
    1. Choose variable type (Discrete/Continuous)  
    2. Define range or bounds  
    3. Enter probability function  
    4. Select moment reference point  
    5. Set max moment order & precision  
    6. Click to compute!
    """)
    
    # Start button
    if st.button("🚀 Start Calculating", type="primary", use_container_width=True):
        st.session_state.show_calculator = True
        st.rerun()

# Initialize session state
if 'show_calculator' not in st.session_state:
    st.session_state.show_calculator = False
if 'calc_precision' not in st.session_state:
    st.session_state.calc_precision = 8
if 'display_precision' not in st.session_state:
    st.session_state.display_precision = 4

# Show landing page or calculator
if not st.session_state.show_calculator:
    show_landing_page()
else:
    # Sidebar
    with st.sidebar:
        st.markdown('<div class="section-header">🎯 Configuration</div>', unsafe_allow_html=True)
        
        # Precision settings
        st.markdown("### 🔢 Precision Settings")
        st.session_state.calc_precision = st.number_input(
            "Calculation Precision (decimal places)",
            min_value=1,
            max_value=15,
            value=8,
            help="Higher precision for internal calculations"
        )
        st.session_state.display_precision = st.number_input(
            "Display Precision (decimal places)",
            min_value=1,
            max_value=15,
            value=4,
            help="How many decimals to show in results"
        )
        
        st.markdown("---")
        st.markdown('<div class="section-header">🧮 Calculator Setup</div>', unsafe_allow_html=True)
        
        var_type = st.selectbox(
            "Choose Variable Type",
            ["Discrete (DRV)", "Continuous (CRV)"],
            index=0,
            help="🎲 Discrete: Countable values\n📊 Continuous: Any value in an interval"
        )
        
        st.markdown("---")
        st.markdown('<div class="section-header">📊 Range Definition</div>', unsafe_allow_html=True)
        
        if var_type == "Discrete (DRV)":
            st.markdown("""
            <div class="info-box">
                <strong>🔢 Discrete Range Examples:</strong><br>
                • Finite: <code>1,2,3,4,5</code><br>
                • Infinite: <code>0,1,2,3,...</code>
            </div>
            """, unsafe_allow_html=True)
            
            range_input = st.text_input(
                "Range Values (comma-separated)",
                value="",  # Blank by default
                help="Enter discrete values. End with '...' for infinite sequences",
                key="range_discrete"
            )
        else:
            st.markdown("""
            <div class="info-box">
                <strong>📊 Continuous Range Examples:</strong><br>
                • Finite: <code>[0, 1]</code><br>
                • Infinite: <code>[0, inf]</code>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                lower_bound_str = st.text_input(
                    "Lower Bound",
                    value="",  # Blank by default
                    help="Use '-inf' for -∞",
                    key="lower_bound"
                )
            with col2:
                upper_bound_str = st.text_input(
                    "Upper Bound",
                    value="",  # Blank by default
                    help="Use 'inf' for ∞",
                    key="upper_bound"
                )
        
        st.markdown("---")
        st.markdown('<div class="section-header">⚡ Probability Function</div>', unsafe_allow_html=True)
        
        if var_type == "Discrete (DRV)":
            prob_func = st.text_area(
                "Probability Mass Function P(X=x)",
                value="",  # Blank by default
                help="Available: factorial, sqrt, exp, log, sin, cos, tan, pi, e",
                height=100,
                key="prob_func_discrete"
            )
        else:
            prob_func = st.text_area(
                "Probability Density Function f(x)",
                value="",  # Blank by default
                help="Available: sqrt, exp, log, sin, cos, tan, pi, e",
                height=100,
                key="prob_func_continuous"
            )
        
        st.markdown("---")
        st.markdown('<div class="section-header">🎯 Moment Configuration</div>', unsafe_allow_html=True)
        
        moment_about = st.selectbox(
            "Calculate moments about:",
            ["About the origin (a = 0)", "About the mean (a = μ)", "About custom value"],
            help="Origin: Raw moments\nMean: Central moments\nCustom: Moments about any point"
        )
        
        if moment_about == "About custom value":
            custom_a = st.number_input(
                "Custom reference point (a)",
                value=0.0,
                step=0.1,
                help="Point around which to calculate moments"
            )
        
        st.markdown("---")
        st.markdown('<div class="section-header">📈 Moment Order</div>', unsafe_allow_html=True)
        
        max_moment_order = st.number_input(
            "Maximum moment order (r)",
            min_value=1,
            value=4,
            step=1,
            help="Calculate moments from 1st to r-th order"
        )
        if max_moment_order > 20:
            st.warning(f"⚠️ Computing up to {int(max_moment_order)}-th moment. This may take significant time and could be numerically unstable.")
        
        with st.expander("⚙️ Advanced Options", expanded=False):
            show_convergence = st.checkbox(
                "Show convergence analysis",
                value=True,
                help="Display detailed convergence information for infinite series"
            )
            max_terms = st.slider(
                "Max terms for infinite series",
                min_value=50,
                max_value=500,
                value=200,
                help="Maximum number of terms to compute for infinite series"
            )

    # Main content
    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown('<div class="section-header">🎲 Analysis Results</div>', unsafe_allow_html=True)
        
        if not prob_func:
            st.info("👆 Enter your probability function in the sidebar to begin calculation")
        else:
            try:
                is_infinite = False
                
                if var_type == "Discrete (DRV)":
                    if not range_input:
                        raise ValueError("Please enter range values")
                    
                    range_values, is_infinite, pattern_info = parse_range_input(range_input)
                    
                    if is_infinite:
                        st.markdown(f"""
                        <div class="info-box">
                            <strong>🔄 Infinite Series Detected:</strong><br>
                            {pattern_info}<br>
                            Computing with first {min(len(range_values), max_terms)} terms
                        </div>
                        """, unsafe_allow_html=True)
                        
                        range_values = range_values[:max_terms]
                    
                    is_valid, message, prob_sum, analysis = EnhancedProbabilityValidator.validate_drv_probabilities(
                        prob_func, range_values, is_infinite
                    )
                else:
                    if not lower_bound_str or not upper_bound_str:
                        raise ValueError("Please enter both lower and upper bounds")
                    
                    lower_bound = parse_continuous_bound(lower_bound_str)
                    upper_bound = parse_continuous_bound(upper_bound_str)
                    
                    if not np.isinf(lower_bound) and not np.isinf(upper_bound) and lower_bound >= upper_bound:
                        raise ValueError("Lower bound must be less than upper bound")
                    
                    range_bounds = (lower_bound, upper_bound)
                    
                    if np.isinf(lower_bound) or np.isinf(upper_bound):
                        bound_desc = f"({lower_bound_str}, {upper_bound_str})"
                        st.markdown(f"""
                        <div class="info-box">
                            <strong>♾️ Infinite Interval:</strong> Integration over {bound_desc}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    is_valid, message, integral_val = EnhancedProbabilityValidator.validate_crv_pdf(prob_func, range_bounds)
                
                if is_valid:
                    st.markdown(f"""
                    <div class="success-box">
                        <strong>✅ Validation Successful:</strong> {message}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if var_type == "Discrete (DRV)" and is_infinite and show_convergence:
                        st.markdown(f"""
                        <div class="info-box">
                            <strong>📊 Series Analysis:</strong><br>
                            • {analysis['convergence_info']}<br>
                            • Terms computed: {analysis['terms_computed']}<br>
                            • Series type: {analysis['series_type']}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    if moment_about == "About the mean (a = μ)":
                        with st.spinner("🧮 Computing mean..."):
                            if var_type == "Discrete (DRV)":
                                if is_infinite:
                                    mean_val, mean_analysis = EnhancedMomentCalculator.calculate_drv_moment_infinite(
                                        prob_func, range_values, 1, 0
                                    )
                                else:
                                    mean_val, mean_analysis = EnhancedMomentCalculator.calculate_drv_moment(
                                        prob_func, range_values, 1, 0, is_infinite
                                    )
                            else:
                                mean_val = EnhancedMomentCalculator.calculate_crv_moment(prob_func, range_bounds, 1, 0)
                        
                        a_value = mean_val
                        st.markdown(f"""
                        <div class="metric-container">
                            <div class="metric-label">Computed Mean (μ)</div>
                            <div class="metric-value">{mean_val:.{st.session_state.display_precision}f}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    elif moment_about == "About the origin (a = 0)":
                        a_value = 0.0
                    else:
                        a_value = custom_a
                    
                    st.markdown(f"#### 🧮 Computing Moments (1 to {max_moment_order})...")
                    
                    progress_bar = st.progress(0)
                    moments = {}
                    moment_analyses = {}
                    
                    for i in range(1, max_moment_order + 1):
                        progress_bar.progress(i / max_moment_order)
                        with st.spinner(f"Computing {i}-th moment..."):
                            if var_type == "Discrete (DRV)":
                                moment_val, moment_analysis = EnhancedMomentCalculator.calculate_drv_moment(
                                    prob_func, range_values, i, a_value, is_infinite
                                )
                                moment_analyses[i] = moment_analysis
                            else:
                                moment_val = EnhancedMomentCalculator.calculate_crv_moment(
                                    prob_func, range_bounds, i, a_value
                                )
                        
                        moments[i] = moment_val
                    
                    progress_bar.empty()
                    
                    # Display moments in compact grid (max 5 per row)
                    st.markdown("#### 📈 Calculated Moments")
                    cols_per_row = min(5, len(moments))
                    rows = (len(moments) + cols_per_row - 1) // cols_per_row
                    
                    for row in range(rows):
                        cols = st.columns(cols_per_row)
                        for col_idx in range(cols_per_row):
                            moment_idx = row * cols_per_row + col_idx + 1
                            if moment_idx in moments:
                                with cols[col_idx]:
                                    r = moment_idx
                                    moment_val = moments[r]
                                    colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#ff6b6b']
                                    color = colors[r % len(colors)]
                                    
                                    st.markdown(f"""
                                    <div class="metric-container" style="background: {color};">
                                        <div class="metric-label">μ{to_subscript(r)}</div>
                                        <div class="metric-value">{moment_val:.{st.session_state.display_precision}f}</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                    
                    if var_type == "Discrete (DRV)" and is_infinite and show_convergence:
                        st.markdown("#### 🔍 Convergence Analysis")
                        convergence_data = []
                        for r in moments.keys():
                            convergence_data.append({
                                'Moment Order': r,
                                'Value': f"{moments[r]:.{st.session_state.display_precision}f}",
                                'Converged': '✅' if moment_analyses[r]['converged'] else '⚠️',
                                'Terms Used': moment_analyses[r]['terms_used'],
                                'Info': moment_analyses[r]['convergence_info']
                            })
                        convergence_df = pd.DataFrame(convergence_data)
                        st.dataframe(convergence_df, use_container_width=True)
                    
                    if moment_about == "About the mean (a = μ)" and len(moments) >= 2:
                        st.markdown("#### 📊 Statistical Measures")
                        variance = moments[2]
                        std_dev = np.sqrt(abs(variance))
                        statistical_measures = [
                            ("Mean (μ)", a_value),
                            ("Variance (σ²)", variance),
                            ("Std Dev (σ)", std_dev)
                        ]
                        
                        if len(moments) >= 3 and std_dev > 1e-15:
                            skewness = moments[3] / (std_dev ** 3)
                            statistical_measures.append(("Skewness", skewness))
                        
                        if len(moments) >= 4 and std_dev > 1e-15:
                            kurtosis = moments[4] / (std_dev ** 4)
                            excess_kurtosis = kurtosis - 3
                            statistical_measures.append(("Kurtosis", kurtosis))
                            statistical_measures.append(("Excess Kurtosis", excess_kurtosis))
                        
                        cols_per_row = min(4, len(statistical_measures))
                        rows = (len(statistical_measures) + cols_per_row - 1) // cols_per_row
                        
                        for row in range(rows):
                            cols = st.columns(cols_per_row)
                            for col_idx in range(cols_per_row):
                                stat_idx = row * cols_per_row + col_idx
                                if stat_idx < len(statistical_measures):
                                    with cols[col_idx]:
                                        label, value = statistical_measures[stat_idx]
                                        colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#ff6b6b', '#45b7d1']
                                        color = colors[stat_idx % len(colors)]
                                        
                                        st.markdown(f"""
                                        <div class="metric-container" style="background: {color};">
                                            <div class="metric-label">{label}</div>
                                            <div class="metric-value">{value:.{st.session_state.display_precision}f}</div>
                                        </div>
                                        """, unsafe_allow_html=True)
                    
                    # Detailed results table (without "Interpretation" column)
                    st.markdown("#### 📋 Detailed Results Table")
                    results_data = []
                    for r, moment_val in moments.items():
                        row = {
                            'Moment Order (r)': r,
                            'Moment Value': f"{moment_val:.{st.session_state.display_precision}f}",
                        }
                        if var_type == "Discrete (DRV)" and is_infinite:
                            row.update({
                                'Convergence': '✅' if moment_analyses[r]['converged'] else '⚠️',
                                'Terms Used': moment_analyses[r]['terms_used']
                            })
                        results_data.append(row)
                    
                    results_df = pd.DataFrame(results_data)
                    st.dataframe(results_df, use_container_width=True)
                
                else:
                    st.markdown(f"""
                    <div class="error-box">
                        <strong>❌ Validation Failed:</strong> {message}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Fixed: Safe access to 'analysis'
                    if var_type == "Discrete (DRV)":
                        debug_info = f"""
                        <div class="warning-box">
                            <strong>🔧 Debug Information:</strong><br>
                            Current probability sum: <code>{prob_sum:.{st.session_state.display_precision}f}</code><br>
                            Expected sum: <code>1.0</code><br>
                            Difference: <code>{abs(prob_sum - 1.0):.{st.session_state.display_precision}f}</code>
                        """
                        # Only add analysis info if it exists
                        if 'analysis' in locals():
                            debug_info += f"<br>Terms computed: <code>{analysis.get('terms_computed', 'N/A')}</code>"
                        debug_info += "</div>"
                        st.markdown(debug_info, unsafe_allow_html=True)
                        
                        if is_infinite and 'analysis' in locals():
                            st.markdown(f"""
                            <div class="info-box">
                                <strong>♾️ Infinite Series Notes:</strong><br>
                                {analysis.get('convergence_info', 'No convergence info available')}
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="warning-box">
                            <strong>🔧 Debug Information:</strong><br>
                            Current integral value: <code>{integral_val:.{st.session_state.display_precision}f}</code><br>
                            Expected value: <code>1.0</code><br>
                            Integration bounds: <code>[{lower_bound_str}, {upper_bound_str}]</code>
                        </div>
                        """, unsafe_allow_html=True)
            
            except Exception as e:
                st.markdown(f"""
                <div class="error-box">
                    <strong>💥 Computation Error:</strong><br>
                    {str(e)}<br><br>
                    <strong>🔍 Common Issues:</strong><br>
                    • Check function syntax<br>
                    • Verify range format<br>
                    • Ensure mathematical validity
                </div>
                """, unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-header">📚 Help & Examples</div>', unsafe_allow_html=True)
        
        with st.expander("🎲 Discrete Examples", expanded=False):
            st.markdown("""
            **🔢 Finite Discrete:**
            ```python
            Range: 1,2,3,4,5
            PMF: 0.2 if 1 <= x <= 5 else 0
            ```
            
            **♾️ Geometric Distribution:**
            ```python
            Range: 0,1,2,3,...
            PMF: 0.3 * (0.7 ** x) if x >= 0 else 0
            ```
            
            **🎯 Poisson Distribution:**
            ```python
            Range: 0,1,2,3,...
            PMF: (exp(-3) * 3**x) / factorial(x) if x >= 0 else 0
            ```
            """)
        
        with st.expander("📊 Continuous Examples", expanded=False):
            st.markdown("""
            **📏 Uniform Distribution:**
            ```python
            Range: [0, 1]
            PDF: 1 if 0 <= x <= 1 else 0
            ```
            
            **⚡ Exponential Distribution:**
            ```python
            Range: [0, inf]
            PDF: 2*exp(-2*x) if x >= 0 else 0
            ```
            
            **🔔 Normal Distribution:**
            ```python
            Range: [-inf, inf]
            PDF: exp(-(x**2)/2) / sqrt(2*pi)
            ```
            """)
        
        with st.expander("♾️ Infinity Guide", expanded=False):
            st.markdown("""
            **🔄 Discrete Infinite Patterns:**
            - `0,1,2,3,...` → 0, 1, 2, 3, ...
            - `1,3,5,7,...` → 1, 3, 5, 7, ...
            
            **📊 Continuous Infinite Bounds:**
            - `inf`, `infinity`, `∞` for +∞
            - `-inf`, `-infinity`, `-∞` for -∞
            """)
        
        with st.expander("🧮 Mathematical Functions", expanded=False):
            st.markdown("""
            **🔧 Available Functions:**
            *Basic Math:*
            - `+`, `-`, `*`, `/`, `**`
            - `sqrt(x)`, `abs(x)`
            
            *Exponential & Logarithmic:*
            - `exp(x)`, `log(x)`, `log10(x)`
            
            *Trigonometric:*
            - `sin(x)`, `cos(x)`, `tan(x)`
            
            *Special Functions:*
            - `factorial(n)` (discrete only)
            
            *Constants:*
            - `pi`, `e`
            """)

    if st.button("🏠 Back to Home", use_container_width=True):
        st.session_state.show_calculator = False
        st.rerun()

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <small>Moments Calculator • Built with Streamlit • © 2024</small>
</div>
""", unsafe_allow_html=True)
