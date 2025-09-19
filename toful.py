import streamlit as st
import numpy as np
from scipy import integrate
import pandas as pd
from typing import Tuple, Union, List
import warnings
warnings.filterwarnings('ignore')

# Set page config for better aesthetics
st.set_page_config(
    page_title="Moments Calculator",
    page_icon="🎲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS with modern dark-friendly design
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
        padding: 2rem 1.5rem;
        border-radius: 16px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
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
        transform: translateY(-5px) scale(1.02);
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.4);
    }

    .metric-container:hover::before {
        left: 100%;
    }

    .metric-label {
        font-size: 0.9rem;
        font-weight: 500;
        margin-bottom: 0.8rem;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .metric-value {
        font-size: 1.8rem;
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
</style>
""", unsafe_allow_html=True)

# Animated main header
st.markdown("""
<div class="main-header">
    <h1>Moments Calculator</h1>
    <p>Advanced moment calculation with enhanced precision</p>
</div>
""", unsafe_allow_html=True)

class InfiniteSeriesHandler:
    """Enhanced handler for infinite discrete series with better convergence detection"""

    @staticmethod
    def detect_series_pattern(values: List[float]) -> Tuple[str, dict]:
        """Detect the pattern in a series and return type and parameters"""
        if len(values) < 2:
            return "unknown", {}

        # Check for arithmetic progression
        diffs = [values[i+1] - values[i] for i in range(len(values)-1)]
        if len(set(diffs)) == 1:  # Constant difference
            return "arithmetic", {"start": values[0], "diff": diffs[0]}

        # Check for geometric progression
        if all(v != 0 for v in values):
            ratios = [values[i+1] / values[i] for i in range(len(values)-1)]
            if all(abs(ratios[i] - ratios[0]) < 1e-15 for i in range(len(ratios))):
                return "geometric", {"start": values[0], "ratio": ratios[0]}

        return "custom", {"values": values}

    @staticmethod
    def generate_extended_series(pattern_type: str, params: dict, max_terms: int = 100) -> List[float]:
        """Generate extended series based on detected pattern"""
        if pattern_type == "arithmetic":
            start, diff = params["start"], params["diff"]
            return [start + i * diff for i in range(max_terms)]

        elif pattern_type == "geometric":
            start, ratio = params["start"], params["ratio"]
            return [start * (ratio ** i) for i in range(max_terms)]

        else:  # custom or unknown
            base_values = params.get("values", [0])
            # Extend with last difference if possible
            if len(base_values) >= 2:
                diff = base_values[-1] - base_values[-2]
                extended = base_values.copy()
                while len(extended) < max_terms:
                    extended.append(extended[-1] + diff)
                return extended
            else:
                # Default to integers starting from the first value
                return [base_values[0] + i for i in range(max_terms)]

    @staticmethod
    def estimate_infinite_sum(func_str: str, values: List[float], pattern_type: str, params: dict) -> Tuple[float, bool, str]:
        """Estimate the sum of infinite series with convergence analysis"""
        safe_dict = {
            'x': 0, 'factorial': np.math.factorial, 'sqrt': np.sqrt,
            'exp': np.exp, 'log': np.log, 'sin': np.sin, 'cos': np.cos,
            'tan': np.tan, 'pi': np.pi, 'e': np.e
        }

        # Calculate first 50 terms for better precision
        partial_sum = 0
        terms = []

        try:
            for i, x_val in enumerate(values[:50]):
                safe_dict['x'] = x_val
                term = eval(func_str, {"__builtins__": {}}, safe_dict)
                terms.append(term)
                partial_sum += term

            # Enhanced convergence analysis
            if len(terms) >= 20:
                # Check if terms are decreasing and approaching zero
                recent_terms = terms[-10:]
                if all(abs(t) < 1e-15 for t in recent_terms):
                    return partial_sum, True, f"Series converges (terms → 0, sum ≈ {partial_sum:.15f})"

                # Ratio test for geometric-like series
                if len(terms) >= 25 and all(abs(t) > 1e-20 for t in terms[-15:]):
                    ratios = [abs(terms[i+1]/terms[i]) for i in range(len(terms)-10, len(terms)-1)]
                    avg_ratio = sum(ratios) / len(ratios)

                    if avg_ratio < 0.9:  # Strong convergence
                        # Estimate remaining sum using geometric series formula
                        remaining_estimate = terms[-1] * avg_ratio / (1 - avg_ratio)
                        estimated_total = partial_sum + remaining_estimate
                        return estimated_total, True, f"Series converges (ratio test, sum ≈ {estimated_total:.15f})"
                    elif avg_ratio > 1.1:
                        return partial_sum, False, "Series appears to diverge (ratio test)"

                # Check for alternating series
                if len(terms) >= 15:
                    signs = [1 if t >= 0 else -1 for t in terms[-15:]]
                    if len(set(signs)) == 2:  # Both positive and negative
                        abs_terms = [abs(t) for t in terms[-10:]]
                        if all(abs_terms[i] >= abs_terms[i+1] for i in range(len(abs_terms)-1)):
                            return partial_sum, True, f"Alternating series converges (sum ≈ {partial_sum:.15f})"

            # Default: use partial sum with warning
            return partial_sum, False, f"Convergence uncertain (partial sum of {len(terms)} terms: {partial_sum:.15f})"

        except Exception as e:
            return 0, False, f"Error in series evaluation: {str(e)}"

class EnhancedProbabilityValidator:
    @staticmethod
    def validate_drv_probabilities(func_str: str, range_values: List[float], is_infinite: bool = False) -> Tuple[bool, str, float, dict]:
        """Enhanced validation with detailed convergence analysis"""
        try:
            safe_dict = {
                'x': 0, 'factorial': np.math.factorial, 'sqrt': np.sqrt,
                'exp': np.exp, 'log': np.log, 'sin': np.sin, 'cos': np.cos,
                'tan': np.tan, 'pi': np.pi, 'e': np.e
            }

            negative_probs = []
            analysis = {"terms_computed": 0, "convergence_info": "", "series_type": "finite"}

            if is_infinite:
                # Use enhanced infinite series handler
                pattern_type, params = InfiniteSeriesHandler.detect_series_pattern(range_values[:10])
                total_prob, converges, convergence_msg = InfiniteSeriesHandler.estimate_infinite_sum(
                    func_str, range_values, pattern_type, params
                )

                # Check for negative probabilities in sample
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
                # Finite case
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

            # Validation results
            if negative_probs:
                return False, f"Negative probabilities detected at x = {negative_probs[:3]}" + ("..." if len(negative_probs) > 3 else ""), total_prob, analysis

            if abs(total_prob - 1.0) > tolerance:
                return False, f"Probabilities sum to {total_prob:.15f}, should be 1.0", total_prob, analysis

            return True, f"Valid probability function (sum = {total_prob:.15f})", total_prob, analysis

        except Exception as e:
            return False, f"Error evaluating function: {str(e)}", 0, analysis

    @staticmethod
    def validate_crv_pdf(func_str: str, range_bounds: Tuple[float, float]) -> Tuple[bool, str, float]:
        """Enhanced continuous validation with higher precision"""
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
                    return max(0, result)  # Ensure non-negative
                except:
                    return 0

            lower, upper = range_bounds

            # Adaptive sampling for different interval types with more points
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

            # Check for negative values
            original_func = lambda x: eval(func_str, {"__builtins__": {}}, {**safe_dict, 'x': x})
            for point in test_points:
                try:
                    val = original_func(point)
                    if val < -1e-15:  # Allow for small numerical errors
                        return False, f"Negative PDF value {val:.15f} at x = {point:.6f}", 0
                except:
                    continue

            # Enhanced numerical integration with higher precision
            try:
                integral_result, error = integrate.quad(
                    pdf_func, lower, upper,
                    limit=500, epsabs=1e-15, epsrel=1e-15
                )
            except:
                # Fallback integration
                integral_result, error = integrate.quad(pdf_func, lower, upper, limit=100)

            if abs(integral_result - 1.0) > 1e-4:
                return False, f"PDF integrates to {integral_result:.15f} ± {error:.2e}, not 1.0", integral_result

            return True, f"Valid PDF (integral = {integral_result:.15f} ± {error:.2e})", integral_result

        except Exception as e:
            return False, f"Error evaluating PDF: {str(e)}", 0

class EnhancedMomentCalculator:
    @staticmethod
    def calculate_drv_moment_infinite(func_str: str, range_values: List[float], r: int, a: float) -> Tuple[float, dict]:
        """Calculate moments for infinite DRV with enhanced precision and convergence analysis"""
        safe_dict = {
            'factorial': np.math.factorial, 'sqrt': np.sqrt,
            'exp': np.exp, 'log': np.log, 'sin': np.sin,
            'cos': np.cos, 'tan': np.tan, 'pi': np.pi, 'e': np.e
        }

        moment = 0
        terms = []
        analysis = {"converged": False, "terms_used": 0, "convergence_info": ""}

        # Calculate terms and monitor convergence with higher precision
        for i, x_val in enumerate(range_values[:200]):  # Increased max terms
            safe_dict['x'] = x_val
            prob = eval(func_str, {"__builtins__": {}}, safe_dict)
            term = ((x_val - a) ** r) * prob
            terms.append(term)
            moment += term

            # Check convergence every 10 terms after the first 30
            if i >= 30 and i % 10 == 0:
                recent_terms = terms[-10:]
                if all(abs(t) < 1e-18 for t in recent_terms):
                    analysis.update({
                        "converged": True,
                        "terms_used": i + 1,
                        "convergence_info": f"Converged after {i+1} terms (terms → 0)"
                    })
                    break

                # Check if partial sums are stabilizing with higher precision
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
                "convergence_info": f"Used {len(terms)} terms, convergence uncertain"
            })

        return moment, analysis

    @staticmethod
    def calculate_drv_moment(func_str: str, range_values: List[float], r: int, a: float, is_infinite: bool = False) -> Tuple[float, dict]:
        """Calculate r-th moment for DRV with enhanced precision"""
        if is_infinite:
            return EnhancedMomentCalculator.calculate_drv_moment_infinite(func_str, range_values, r, a)

        # Finite case with higher precision
        safe_dict = {
            'factorial': np.math.factorial, 'sqrt': np.sqrt,
            'exp': np.exp, 'log': np.log, 'sin': np.sin,
            'cos': np.cos, 'tan': np.tan, 'pi': np.pi, 'e': np.e
        }

        moment = 0
        for x_val in range_values:
            safe_dict['x'] = x_val
            prob = eval(func_str, {"__builtins__": {}}, safe_dict)
            moment += ((x_val - a) ** r) * prob

        analysis = {
            "converged": True,
            "terms_used": len(range_values),
            "convergence_info": f"Exact calculation with {len(range_values)} terms"
        }

        return moment, analysis

    @staticmethod
    def calculate_crv_moment(func_str: str, range_bounds: Tuple[float, float], r: int, a: float) -> float:
        """Calculate r-th moment for CRV with enhanced precision"""
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
            # Fallback integration
            moment, _ = integrate.quad(integrand, range_bounds[0], range_bounds[1], limit=100)
            return moment

def parse_range_input(range_input: str) -> Tuple[List[float], bool, str]:
    """Enhanced range parsing with pattern detection"""
    range_input = range_input.strip()
    is_infinite = False
    pattern_info = ""

    if range_input.endswith("...") or range_input.endswith("…"):
        is_infinite = True
        base_values_str = range_input.replace("...", "").replace("…", "").strip()
        base_values = [float(x.strip()) for x in base_values_str.split(',') if x.strip()]

        # Detect pattern
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

    # Regular finite range
    values = [float(x.strip()) for x in range_input.split(',')]
    pattern_info = f"Finite sequence with {len(values)} values"
    return sorted(values), is_infinite, pattern_info

def parse_continuous_bound(bound_str: str) -> float:
    """Parse continuous bound with enhanced infinity handling"""
    bound_str = bound_str.strip().lower()
    infinity_variants = ['inf', 'infinity', '∞', '+inf', '+infinity', '+∞']
    neg_infinity_variants = ['-inf', '-infinity', '-∞']

    if bound_str in infinity_variants:
        return np.inf
    elif bound_str in neg_infinity_variants:
        return -np.inf
    else:
        return float(bound_str)

# Sidebar with enhanced UI
with st.sidebar:
    st.markdown('<div class="section-header">🎯 Configuration</div>', unsafe_allow_html=True)

    # Progress indicator
    st.markdown("""
    <div class="progress-container">
        <div class="progress-bar" style="width: 25%"></div>
    </div>
    <small style="color: #666;">Step 1 of 5: Variable Type</small>
    """, unsafe_allow_html=True)

    # Step 1: Variable type with enhanced descriptions
    var_type = st.selectbox(
        "Choose Variable Type",
        ["Discrete (DRV)", "Continuous (CRV)"],
        help="🎲 Discrete: Countable values (coins, dice)\n📊 Continuous: Any value in an interval (height, weight)"
    )

    st.markdown("---")

    # Step 2: Range input with better guidance
    st.markdown('<div class="section-header">📊 Range Definition</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="progress-container">
        <div class="progress-bar" style="width: 40%"></div>
    </div>
    <small style="color: #666;">Step 2 of 5: Define Range</small>
    """, unsafe_allow_html=True)

    if var_type == "Discrete (DRV)":
        st.markdown("""
        <div class="info-box">
            <strong>🔢 Discrete Range Examples:</strong><br>
            • Finite: <code>1,2,3,4,5</code><br>
            • Infinite arithmetic: <code>0,1,2,3,...</code><br>
            • Infinite geometric: <code>1,2,4,8,...</code><br>
            • Custom pattern: <code>1,4,9,16,...</code>
        </div>
        """, unsafe_allow_html=True)

        range_input = st.text_input(
            "Range Values (comma-separated)",
            value="0,1,2,3,...",
            help="📝 Enter discrete values. End with '...' for infinite sequences",
            key="range_discrete"
        )
    else:
        st.markdown("""
        <div class="info-box">
            <strong>📊 Continuous Range Examples:</strong><br>
            • Finite: <code>[0, 1]</code><br>
            • Semi-infinite: <code>[0, inf]</code><br>
            • Infinite: <code>[-inf, inf]</code><br>
            Use 'inf' or '∞' for infinity
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            lower_bound_str = st.text_input(
                "Lower Bound",
                value="0",
                help="📉 Use '-inf' for -∞",
                key="lower_bound"
            )
        with col2:
            upper_bound_str = st.text_input(
                "Upper Bound",
                value="1",
                help="📈 Use 'inf' for ∞",
                key="upper_bound"
            )

    st.markdown("---")

    # Step 3: Probability function with examples
    st.markdown('<div class="section-header">⚡ Probability Function</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="progress-container">
        <div class="progress-bar" style="width: 60%"></div>
    </div>
    <small style="color: #666;">Step 3 of 5: Define Function</small>
    """, unsafe_allow_html=True)

    # Function examples dropdown
    if var_type == "Discrete (DRV)":
        example_functions = {
            "Geometric Distribution": "0.5 * (0.5 ** x) if x >= 0 else 0",
            "Poisson-like": "(2**x * exp(-2)) / factorial(x) if x >= 0 else 0",
            "Custom Finite": "0.25 if 0 <= x <= 3 else 0",
            "Negative Binomial": "(factorial(x+1)/(factorial(1)*factorial(x))) * (0.5**2) * (0.5**x) if x >= 0 else 0"
        }
    else:
        example_functions = {
            "Exponential Distribution": "2*exp(-2*x) if x >= 0 else 0",
            "Beta Distribution": "6*x*(1-x) if 0 <= x <= 1 else 0",
            "Normal-like": "exp(-(x**2)/2) / sqrt(2*pi)",
            "Uniform Distribution": "1 if 0 <= x <= 1 else 0"
        }

    selected_example = st.selectbox(
        "📚 Choose Example or Custom",
        ["Custom"] + list(example_functions.keys()),
        key="example_selector"
    )

    default_func = example_functions.get(selected_example,
        "0.5 * (0.5 ** x) if x >= 0 else 0" if var_type == "Discrete (DRV)" else "exp(-x) if x >= 0 else 0")

    if var_type == "Discrete (DRV)":
        prob_func = st.text_area(
            "Probability Mass Function P(X=x)",
            value=default_func if selected_example != "Custom" else "0.5 * (0.5 ** x) if x >= 0 else 0",
            help="🧮 Available: factorial, sqrt, exp, log, sin, cos, tan, pi, e",
            height=100,
            key="prob_func_discrete"
        )
    else:
        prob_func = st.text_area(
            "Probability Density Function f(x)",
            value=default_func if selected_example != "Custom" else "exp(-x) if x >= 0 else 0",
            help="🧮 Available: sqrt, exp, log, sin, cos, tan, pi, e",
            height=100,
            key="prob_func_continuous"
        )

    st.markdown("---")

    # Step 4: Moment calculation options
    st.markdown('<div class="section-header">🎯 Moment Configuration</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="progress-container">
        <div class="progress-bar" style="width: 80%"></div>
    </div>
    <small style="color: #666;">Step 4 of 5: Moment Settings</small>
    """, unsafe_allow_html=True)

    moment_about = st.selectbox(
        "Calculate moments about:",
        ["About the origin (a = 0)", "About the mean (a = μ)", "About custom value"],
        help="🎯 Origin: Raw moments\n📊 Mean: Central moments\n⚙️ Custom: Moments about any point"
    )

    if moment_about == "About custom value":
        custom_a = st.number_input(
            "Custom reference point (a)",
            value=0.0,
            step=0.1,
            help="📍 Point around which to calculate moments"
        )

    st.markdown("---")

    # Step 5: Maximum moment order
    st.markdown('<div class="section-header">📈 Moment Order</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="progress-container">
        <div class="progress-bar" style="width: 100%"></div>
    </div>
    <small style="color: #666;">Step 5 of 5: Calculation Range</small>
    """, unsafe_allow_html=True)

    max_moment_order = st.number_input(
        "Maximum moment order (r)",
        min_value=1,
        max_value=20,
        value=4,
        step=1,
        help="📊 Calculate moments from 1st to r-th order"
    )

    # Advanced options
    with st.expander("⚙️ Advanced Options", expanded=False):
        show_convergence = st.checkbox(
            "Show convergence analysis",
            value=True,
            help="📈 Display detailed convergence information for infinite series"
        )
        max_terms = st.slider(
            "Max terms for infinite series",
            min_value=50,
            max_value=500,
            value=200,
            help="🔄 Maximum number of terms to compute for infinite series"
        )

# Main content area with enhanced layout
col1, col2 = st.columns([3, 2])

with col1:
    st.markdown('<div class="section-header">🎲 Analysis Results</div>', unsafe_allow_html=True)

    # Process inputs and validate
    try:
        # Parse range with enhanced pattern detection
        if var_type == "Discrete (DRV)":
            range_values, is_infinite, pattern_info = parse_range_input(range_input)

            if is_infinite:
                st.markdown(f"""
                <div class="info-box">
                    <strong>🔄 Infinite Series Detected:</strong><br>
                    {pattern_info}<br>
                    Computing with first {min(len(range_values), max_terms)} terms
                </div>
                """, unsafe_allow_html=True)

                # Limit to max_terms
                range_values = range_values[:max_terms]

            # Enhanced validation
            is_valid, message, prob_sum, analysis = EnhancedProbabilityValidator.validate_drv_probabilities(
                prob_func, range_values, is_infinite
            )

        else:
            try:
                lower_bound = parse_continuous_bound(lower_bound_str)
                upper_bound = parse_continuous_bound(upper_bound_str)
            except ValueError as e:
                raise ValueError(f"Invalid bound format: {str(e)}")

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

            # Validate PDF
            is_valid, message, integral_val = EnhancedProbabilityValidator.validate_crv_pdf(prob_func, range_bounds)

        # Display validation results with enhanced styling
        if is_valid:
            st.markdown(f"""
            <div class="success-box">
                <strong>✅ Validation Successful:</strong> {message}
            </div>
            """, unsafe_allow_html=True)

            # Show detailed analysis for infinite series
            if var_type == "Discrete (DRV)" and is_infinite and show_convergence:
                st.markdown(f"""
                <div class="info-box">
                    <strong>📊 Series Analysis:</strong><br>
                    • {analysis['convergence_info']}<br>
                    • Terms computed: {analysis['terms_computed']}<br>
                    • Series type: {analysis['series_type']}<br>
                    {f"• Pattern: {analysis.get('pattern_type', 'N/A')}" if 'pattern_type' in analysis else ""}
                </div>
                """, unsafe_allow_html=True)

            # Calculate reference point for moments
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
                    <div class="metric-value">{mean_val:.15f}</div>
                </div>
                """, unsafe_allow_html=True)

            elif moment_about == "About the origin (a = 0)":
                a_value = 0.0
            else:
                a_value = custom_a

            # Calculate multiple moments with progress indication
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

            # Display moments with enhanced styling
            st.markdown("#### 📈 Calculated Moments")

            # Display moments in a responsive grid
            num_cols = min(6, len(moments))  # Max 6 columns
            if len(moments) <= 6:
                cols = st.columns(num_cols)
                for i, (r, moment_val) in enumerate(moments.items()):
                    with cols[i]:
                        # Color gradient based on moment order
                        colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#ff6b6b', '#45b7d1']
                        color_idx = i % len(colors)
                        st.markdown(f"""
                        <div class="metric-container" style="background: {colors[color_idx]};">
                            <div class="metric-label">μ_{r}({a_value:.2f})</div>
                            <div class="metric-value">{moment_val:.12f}</div>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                # For many moments, display in rows
                rows = (len(moments) + 5) // 6  # 6 moments per row
                for row in range(rows):
                    start_idx = row * 6
                    end_idx = min(start_idx + 6, len(moments))
                    row_moments = list(moments.items())[start_idx:end_idx]
                    
                    cols = st.columns(len(row_moments))
                    for i, (r, moment_val) in enumerate(row_moments):
                        with cols[i]:
                            colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#ff6b6b', '#45b7d1']
                            color_idx = (start_idx + i) % len(colors)
                            st.markdown(f"""
                            <div class="metric-container" style="background: {colors[color_idx]};">
                                <div class="metric-label">μ_{r}({a_value:.2f})</div>
                                <div class="metric-value">{moment_val:.12f}</div>
                            </div>
                            """, unsafe_allow_html=True)

            # Show convergence details for infinite DRV
            if var_type == "Discrete (DRV)" and is_infinite and show_convergence:
                st.markdown("#### 🔍 Convergence Analysis")

                convergence_df = pd.DataFrame([
                    {
                        'Moment Order': r,
                        'Value': f"{moments[r]:.15f}",
                        'Converged': '✅' if moment_analyses[r]['converged'] else '⚠️',
                        'Terms Used': moment_analyses[r]['terms_used'],
                        'Info': moment_analyses[r]['convergence_info']
                    }
                    for r in moments.keys()
                ])

                st.dataframe(convergence_df, use_container_width=True)

            # Statistical measures for central moments
            if moment_about == "About the mean (a = μ)" and len(moments) >= 2:
                st.markdown("#### 📊 Statistical Measures")

                variance = moments[2]
                std_dev = np.sqrt(abs(variance))

                statistical_measures = [
                    ("Mean (μ)", a_value),
                    ("Variance (σ²)", variance),
                    ("Std Dev (σ)", std_dev)
                ]

                # Add skewness and kurtosis if available
                if len(moments) >= 3 and std_dev > 1e-15:
                    skewness = moments[3] / (std_dev ** 3)
                    statistical_measures.append(("Skewness", skewness))

                if len(moments) >= 4 and std_dev > 1e-15:
                    kurtosis = moments[4] / (std_dev ** 4)
                    excess_kurtosis = kurtosis - 3
                    statistical_measures.extend([
                        ("Kurtosis", kurtosis),
                        ("Excess Kurtosis", excess_kurtosis)
                    ])

                # Create metric cards
                num_stats = len(statistical_measures)
                if num_stats <= 5:
                    metric_cols = st.columns(num_stats)
                    colors = ["#667eea", "#764ba2", "#f093fb", "#f5576c", "#ff6b6b"]
                    
                    for i, (label, value) in enumerate(statistical_measures):
                        with metric_cols[i]:
                            st.markdown(f"""
                            <div class="metric-container" style="background: {colors[i]};">
                                <div class="metric-label">{label}</div>
                                <div class="metric-value">{value:.12f}</div>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    # Display in rows for many statistics
                    rows = (num_stats + 4) // 5
                    colors = ["#667eea", "#764ba2", "#f093fb", "#f5576c", "#ff6b6b"]
                    
                    for row in range(rows):
                        start_idx = row * 5
                        end_idx = min(start_idx + 5, num_stats)
                        row_stats = statistical_measures[start_idx:end_idx]
                        
                        metric_cols = st.columns(len(row_stats))
                        for i, (label, value) in enumerate(row_stats):
                            with metric_cols[i]:
                                color_idx = (start_idx + i) % len(colors)
                                st.markdown(f"""
                                <div class="metric-container" style="background: {colors[color_idx]};">
                                    <div class="metric-label">{label}</div>
                                    <div class="metric-value">{value:.12f}</div>
                                </div>
                                """, unsafe_allow_html=True)

            # Detailed results table
            st.markdown("#### 📋 Detailed Results Table")

            results_data = []
            for r, moment_val in moments.items():
                row = {
                    'Moment Order (r)': r,
                    'Moment Value': f"{moment_val:.18f}",
                    'About Point (a)': f"{a_value:.12f}",
                    'Interpretation': f"E[(X-{a_value:.2f})^{r}]"
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

            # Provide helpful debugging information
            if var_type == "Discrete (DRV)":
                st.markdown(f"""
                <div class="warning-box">
                    <strong>🔧 Debug Information:</strong><br>
                    Current probability sum: <code>{prob_sum:.15f}</code><br>
                    Expected sum: <code>1.0</code><br>
                    Difference: <code>{abs(prob_sum - 1.0):.15f}</code><br>
                    Terms computed: <code>{analysis.get('terms_computed', 'N/A')}</code>
                </div>
                """, unsafe_allow_html=True)

                # Show convergence issues for infinite series
                if is_infinite:
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
                    Current integral value: <code>{integral_val:.15f}</code><br>
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

    # Enhanced help sections
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

        **🎪 Negative Binomial:**
        ```python
        Range: 0,1,2,3,...
        PMF: (factorial(x+r-1)/(factorial(r-1)*factorial(x))) * (p**r) * ((1-p)**x)
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

        **🎨 Beta Distribution:**
        ```python
        Range: [0, 1]
        PDF: 6*x*(1-x) if 0 <= x <= 1 else 0
        ```
        """)

    with st.expander("♾️ Infinity Guide", expanded=False):
        st.markdown("""
        **🔄 Discrete Infinite Patterns:**

        *Arithmetic Sequences:*
        - `0,1,2,3,...` → 0, 1, 2, 3, 4, 5, ...
        - `1,3,5,7,...` → 1, 3, 5, 7, 9, 11, ...
        - `2,5,8,11,...` → 2, 5, 8, 11, 14, 17, ...

        *Geometric Sequences:*
        - `1,2,4,8,...` → 1, 2, 4, 8, 16, 32, ...
        - `3,6,12,24,...` → 3, 6, 12, 24, 48, 96, ...

        **📊 Continuous Infinite Bounds:**
        - `inf`, `infinity`, `∞` for +∞
        - `-inf`, `-infinity`, `-∞` for -∞
        - Mixed: `[0, inf]`, `[-inf, 0]`, `[-inf, inf]`
        """)

    with st.expander("🧮 Mathematical Functions", expanded=False):
        st.markdown("""
        **🔧 Available Functions:**

        *Basic Math:*
        - `+`, `-`, `*`, `/`, `**` (power)
        - `sqrt(x)` - Square root
        - `abs(x)` - Absolute value

        *Exponential & Logarithmic:*
        - `exp(x)` - e^x
        - `log(x)` - Natural log (ln)
        - `log10(x)` - Base-10 log

        *Trigonometric:*
        - `sin(x)`, `cos(x)`, `tan(x)`
        - `asin(x)`, `acos(x)`, `atan(x)`

        *Special Functions:*
        - `factorial(n)` - n! (discrete only)
        - `gamma(x)` - Gamma function

        *Constants:*
        - `pi` - π ≈ 3.14159
        - `e` - Euler's number ≈ 2.71828
        """)

    with st.expander("🎯 Moment Interpretation", expanded=False):
        st.markdown("""
        **📊 Moment Meanings:**

        *Raw Moments (about origin):*
        - **μ₁(0)** = Mean (E[X])
        - **μ₂(0)** = Second moment (E[X²])
        - **μ₃(0)** = Third moment (E[X³])
        - **μᵣ(0)** = r-th moment (E[Xʳ])

        *Central Moments (about mean):*
        - **μ₁(μ)** = 0 (always)
        - **μ₂(μ)** = Variance (σ²)
        - **μ₃(μ)** = Used for skewness
        - **μ₄(μ)** = Used for kurtosis

        *Statistical Measures:*
        - **Skewness** = μ₃/σ³ (asymmetry)
        - **Kurtosis** = μ₄/σ⁴ (tail heaviness)
        - **Excess Kurtosis** = Kurtosis - 3

        *Higher Order Moments:*
        - Capture distributional shape
        - Useful for risk analysis
        - Important in financial modeling
        """)

# Enhanced footer with statistics
st.markdown("---")

# Show computation statistics if available
if 'is_valid' in locals() and is_valid:
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if var_type == "Discrete (DRV)" and is_infinite:
            terms_used = max([moment_analyses[r]['terms_used'] for r in moments.keys()])
            st.metric("Terms Computed", f"{terms_used}")
        else:
            st.metric("Variable Type", "Continuous" if var_type == "Continuous (CRV)" else "Discrete")

    with col2:
        if 'prob_sum' in locals():
            st.metric("Probability Sum", f"{prob_sum:.12f}")
        elif 'integral_val' in locals():
            st.metric("PDF Integral", f"{integral_val:.12f}")

    with col3:
        if 'a_value' in locals():
            st.metric("Reference Point", f"{a_value:.8f}")

    with col4:
        if var_type == "Discrete (DRV)" and is_infinite:
            convergence_count = sum(1 for r in moments.keys() if moment_analyses[r]['converged'])
            st.metric("Converged Moments", f"{convergence_count}/{len(moments)}")
        elif 'moments' in locals():
            st.metric("Moments Calculated", f"{len(moments)}")
