import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

# Enhanced CSS with modern, clean design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    .main {
        font-family: 'Inter', sans-serif;
    }
    /* Clean gradient header */
    .main-header {
        background: linear-gradient(135deg, #4361ee, #3a0ca3);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 6px 20px rgba(67, 97, 238, 0.2);
    }
    .main-header h1 {
        margin: 0;
        font-size: 2.8rem;
        font-weight: 700;
        text-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    .main-header p {
        margin: 1rem 0 0 0;
        font-size: 1.2rem;
        opacity: 0.95;
        font-weight: 400;
    }
    /* Clean input styling */
    .stSelectbox > div > div > div,
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stNumberInput > div > div > input {
        background: rgba(255, 255, 255, 0.95);
        border: 2px solid #e2e8f0;
        border-radius: 12px;
        color: #2d3436;
        font-size: 14px;
        transition: all 0.3s ease;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        padding: 12px;
    }
    .stSelectbox > div > div > div:hover,
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus,
    .stNumberInput > div > div > input:focus {
        border-color: #4361ee;
        box-shadow: 0 0 15px rgba(67, 97, 238, 0.2);
        transform: translateY(-1px);
    }
    .stTextArea > div > div > textarea {
        font-family: 'SF Mono', 'Monaco', 'Inconsolata', 'Roboto Mono', monospace;
        min-height: 120px;
    }
    /* Metric cards with subtle animation */
    .metric-container {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        transition: all 0.3s ease;
        border: 2px solid #f1f5f9;
    }
    .metric-container:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 16px rgba(0,0,0,0.12);
        border-color: #cbd5e1;
    }
    .metric-label {
        font-size: 0.9rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        color: #475569;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        margin: 0;
        font-family: 'SF Mono', monospace;
        color: #1e293b;
    }
    /* Clean message boxes */
    .error-box, .success-box, .info-box, .warning-box {
        padding: 1.2rem;
        border-radius: 12px;
        margin: 1rem 0;
        font-weight: 500;
        border-left: 4px solid;
        background-color: white;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    .error-box {
        border-left-color: #ef4444;
        color: #dc2626;
    }
    .success-box {
        border-left-color: #10b981;
        color: #059669;
    }
    .info-box {
        border-left-color: #3b82f6;
        color: #2563eb;
    }
    .warning-box {
        border-left-color: #f59e0b;
        color: #d97706;
    }
    /* Section headers */
    .section-header {
        color: #1e293b !important;
        font-size: 1.5rem;
        font-weight: 700 !important;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e2e8f0;
    }
    /* Clean dataframe styling */
    .dataframe {
        font-size: 14px;
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        background: white;
    }
    .dataframe th {
        background-color: #f8fafc;
        color: #334155;
        font-weight: 600;
        padding: 12px 10px;
        text-align: left;
    }
    .dataframe td {
        padding: 10px;
        border-bottom: 1px solid #f1f5f9;
    }
    .dataframe tr:hover {
        background-color: #f8fafc;
    }
    /* Sidebar enhancements */
    .css-1d391kg {
        background: #f8fafc;
    }
    /* Custom expander styling */
    .streamlit-expanderHeader {
        background: #f1f5f9;
        border-radius: 8px;
        font-weight: 600;
        color: #334155;
        padding: 10px 15px;
        transition: all 0.2s ease;
    }
    .streamlit-expanderHeader:hover {
        background: #e2e8f0;
    }
    /* Progress bar */
    .stProgress > div > div > div > div {
        background-color: #4361ee;
    }
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem 0 1rem;
        color: #64748b;
        font-size: 0.9rem;
        border-top: 1px solid #e2e8f0;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Main Header
st.markdown("""
<div class="main-header">
    <h1>Moments Calculator</h1>
    <p>Calculate statistical moments for discrete and continuous random variables</p>
</div>
""", unsafe_allow_html=True)

class EnhancedProbabilityValidator:
    @staticmethod
    def validate_drv_probabilities(func_str: str, range_values: List[float]) -> Tuple[bool, str, float]:
        """Validate discrete probability mass function"""
        try:
            safe_dict = {
                'x': 0, 'factorial': np.math.factorial, 'sqrt': np.sqrt,
                'exp': np.exp, 'log': np.log, 'sin': np.sin, 'cos': np.cos,
                'tan': np.tan, 'pi': np.pi, 'e': np.e
            }
            negative_probs = []
            total_prob = 0
            
            for x_val in range_values:
                safe_dict['x'] = x_val
                prob = eval(func_str, {"__builtins__": {}}, safe_dict)
                if prob < 0:
                    negative_probs.append(x_val)
                total_prob += prob

            # Validation results
            if negative_probs:
                return False, f"Negative probabilities detected at x = {negative_probs[:3]}" + ("..." if len(negative_probs) > 3 else ""), total_prob
            
            if abs(total_prob - 1.0) > 1e-10:
                return False, f"Probabilities sum to {total_prob:.6f}, should be 1.0", total_prob
            
            return True, f"Valid probability function (sum = {total_prob:.6f})", total_prob
            
        except Exception as e:
            return False, f"Error evaluating function: {str(e)}", 0

    @staticmethod
    def validate_crv_pdf(func_str: str, range_bounds: Tuple[float, float]) -> Tuple[bool, str, float]:
        """Validate continuous probability density function"""
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
            
            # Check for negative values at sample points
            if np.isinf(lower) and np.isinf(upper):
                test_points = np.linspace(-10, 10, 50)
            elif np.isinf(lower):
                test_points = np.linspace(upper-10, upper, 25)
            elif np.isinf(upper):
                test_points = np.linspace(lower, lower+10, 25)
            else:
                test_points = np.linspace(lower, upper, 50)

            for point in test_points:
                try:
                    val = eval(func_str, {"__builtins__": {}}, {**safe_dict, 'x': point})
                    if val < -1e-10:
                        return False, f"Negative PDF value {val:.6f} at x = {point:.3f}", 0
                except:
                    continue

            # Numerical integration
            try:
                integral_result, error = integrate.quad(
                    pdf_func, lower, upper,
                    limit=200, epsabs=1e-10, epsrel=1e-10
                )
            except:
                integral_result, error = integrate.quad(pdf_func, lower, upper, limit=50)

            if abs(integral_result - 1.0) > 1e-3:
                return False, f"PDF integrates to {integral_result:.6f} ± {error:.2e}, not 1.0", integral_result

            return True, f"Valid PDF (integral = {integral_result:.6f} ± {error:.2e})", integral_result

        except Exception as e:
            return False, f"Error evaluating PDF: {str(e)}", 0

class EnhancedMomentCalculator:
    @staticmethod
    def calculate_drv_moment(func_str: str, range_values: List[float], r: int, a: float) -> float:
        """Calculate r-th moment for discrete random variable"""
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
            
        return moment

    @staticmethod
    def calculate_crv_moment(func_str: str, range_bounds: Tuple[float, float], r: int, a: float) -> float:
        """Calculate r-th moment for continuous random variable"""
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
                limit=200, epsabs=1e-12, epsrel=1e-12
            )
            return moment
        except:
            moment, _ = integrate.quad(integrand, range_bounds[0], range_bounds[1], limit=50)
            return moment

def parse_range_input(range_input: str) -> List[float]:
    """Parse discrete range input"""
    range_input = range_input.strip()
    values = [float(x.strip()) for x in range_input.split(',')]
    return sorted(values)

def parse_continuous_bound(bound_str: str) -> float:
    """Parse continuous bound"""
    bound_str = bound_str.strip().lower()
    infinity_variants = ['inf', 'infinity', '∞', '+inf', '+infinity', '+∞']
    neg_infinity_variants = ['-inf', '-infinity', '-∞']
    
    if bound_str in infinity_variants:
        return np.inf
    elif bound_str in neg_infinity_variants:
        return -np.inf
    else:
        return float(bound_str)

def create_visualization(var_type: str, func_str: str, range_data):
    """Create visualizations for the probability functions"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    plt.style.use('seaborn-v0_8-whitegrid')
    colors = ['#4361ee', '#3a0ca3']
    
    safe_dict = {
        'x': 0, 'factorial': np.math.factorial, 'sqrt': np.sqrt,
        'exp': np.exp, 'log': np.log, 'sin': np.sin, 'cos': np.cos,
        'tan': np.tan, 'pi': np.pi, 'e': np.e
    }
    
    try:
        if var_type == "Discrete (DRV)":
            range_values = range_data
            x_vals = range_values
            y_vals = []
            
            for x_val in x_vals:
                safe_dict['x'] = x_val
                try:
                    prob = eval(func_str, {"__builtins__": {}}, safe_dict)
                    y_vals.append(max(0, prob))
                except:
                    y_vals.append(0)
            
            # PMF plot
            ax1.bar(x_vals, y_vals, alpha=0.8, color=colors[0], edgecolor='white', linewidth=1.2)
            ax1.set_title('Probability Mass Function (PMF)', fontsize=13, fontweight='bold')
            ax1.set_xlabel('x', fontsize=11)
            ax1.set_ylabel('P(X = x)', fontsize=11)
            ax1.grid(True, alpha=0.3)
            
            # CDF plot
            cdf_vals = np.cumsum(y_vals)
            ax2.step(x_vals, cdf_vals, where='post', color=colors[1], linewidth=2.5)
            ax2.fill_between(x_vals, cdf_vals, alpha=0.2, color=colors[1], step='post')
            ax2.set_title('Cumulative Distribution Function (CDF)', fontsize=13, fontweight='bold')
            ax2.set_xlabel('x', fontsize=11)
            ax2.set_ylabel('F(x)', fontsize=11)
            ax2.grid(True, alpha=0.3)
            
        else:  # Continuous
            lower, upper = range_data
            plot_lower = max(lower, -10) if np.isinf(lower) else lower
            plot_upper = min(upper, 10) if np.isinf(upper) else upper
            x_vals = np.linspace(plot_lower, plot_upper, 500)
            y_vals = []
            
            for x_val in x_vals:
                safe_dict['x'] = x_val
                try:
                    pdf_val = eval(func_str, {"__builtins__": {}}, safe_dict)
                    y_vals.append(max(0, pdf_val))
                except:
                    y_vals.append(0)
            
            y_vals = np.array(y_vals)
            
            # PDF plot
            ax1.plot(x_vals, y_vals, color=colors[0], linewidth=3, alpha=0.8)
            ax1.fill_between(x_vals, y_vals, alpha=0.2, color=colors[0])
            ax1.set_title('Probability Density Function (PDF)', fontsize=13, fontweight='bold')
            ax1.set_xlabel('x', fontsize=11)
            ax1.set_ylabel('f(x)', fontsize=11)
            ax1.grid(True, alpha=0.3)
            
            # CDF plot (approximate)
            dx = x_vals[1] - x_vals[0]
            cdf_vals = np.cumsum(y_vals) * dx
            ax2.plot(x_vals, cdf_vals, color=colors[1], linewidth=3, alpha=0.8)
            ax2.fill_between(x_vals, cdf_vals, alpha=0.2, color=colors[1])
            ax2.set_title('Cumulative Distribution Function (CDF)', fontsize=13, fontweight='bold')
            ax2.set_xlabel('x', fontsize=11)
            ax2.set_ylabel('F(x)', fontsize=11)
            ax2.grid(True, alpha=0.3)
            
            if np.isinf(lower) or np.isinf(upper):
                bound_desc = f"[{'-∞' if np.isinf(lower) else lower:.1f}, {'∞' if np.isinf(upper) else upper:.1f}]"
                ax1.text(0.02, 0.95, f'Domain: {bound_desc}',
                        transform=ax1.transAxes, ha='left', va='top',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7),
                        fontsize=9)
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        # Fallback: create empty plots with error message
        ax1.text(0.5, 0.5, f'Visualization Error:\n{str(e)}',
                transform=ax1.transAxes, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='red', alpha=0.3))
        ax2.text(0.5, 0.5, 'Unable to create\nvisualization',
                transform=ax2.transAxes, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='red', alpha=0.3))
        plt.tight_layout()
        return fig

# Sidebar Configuration
with st.sidebar:
    st.markdown('<div class="section-header">⚙️ Configuration</div>', unsafe_allow_html=True)
    
    # Step 1: Variable type
    var_type = st.selectbox(
        "Choose Variable Type",
        ["Discrete (DRV)", "Continuous (CRV)"],
        help="🎲 Discrete: Countable values (coins, dice, counts)\n📊 Continuous: Any value in an interval (height, weight, time)"
    )
    
    st.markdown("---")
    
    # Step 2: Range input
    st.markdown('<div class="section-header">📊 Define Range</div>', unsafe_allow_html=True)
    
    if var_type == "Discrete (DRV)":
        st.markdown("""
        <div class="info-box">
            <strong>Discrete Range Examples:</strong><br>
            • <code>0,1,2,3,4,5</code><br>
            • <code>1,2,3,4,5,6</code> (for a die)<br>
            • <code>0,1,2,3,4</code> (for binomial n=4)
        </div>
        """, unsafe_allow_html=True)
        range_input = st.text_input(
            "Range Values (comma-separated)",
            value="0,1,2,3,4,5",
            help="Enter discrete values separated by commas",
            key="range_discrete"
        )
    else:
        st.markdown("""
        <div class="info-box">
            <strong>Continuous Range Examples:</strong><br>
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
                help="Use '-inf' for -∞",
                key="lower_bound"
            )
        with col2:
            upper_bound_str = st.text_input(
                "Upper Bound",
                value="1",
                help="Use 'inf' for ∞",
                key="upper_bound"
            )
    
    st.markdown("---")
    
    # Step 3: Probability function
    st.markdown('<div class="section-header">⚡ Probability Function</div>', unsafe_allow_html=True)
    
    # Function examples dropdown
    if var_type == "Discrete (DRV)":
        example_functions = {
            "Custom": "",
            "Uniform (6-sided die)": "1/6 if 1 <= x <= 6 else 0",
            "Binomial (n=5, p=0.3)": "(factorial(5)/(factorial(x)*factorial(5-x))) * (0.3**x) * (0.7**(5-x)) if 0 <= x <= 5 else 0",
            "Custom Finite": "0.2 if 0 <= x <= 4 else 0"
        }
    else:
        example_functions = {
            "Custom": "",
            "Uniform [0,1]": "1 if 0 <= x <= 1 else 0",
            "Exponential (λ=2)": "2*exp(-2*x) if x >= 0 else 0",
            "Beta (α=2, β=3)": "12*x*(1-x)**2 if 0 <= x <= 1 else 0",
            "Standard Normal": "exp(-x**2/2) / sqrt(2*pi)"
        }
    
    selected_example = st.selectbox(
        "Choose Example or Custom",
        list(example_functions.keys()),
        key="example_selector"
    )
    
    default_func = example_functions[selected_example]
    
    if var_type == "Discrete (DRV)":
        prob_func = st.text_area(
            "Probability Mass Function P(X=x)",
            value=default_func if default_func else "1/6 if 1 <= x <= 6 else 0",
            help="Available: factorial, sqrt, exp, log, sin, cos, tan, pi, e",
            height=100,
            key="prob_func_discrete"
        )
    else:
        prob_func = st.text_area(
            "Probability Density Function f(x)",
            value=default_func if default_func else "1 if 0 <= x <= 1 else 0",
            help="Available: sqrt, exp, log, sin, cos, tan, pi, e",
            height=100,
            key="prob_func_continuous"
        )
    
    st.markdown("---")
    
    # Step 4: Moment calculation options
    st.markdown('<div class="section-header">🎯 Moment Settings</div>', unsafe_allow_html=True)
    
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
            help="Point around which to calculate moments"
        )
    
    # Add moment order input
    max_moment_order = st.number_input(
        "Maximum Moment Order (r)",
        min_value=1,
        max_value=20,
        value=4,
        step=1,
        help="Calculate moments from 1st up to this order (e.g., 5 for 1st to 5th moment)",
        key="max_moment_order"
    )
    
    # Advanced options
    with st.expander("🖼️ Visualization Options", expanded=True):
        show_visualization = st.checkbox(
            "Show probability plots",
            value=True,
            help="Generate PDF/PMF and CDF visualizations"
        )

# Main content area
col1, col2 = st.columns([3, 2])

with col1:
    st.markdown('<div class="section-header">🎲 Analysis Results</div>', unsafe_allow_html=True)
    
    # Process inputs and validate
    try:
        # Parse range
        if var_type == "Discrete (DRV)":
            range_values = parse_range_input(range_input)
            # Validation
            is_valid, message, prob_sum = EnhancedProbabilityValidator.validate_drv_probabilities(prob_func, range_values)
        else:
            lower_bound = parse_continuous_bound(lower_bound_str)
            upper_bound = parse_continuous_bound(upper_bound_str)
            
            if not np.isinf(lower_bound) and not np.isinf(upper_bound) and lower_bound >= upper_bound:
                raise ValueError("Lower bound must be less than upper bound")
                
            range_bounds = (lower_bound, upper_bound)
            
            # Validate PDF
            is_valid, message, integral_val = EnhancedProbabilityValidator.validate_crv_pdf(prob_func, range_bounds)
        
        # Display validation results
        if is_valid:
            st.markdown(f"""
            <div class="success-box">
                <strong>✅ Validation Successful:</strong> {message}
            </div>
            """, unsafe_allow_html=True)
            
            # Calculate reference point for moments
            if moment_about == "About the mean (a = μ)":
                with st.spinner("🧮 Computing mean..."):
                    if var_type == "Discrete (DRV)":
                        mean_val = EnhancedMomentCalculator.calculate_drv_moment(prob_func, range_values, 1, 0)
                    else:
                        mean_val = EnhancedMomentCalculator.calculate_crv_moment(prob_func, range_bounds, 1, 0)
                a_value = mean_val
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-label">Computed Mean (μ)</div>
                    <div class="metric-value">{mean_val:.6f}</div>
                </div>
                """, unsafe_allow_html=True)
            elif moment_about == "About the origin (a = 0)":
                a_value = 0.0
            else:
                a_value = custom_a
            
            # Create list of moment orders
            moment_orders = list(range(1, int(max_moment_order) + 1))
            total_moments = len(moment_orders)
            
            # Calculate multiple moments with progress indication
            st.markdown("#### 🧮 Computing Moments...")
            progress_bar = st.progress(0)
            moments = {}
            
            for i, r in enumerate(moment_orders):
                progress_bar.progress((i + 1) / total_moments)
                with st.spinner(f"Computing {r}-th moment..."):
                    if var_type == "Discrete (DRV)":
                        moment_val = EnhancedMomentCalculator.calculate_drv_moment(prob_func, range_values, r, a_value)
                    else:
                        moment_val = EnhancedMomentCalculator.calculate_crv_moment(prob_func, range_bounds, r, a_value)
                moments[r] = moment_val
            
            progress_bar.empty()
            
            # Display moments
            st.markdown("#### 📈 Calculated Moments")
            
            # Create rows of 4 metrics each
            for i in range(0, len(moment_orders), 4):
                cols = st.columns(min(4, len(moment_orders) - i))
                for j in range(len(cols)):
                    r = moment_orders[i + j]
                    moment_val = moments[r]
                    with cols[j]:
                        st.markdown(f"""
                        <div class="metric-container">
                            <div class="metric-label">μ<sub>{r}</sub>({a_value:.2f})</div>
                            <div class="metric-value">{moment_val:.6f}</div>
                        </div>
                        """, unsafe_allow_html=True)
            
            # Statistical measures for central moments (only if we have enough moments)
            if moment_about == "About the mean (a = μ)" and len(moments) >= 4 and all(r in moments for r in [2, 3, 4]):
                st.markdown("#### 📊 Statistical Measures")
                variance = moments[2]
                std_dev = np.sqrt(abs(variance))
                
                if std_dev > 1e-10:
                    skewness = moments[3] / (std_dev ** 3)
                    kurtosis = moments[4] / (std_dev ** 4)
                    excess_kurtosis = kurtosis - 3
                else:
                    skewness = np.nan
                    kurtosis = np.nan
                    excess_kurtosis = np.nan
                
                # Create metric cards for statistical measures
                stat_cols = st.columns(5)
                statistical_measures = [
                    ("Mean (μ)", a_value, "#4361ee"),
                    ("Variance (σ²)", variance, "#3a0ca3"),
                    ("Std Dev (σ)", std_dev, "#4cc9f0"),
                    ("Skewness", skewness, "#f72585"),
                    ("Excess Kurtosis", excess_kurtosis, "#7209b7")
                ]
                
                for i, (label, value, color) in enumerate(statistical_measures):
                    with stat_cols[i]:
                        if not np.isnan(value):
                            st.markdown(f"""
                            <div class="metric-container" style="border-color: {color};">
                                <div class="metric-label">{label}</div>
                                <div class="metric-value">{value:.6f}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="metric-container" style="border-color: #94a3b8;">
                                <div class="metric-label">{label}</div>
                                <div class="metric-value">N/A</div>
                            </div>
                            """, unsafe_allow_html=True)
            
            # Detailed results table
            st.markdown("#### 📋 Detailed Results Table")
            results_data = []
            for r, moment_val in moments.items():
                row = {
                    'Moment Order (r)': r,
                    'Moment Value': f"{moment_val:.12f}",
                    'About Point (a)': f"{a_value:.6f}",
                    'Interpretation': f"E[(X-{a_value:.2f})^{r}]"
                }
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
                    Current probability sum: <code>{prob_sum:.8f}</code><br>
                    Expected sum: <code>1.0</code><br>
                    Difference: <code>{abs(prob_sum - 1.0):.8f}</code>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="warning-box">
                    <strong>🔧 Debug Information:</strong><br>
                    Current integral value: <code>{integral_val:.8f}</code><br>
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
    st.markdown('<div class="section-header">📊 Visualizations & Help</div>', unsafe_allow_html=True)
    
    # Show visualizations if validation passed and option is enabled
    if 'is_valid' in locals() and is_valid and show_visualization:
        try:
            with st.spinner("🎨 Creating visualizations..."):
                if var_type == "Discrete (DRV)":
                    fig = create_visualization(var_type, prob_func, range_values)
                else:
                    fig = create_visualization(var_type, prob_func, range_bounds)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)  # Clean up memory
        except Exception as e:
            st.markdown(f"""
            <div class="warning-box">
                <strong>⚠️ Visualization Error:</strong><br>
                {str(e)}<br>
                <small>Visualization is optional and doesn't affect calculations</small>
            </div>
            """, unsafe_allow_html=True)
    
    # Help sections
    with st.expander("🎲 Discrete Examples", expanded=False):
        st.markdown("""
        **Discrete Probability Examples:**
        ```python
        # Uniform (fair 6-sided die)
        Range: 1,2,3,4,5,6
        PMF: 1/6 if 1 <= x <= 6 else 0
        ```
        ```python
        # Binomial (n=5, p=0.3)
        Range: 0,1,2,3,4,5
        PMF: (factorial(5)/(factorial(x)*factorial(5-x))) * (0.3**x) * (0.7**(5-x)) if 0 <= x <= 5 else 0
        ```
        ```python
        # Custom distribution
        Range: 0,1,2,3,4
        PMF: 0.2 if 0 <= x <= 4 else 0
        ```
        """)
    
    with st.expander("📊 Continuous Examples", expanded=False):
        st.markdown("""
        **Continuous Probability Examples:**
        ```python
        # Uniform distribution [0,1]
        Range: [0, 1]
        PDF: 1 if 0 <= x <= 1 else 0
        ```
        ```python
        # Exponential distribution (λ=2)
        Range: [0, inf]
        PDF: 2*exp(-2*x) if x >= 0 else 0
        ```
        ```python
        # Standard Normal distribution
        Range: [-inf, inf]
        PDF: exp(-x**2/2) / sqrt(2*pi)
        ```
        ```python
        # Beta distribution (α=2, β=3)
        Range: [0, 1]
        PDF: 12*x*(1-x)**2 if 0 <= x <= 1 else 0
        ```
        """)
    
    with st.expander("🧮 Mathematical Functions", expanded=False):
        st.markdown("""
        **Available Functions:**
        *Basic Math:*
        - `+`, `-`, `*`, `/`, `**` (power)
        - `sqrt(x)` - Square root
        - `abs(x)` - Absolute value
        *Exponential & Logarithmic:*
        - `exp(x)` - e^x
        - `log(x)` - Natural log (ln)
        *Trigonometric:*
        - `sin(x)`, `cos(x)`, `tan(x)`
        *Special Functions:*
        - `factorial(n)` - n! (discrete only)
        *Constants:*
        - `pi` - π ≈ 3.14159
        - `e` - Euler's number ≈ 2.71828
        """)
    
    with st.expander("🎯 Moment Interpretation", expanded=False):
        st.markdown("""
        **Moment Meanings:**
        *Raw Moments (about origin):*
        - **μ₁(0)** = Mean (E[X])
        - **μ₂(0)** = Second moment (E[X²])
        - **μ₃(0)** = Third moment (E[X³])
        - **μ₄(0)** = Fourth moment (E[X⁴])
        *Central Moments (about mean):*
        - **μ₁(μ)** = 0 (always)
        - **μ₂(μ)** = Variance (σ²)
        - **μ₃(μ)** = Used for skewness
        - **μ₄(μ)** = Used for kurtosis
        *Statistical Measures:*
        - **Skewness** = μ₃/σ³ (asymmetry)
        - **Kurtosis** = μ₄/σ⁴ (tail heaviness)
        - **Excess Kurtosis** = Kurtosis - 3
        """)

# Footer
st.markdown("""
<div class="footer">
    <p>Moments Calculator | Discrete and Continuous Random Variables</p>
    <p>For educational purposes. Always verify mathematical validity of inputs.</p>
</div>
""", unsafe_allow_html=True)
