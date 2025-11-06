# core.py
import numpy as np
from scipy import integrate
from typing import Tuple, Union, List
import math

# Unicode subscript mapping
SUBSCRIPT_MAP = {
    0: '₀', 1: '₁', 2: '₂', 3: '₃', 4: '₄',
    5: '₅', 6: '₆', 7: '₇', 8: '₈', 9: '₉'
}

def to_subscript(n):
    """Convert number to subscript string"""
    return ''.join(SUBSCRIPT_MAP.get(int(d), str(d)) for d in str(n))

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
            'x': 0, 'factorial': math.factorial, 'sqrt': np.sqrt,
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
        analysis = {"terms_computed": 0, "convergence_info": "", "series_type": "finite"}
        try:
            safe_dict = {
                'x': 0, 'factorial': math.factorial, 'sqrt': np.sqrt,
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
            'factorial': math.factorial, 'sqrt': np.sqrt,
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
            'factorial': math.factorial, 'sqrt': np.sqrt,
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
