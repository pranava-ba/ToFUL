# ToFUL - Tool for Uncertainty

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://toful1.streamlit.app/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**A powerful interactive tool for computing statistical moments of probability distributions.**

Calculate raw moments, central moments, variance, skewness, kurtosis, and more for both discrete and continuous random variables with real-time validation and beautiful visualizations.

🚀 **[Try it now](https://toful1.streamlit.app/)** - No installation needed!

---

## Features

- 📊 **Discrete & Continuous RVs** - Handle both DRVs and CRVs
- 🎯 **Adjustable Precision** - Up to 15 decimal places
- ✅ **Real-time Validation** - Immediate syntax and probability checking
- 📈 **Comprehensive Analysis** - Full moment calculations with convergence analysis

## Quick Start

### Online (Recommended)
Visit **[toful1.streamlit.app](https://toful1.streamlit.app/)** - works on any device!

### Local Installation

```bash
# Clone repository
git clone https://github.com/pranava-ba/ToFUL.git
cd ToFUL

# Install dependencies
pip install streamlit numpy scipy pandas

# Run app
streamlit run app.py
```

## Usage Example

**Geometric Distribution (Discrete):**
```
Range: 0,1,2,3,...
PMF: 0.3 * (0.7 ** x) if x >= 0 else 0
```

**Exponential Distribution (Continuous):**
```
Range: [0, inf]
PDF: 2*exp(-2*x) if x >= 0 else 0
```
[![Documentation Status](https://readthedocs.org/projects/toful/badge/?version=latest)](https://toful.readthedocs.io/en/latest/?badge=latest)

## Contributing

Contributions welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) before submitting PRs.

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

⭐ **Star this repo** if you find it useful!  
🐛 **[Report issues](https://github.com/pranava-ba/ToFUL/issues)** or suggest features

Built with Streamlit, NumPy, SciPy, and ❤️
