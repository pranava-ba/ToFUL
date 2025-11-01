# 🎲 Moments Calculator

**Calculate statistical moments for discrete and continuous random variables with beautiful visualization and real-time validation.**

**Live App**: [https://toful1.streamlit.app/](https://toful1.streamlit.app/) ![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-green)

A powerful, interactive web application built with **Streamlit** that helps students, researchers, and data scientists compute and understand statistical moments of probability distributions.

---

## ✨ Features

- **Support for Discrete & Continuous RVs**  
  Handle both discrete random variables (DRV) and continuous random variables (CRV).

- **Adjustable Precision**  
  Control calculation precision (up to 15 decimal places) and display formatting independently.

- **Real-time Validation & Error Guidance**  
  Get immediate feedback on function syntax, probability sums, and mathematical validity.


- **Comprehensive Moment Analysis**  
  Compute raw moments, central moments, variance, skewness, kurtosis, and more.



---

## 🚀 Try It Live

No installation needed! Just visit:

### 👉 [https://toful1.streamlit.app/](https://toful1.streamlit.app/)

Works on desktop and mobile browsers.

---

## 📊 How to Use

### 1. Choose Variable Type
Select either **Discrete (DRV)** or **Continuous (CRV)**.

### 2. Define Range or Bounds
- **Discrete**: Enter comma-separated values (e.g., `1,2,3,4,5` or `0,1,2,3,...` for infinite series)
- **Continuous**: Enter lower and upper bounds (e.g., `0, 1` or `-inf, inf`)

### 3. Enter Probability Function
Write your probability function using available mathematical functions:
- `factorial(x)`, `sqrt(x)`, `exp(x)`, `log(x)`
- `sin(x)`, `cos(x)`, `tan(x)`
- Constants: `pi`, `e`

### 4. Select Moment Reference Point
Choose to calculate moments about:
- The origin (a = 0)
- The mean (a = μ)
- A custom value

### 5. Set Parameters
- Maximum moment order 
- Calculation and display precision

### 6. Click "Compute!"
View your results with convergence analysis and statistical measures.

---

## 🧮 Examples

### Discrete Examples

**Finite Uniform Distribution:**
```
Range: 1,2,3,4,5
PMF: 0.2 if 1 <= x <= 5 else 0
```

**Geometric Distribution:**
```
Range: 0,1,2,3,...
PMF: 0.3 * (0.7 ** x) if x >= 0 else 0
```

**Poisson Distribution:**
```
Range: 0,1,2,3,...
PMF: (exp(-3) * 3**x) / factorial(x) if x >= 0 else 0
```

### Continuous Examples

**Uniform Distribution:**
```
Range: [0, 1]
PDF: 1 if 0 <= x <= 1 else 0
```

**Exponential Distribution:**
```
Range: [0, inf]
PDF: 2*exp(-2*x) if x >= 0 else 0
```

**Normal Distribution:**
```
Range: [-inf, inf]
PDF: exp(-(x**2)/2) / sqrt(2*pi)
```

---

## 🛠️ Run Locally (For Developers)

### Prerequisites

- Python 3.8+
- pip (Python package manager)

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/moments-calculator.git
cd moments-calculator
```

2. **Create and activate a virtual environment (recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install streamlit numpy scipy pandas
```

### Running the App

```bash
streamlit run app.py
```

The app will automatically open in your default web browser at `http://localhost:8501`.

---

## 📁 Project Structure

```
moments-calculator/
├── app.py                 # Main application file
├── requirements.txt       # Dependencies (optional)
├── README.md              # This file
└── LICENSE                # MIT License
```

---

## 🛠️ Troubleshooting

### Common Issues & Solutions

**Q: I get "module 'numpy' has no attribute 'math'"**  
✅ **Fixed in deployed version.** Uses `math.factorial` instead of `np.math.factorial`.

**Q: The landing page doesn't render properly**  
✅ **Fixed.** Uses Streamlit-native rendering for maximum compatibility.

**Q: My probability function isn't working**  
✅ Check that:
- You're using allowed functions (factorial, sqrt, exp, etc.)
- Your syntax is correct (Python syntax)
- Probabilities sum to 1.0 (discrete) or PDF integrates to 1.0 (continuous)

**Q: Infinite series aren't converging**  
✅ The app provides convergence analysis. Try:
- Increasing the maximum terms (up to 500)
- Checking your function for mathematical errors
- Using the debug information provided

---

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a pull request

---

## 📄 License

This project is licensed under the **MIT License**.

```
MIT License

Copyright (c) 2024 [Your Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 📞 Support

If you encounter any issues or have questions, please [open an issue](https://github.com/yourusername/moments-calculator/issues) in this repository — or just enjoy the live app at:

> 🚀 [https://toful1.streamlit.app/](https://toful1.streamlit.app/)

---

⭐ **Star this repo if you find it useful!**  
🐞 **Report bugs or suggest features — your feedback helps improve the app!**

---

**Happy Calculating!** 🎯  
*Built with Streamlit, NumPy, SciPy, and ❤️*

---

### 💡 Pro Tip

Bookmark [https://toful1.streamlit.app/](https://toful1.streamlit.app/) for quick access anytime — no installation required!

---
## RTD
https://toful-rtd.readthedocs.io/en/latest/index.html
