Quick Start Guide
=================

Get up and running with ToFUL in 5 minutes!

Using the Web App
-----------------

The easiest way to use ToFUL is through our hosted web application:

1. Visit https://toful1.streamlit.app/
2. Select your distribution type (Discrete or Continuous)
3. Enter your parameters
4. View instant results!

Step-by-Step Example
---------------------

Let's calculate moments for a **Geometric Distribution**.

Step 1: Select Distribution Type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Click the **"Discrete (DRV)"** button at the top of the page.

Step 2: Enter Range
~~~~~~~~~~~~~~~~~~~

In the range input field, enter:

.. code-block:: text

   0,1,2,3,...

This defines an infinite series starting from 0.

Step 3: Define Your PMF
~~~~~~~~~~~~~~~~~~~~~~~

In the probability mass function field, enter:

.. code-block:: python

   0.3 * (0.7 ** x) if x >= 0 else 0

This represents a geometric distribution with parameter p = 0.3.

Step 4: Configure Settings (Optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **About**: Leave as "Origin (a = 0)" for raw moments
* **Maximum Order**: Set to 4 for moments up to the 4th order
* **Precision**: Keep default or adjust as needed

Step 5: Calculate
~~~~~~~~~~~~~~~~~

Click the **"Calculate Moments"** button and view your results!

Understanding the Output
-------------------------

The app displays:

**Raw Moments (μ'ₙ)**
   Moments about the origin: E[Xⁿ]

**Central Moments (μₙ)**
   Moments about the mean: E[(X - μ)ⁿ]

**Key Statistics**
   * **Mean (μ)**: First raw moment
   * **Variance (σ²)**: Second central moment
   * **Skewness**: Normalized third central moment
   * **Kurtosis**: Normalized fourth central moment

**Convergence Analysis**
   For infinite series, shows whether the calculations converged

Try More Examples
-----------------

**Poisson Distribution**

.. code-block:: text

   Range: 0,1,2,3,...
   PMF: (exp(-3) * 3**x) / factorial(x) if x >= 0 else 0

**Uniform Distribution (Continuous)**

.. code-block:: text

   Range: [0, 1]
   PDF: 1 if 0 <= x <= 1 else 0

Next Steps
----------

* Explore :doc:`/examples/discrete-distributions` for more examples
* Learn about :doc:`/user-guide/mathematical-functions` available
* Read :doc:`/theory/statistical-moments` to understand the math
