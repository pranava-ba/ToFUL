ToFUL - Tool for Uncertainty
============================

**A powerful interactive tool for computing statistical moments of probability distributions.**

ToFUL helps students, researchers, and data scientists compute and understand statistical moments
of both discrete and continuous random variables with real-time validation and beautiful visualizations.

.. image:: https://static.streamlit.io/badges/streamlit_badge_black_white.svg
   :target: https://toful1.streamlit.app/
   :alt: Try the Live App

.. image:: https://img.shields.io/badge/python-3.8+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python 3.8+

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
   :alt: MIT License

🚀 Quick Links
------------------

* `Live Web App <https://toful1.streamlit.app/>`_ - No installation needed!
* `GitHub Repository <https://github.com/pranava-ba/ToFUL>`_
* `Report Issues <https://github.com/pranava-ba/ToFUL/issues>`_

✨ Key Features
---------------

📊 **Discrete & Continuous Random Variables**
   Handle both DRVs and CRVs with appropriate mathematical tools

🎯 **Adjustable Precision**
   Control calculation precision up to 15 decimal places

✅ **Real-time Validation**
   Get immediate feedback on function syntax and probability validity

📈 **Comprehensive Moment Analysis**
   Calculate raw moments, central moments, variance, skewness, kurtosis, and more

🔬 **Convergence Analysis**
   Visual feedback for infinite series convergence

Quick Example
-------------

**Discrete: Geometric Distribution**

.. code-block:: python

   Range: 0,1,2,3,...
   PMF: 0.3 * (0.7 ** x) if x >= 0 else 0

**Continuous: Exponential Distribution**

.. code-block:: python

   Range: [0, inf]
   PDF: 2*exp(-2*x) if x >= 0 else 0

Documentation Contents
----------------------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   getting-started/installation
   getting-started/quickstart
   getting-started/web-app

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user-guide/discrete-rv
   user-guide/continuous-rv
   user-guide/mathematical-functions
   user-guide/precision-settings
   user-guide/interpreting-results

.. toctree::
   :maxdepth: 2
   :caption: Examples

   examples/discrete-distributions
   examples/continuous-distributions
   examples/advanced-examples

.. toctree::
   :maxdepth: 2
   :caption: Theory & Background

   theory/statistical-moments
   theory/raw-vs-central
   theory/skewness-kurtosis
   theory/convergence

.. toctree::
   :maxdepth: 2
   :caption: Troubleshooting

   troubleshooting/common-errors
   troubleshooting/debugging-functions
   troubleshooting/convergence-issues

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api-reference/modules

.. toctree::
   :maxdepth: 1
   :caption: Development

   contributing/index
   changelog

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
