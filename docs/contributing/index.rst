.. _contributing:

Contributing to ToFUL
=====================

Thank you for your interest in contributing to ToFUL! We welcome contributions from the community.

.. contents:: Table of Contents
   :local:
   :depth: 2

.. toctree::
   :maxdepth: 2
   :caption: Contributing Guide

   getting-started
   code-standards
   testing
   documentation
   pull-requests
   code-of-conduct

Quick Start for Contributors
-----------------------------

1. **Fork and Clone**

   .. code-block:: bash

      git clone https://github.com/YOUR_USERNAME/ToFUL.git
      cd ToFUL

2. **Set Up Environment**

   .. code-block:: bash

      python -m venv venv
      source venv/bin/activate  # On Windows: venv\Scripts\activate
      pip install -e ".[dev]"

3. **Create a Branch**

   .. code-block:: bash

      git checkout -b feature/your-feature-name

4. **Make Changes and Test**

   .. code-block:: bash

      # Make your changes
      pytest tests/
      black toful/
      flake8 toful/

5. **Submit Pull Request**

   Push your changes and create a pull request on GitHub.

Ways to Contribute
------------------

Code Contributions
^^^^^^^^^^^^^^^^^^

* Implement new distributions
* Add statistical tests
* Improve performance
* Fix bugs

Documentation
^^^^^^^^^^^^^

* Improve docstrings
* Add examples
* Write tutorials
* Fix typos

Testing
^^^^^^^

* Write unit tests
* Add integration tests
* Improve test coverage

Community
^^^^^^^^^

* Answer questions
* Review pull requests
* Report bugs
* Suggest features

Getting Help
------------

* **GitHub Discussions**: Ask questions and discuss ideas
* **GitHub Issues**: Report bugs and request features
* **Email**: Contact maintainers at toful@example.com

Recognition
-----------

All contributors are recognized in our :doc:`../changelog` and :doc:`contributors` page.

Thank you for helping make ToFUL better!
