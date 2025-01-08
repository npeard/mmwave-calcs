# mmWave Lab Physics Models

A repository of shared code for calculations of interest to mmWave Lab at Stanford University.

For a minimal working example of the format and function we are going for,
please see the "/models/gatefidelity.py", "/plotters/plot_gatefidelity.py",
and "/notebooks/GateErrors.ipynb" collection.

## Introduction for New Group Members
The idea is that all calculation-heavy code will be placed in the /models
directory. These are calculations that everyone will use; therefore, the
source code should be re-used and scrutinized often to check for factors of
$2\pi$, sign errors, etc. Use these models for larger calculations and plots,
and everyone will understand where the underlying code came from, there will
be fewer silly mistakes, and hence shorter discussions and meetings!

If you make a very complex plot, or multiple plots for the same model, you
should write a Python file that contains your plotting functions and
avoid cluttered Jupyter notebooks. These files should be named with the
prefix "plot_X.py" and placed in the /plotters directory.

Finally, your stunning Jupyter notebook can go in the /notebooks
directory. The notebook should call functions and classes that are
implemented either in /plotters or in /models to benefit from the shared
(error-checked) code base. These notebooks can be individualized to
your specific use case, but they should still be synced to the Github repo.

You can edit the Python code in the repo and your Jupyter notebook
simultaneously. Restart your Jupyter kernel to reload any Python code you
edit in your Python IDE.

## Quick Start for Contributors

1. Clone the repository:
   ```bash
   git clone https://github.com/username/your-project.git
   cd your-project
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

4. Install pre-commit hooks:
   ```bash
   pip install pre-commit nbstripout
   pre-commit install
   ```

5. Create a new branch for your feature:
   ```bash
   git checkout -b feature-name
   ```

6. Make your changes and run tests:
   ```bash
   pytest tests/
   ```

7. Commit and push your changes:
   ```bash
   git add .
   git commit -m "Description of changes"
   git push origin feature-name
   ```

8. Open a Pull Request on GitHub
