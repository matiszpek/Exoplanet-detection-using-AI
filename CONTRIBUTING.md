# Contributing to Exoplanet Detection using AI

Thank you for your interest in contributing to our NASA Space Apps Challenge project! This document provides guidelines for contributing to the exoplanet detection system.

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Git for version control
- Jupyter Notebook environment
- Basic knowledge of machine learning and astronomy

### Setting Up Development Environment

1. **Fork and Clone the Repository**
```bash
git clone https://github.com/your-username/Exoplanet-detection-using-AI.git
cd Exoplanet-detection-using-AI
```

2. **Create Virtual Environment**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Install Development Dependencies**
```bash
pip install pytest black flake8 sphinx
```

## 🛠️ Development Guidelines

### Code Style

We follow PEP 8 guidelines with some modifications:

- **Line length**: 88 characters (Black default)
- **String quotes**: Use double quotes for strings
- **Imports**: Organize imports in the following order:
  1. Standard library imports
  2. Third-party imports
  3. Local application imports

### Code Formatting

Use Black for code formatting:
```bash
black src/ notebooks/
```

### Linting

Run flake8 for code linting:
```bash
flake8 src/ --max-line-length=88
```

### Documentation

- **Docstrings**: Use Google-style docstrings for all functions and classes
- **Comments**: Write clear, concise comments explaining complex logic
- **Notebooks**: Include markdown cells explaining each section

## 📝 Contributing Process

### 1. Choose an Issue

- Look at open issues in the GitHub repository
- Comment on the issue you'd like to work on
- Wait for assignment before starting work

### 2. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b bugfix/issue-description
```

### 3. Make Changes

- Follow the code style guidelines
- Add tests for new functionality
- Update documentation as needed
- Ensure notebooks run without errors

### 4. Testing

#### Unit Tests
```bash
pytest tests/
```

#### Notebook Testing
Ensure all notebooks can run from start to finish:
```bash
jupyter nbconvert --execute --to notebook notebooks/*.ipynb
```

### 5. Commit Changes

Use conventional commit messages:
```bash
git commit -m "feat: add new feature"
git commit -m "fix: resolve issue with data loading"
git commit -m "docs: update README"
```

### 6. Submit Pull Request

1. Push your branch to your fork
2. Create a pull request with:
   - Clear title and description
   - Reference to related issues
   - Screenshots/plots if applicable
   - Test results

## 🧪 Types of Contributions

### Code Contributions

- **New features**: Machine learning models, data preprocessing tools
- **Bug fixes**: Error corrections, performance improvements
- **Optimization**: Code efficiency, memory usage improvements

### Documentation

- **README updates**: Installation instructions, usage examples
- **Code documentation**: Docstrings, inline comments
- **Tutorials**: Jupyter notebooks, how-to guides

### Data Science

- **Feature engineering**: New derived features for exoplanet detection
- **Model improvements**: Better algorithms, hyperparameter tuning
- **Evaluation metrics**: New ways to assess model performance

### Visualization

- **Interactive plots**: Plotly, Bokeh visualizations
- **Dashboard creation**: Model performance dashboards
- **Data exploration**: New ways to visualize astronomical data

## 🔬 Scientific Contributions

### Research Areas

- **False positive reduction**: Better discrimination techniques
- **Multi-mission data**: Combining Kepler, TESS, and K2 data
- **Ensemble methods**: Advanced model combination strategies
- **Interpretability**: Understanding what models learn

### Data Quality

- **Data validation**: Checks for data integrity
- **Outlier detection**: Identifying problematic observations
- **Cross-validation**: Robust evaluation strategies

## 📊 Notebook Guidelines

### Structure

1. **Title and Introduction**: Clear explanation of notebook purpose
2. **Imports and Setup**: All necessary libraries and configurations
3. **Data Loading**: Clear data source and loading process
4. **Exploratory Analysis**: Data understanding and visualization
5. **Methodology**: Clear explanation of approach
6. **Results**: Model performance and evaluation
7. **Conclusions**: Key findings and next steps

### Best Practices

- **Clear markdown**: Explain each section thoroughly
- **Reproducible**: Set random seeds for consistency
- **Modular**: Break complex operations into functions
- **Visualizations**: Include relevant plots and charts
- **Error handling**: Robust code that handles edge cases

## 🚦 Review Process

### Code Review Criteria

- **Functionality**: Does the code work as intended?
- **Quality**: Is the code well-written and maintainable?
- **Testing**: Are there adequate tests?
- **Documentation**: Is the code well-documented?
- **Performance**: Are there any performance concerns?

### Scientific Review

- **Methodology**: Is the approach scientifically sound?
- **Validation**: Are results properly validated?
- **Interpretation**: Are conclusions supported by data?
- **Reproducibility**: Can results be reproduced?

## 🐛 Reporting Issues

### Bug Reports

Include:
- **Environment**: Python version, OS, dependencies
- **Steps to reproduce**: Clear reproduction steps
- **Expected behavior**: What should happen
- **Actual behavior**: What actually happens
- **Error messages**: Full error traceback if applicable

### Feature Requests

Include:
- **Use case**: Why is this feature needed?
- **Description**: What should the feature do?
- **Examples**: How would it be used?
- **Implementation ideas**: Suggestions for implementation

## 🎓 Learning Resources

### Astronomy and Exoplanets

- [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/)
- [Kepler Mission Documentation](https://www.nasa.gov/mission_pages/kepler/main/index.html)
- [TESS Mission Resources](https://tess.mit.edu/)

### Machine Learning

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Pandas Documentation](https://pandas.pydata.org/)
- [Astronomical Data Analysis with Python](https://astropy.org/)

### Best Practices

- [Python Code Quality](https://realpython.com/python-code-quality/)
- [Jupyter Notebook Best Practices](https://jupyter-notebook.readthedocs.io/)
- [Git Workflow](https://www.atlassian.com/git/tutorials/comparing-workflows)

## 📞 Communication

### Channels

- **GitHub Issues**: Technical discussions, bug reports
- **Pull Request Comments**: Code review discussions
- **Project Discussions**: General project questions

### Response Times

- **Issues**: We aim to respond within 2-3 days
- **Pull Requests**: Initial review within 1 week
- **Questions**: Community help usually within 24 hours

## 🏆 Recognition

Contributors will be:
- Listed in project acknowledgments
- Credited in scientific outputs
- Invited to present at project meetings
- Considered for collaboration opportunities

## 📜 Code of Conduct

### Our Standards

- **Respectful**: Treat all contributors with respect
- **Inclusive**: Welcome diverse perspectives and backgrounds
- **Collaborative**: Work together toward common goals
- **Professional**: Maintain professional standards in all interactions

### Enforcement

Violations of the code of conduct should be reported to project maintainers. We reserve the right to remove contributions or ban contributors who violate these standards.

---

Thank you for contributing to advancing exoplanet science! 🚀🌟