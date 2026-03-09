# Copilot Instructions

This file provides guidelines for GitHub Copilot to assist in code generation.

File location: `.github/copilot-instructions.md`

## Repository Overview

Important folders:

Tech stack: Python, UV, TensorFlow

## Documentation Standards

- Technical, concise language - bullet points or tables
- Never use emoticons or emojis
- Code examples with proper syntax highlighting
- Focus on what, how, and why - skip motivational language
- Markdown format

## Git

- **Types:** feat, fix, docs, refactor, chore
- Example format for commit messages:
    ```
    type(scope): subject

    body
    ```
- **Example:** `feat(db): add pgsql <database-name>`

## Security

- **Never include:** Passwords, API keys, private keys, connection strings with credentials

- **Instead:** Reference Keeper/Azure Key Vault, use K8s Secrets, environment variables

## Python

- package manager default: uv 
- package manager arcpy (ArcGIS Pro): conda

### Python: UV environment setup
- pyproject.toml available: `uv sync`
- no pyproject.toml available: 

    ```bash
    # workspace root
    uv init <env-name>
    uv add package1 package2
    uv lock && uv sync
    ```
- Structure:
    ```
    <project-name>/
    ├── pyproject.toml
    ├── uv.lock
    └── README.md
    ```



### Python: Comments and Documentation

- Keep inline comments minimal and concise
- Do not remove existing comments
- Always include docstrings for functions and classes
- reStructuredText (reST) style docstrings (use :param and :return: tags)
- Example of a docstring:

    ```python
    def example_function(param1, param2):
        """
        Example function that demonstrates the use of docstrings.

        :param param1: An integer parameter.
        :param param2: A string parameter.
        :return: True if successful, False otherwise.
        """
        return True
    ```

### Python: Code Style

- PEP8
- line length: 88 characters
- **Print/log statements:** keep on single line (no wrapping), line length unlimited for readability
- Use double quotes for strings
- Use f-strings for string formatting (print statements, etc)
- Use `%` for string formatting in log messages
- Organize imports: standard library, third-party, local modules

### Python: Performance Considerations

- Avoid global variables
- Prefer generators over lists for large datasets
- Use context managers (with statements) for resource management
- Use lazy evaluation where possible

### Python: Function Design

- Use functional programming principles
- Do not uses classes unless explicitly requested
- Single responsibility principle for functions

### Python: Header templates

#### Python: Stand alone files

- Located in `/scripts`
- Header template for stand alone files:

  ```python
  # Name:           <name>.py
  # Authors:        <name1>, <name2>
  # Description:    <description>
  # Requirements:   <requirements>
  # - python env:   <conda env name or uv env name>
  # - arcgispro:    ArcGIS Pro v.<version> (if applicable)
  # - packages:     <list of special packages>
  ```

### Python: Modules and packages

- Located in `/src`
- __ini__.py does not need a header/module description.
- Header template of modules:

  ```python
  """Module description"""
  ```

#### Python: Stand alone notebooks

```markdown
# Title

**Authors**: name1, name2

**Confluence**: [Confluence title](https://miljodir.atlassian.net/wiki/spaces/)

**Description**: Description in 1-2 sentences or bullet points.

**Requirements:**

- python env:   conda env name or uv env name
- arcgispro:    ArcGIS Pro v.%version% (if applicable)
- packages:     list of special packages
```
