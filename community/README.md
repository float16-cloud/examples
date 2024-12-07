## Contribution Guidelines

We welcome contributions to this repository. If you have an example you would like to add, please open a pull request.

### How to contribute

1. Fork the repository
2. Create a new branch
3. Add your example under the `community/<your_name>/<example_name>` directory
4. New examples must include 4 files at a minimum:
    - `app.py` - The main application file for execute
    - `README.md` - A description of the example
    - `requirements.txt` - A list of dependencies required to run the example
    - `float16.conf` - A configuration file for the example

    For your utils or other files, you can add them in the same directory.

    `README.md` template:
    
    ```

    # Example Name

    ## Description (Must)

    A brief description of the example.

    ## High Level Overview (Optional)

    A high level overview of the example.

    Diagrams are encouraged.

    ## Libraries (Must)

    - Library 1
    - Library 2
    - Library 3

    ## GPU Configuration (Must)

    - H100, L40s, L4 or etc.

    ## Expected Performance (Must)

    - Performance
    - Latency
    - Throughput

    ## Your profile (Optional)

    - Social media links
    - Personal website

    ```
