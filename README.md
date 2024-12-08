# Float16 Serverless Examples

This repository contains a collection of examples that demonstrate how to use the Float16 Serverless platform.

This repository is a work in progress. We will be adding more examples over time. 

If you have a specific example you would like to see, please open an issue.

[Float16.cloud](https://float16.cloud)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Float16 CLI (NPM)](https://www.npmjs.com/package/@float16/cli)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Documentation](https://docs.float16.cloud)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Discord](https://discord.com/invite/j2DVTMjr67)&nbsp;&nbsp;&nbsp;


## Table of Contents

- [Latest News](#latest-news)
- [Latest Examples](#latest-examples)
- [Getting Started](#getting-started)
- [Examples](#examples)
- [Contribution Guidelines](#contribution-guidelines)
- [License](#license)
- [Contact us](#contact-us)

## Latest News

- **2024-12-09**: Initial release of the Float16 Serverless Examples repository.

## Latest Examples

- **2024-12-09**: Added [FastAPI Serving](./fastapi_helloworld) and [Helloworld](./helloWorld/) example.

## Getting Started

1. Install the Float16 CLI via NPM.

```
npm install -g @float16/cli 
```

2. Login to your Float16 account.

```
float16 login --token <your_token>
```

3. Start using the Float16 CLI with the examples.

```
float16 example <example_name>
```

4. Run the example.

```
float16 run app.py
```

## Examples

### List all examples
List all examples available in the repository.

```
float16 example list
```

### Clone an example

```
float16 example <example_dir>

float16 example helloworld
```

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
    # Example Name (Dev mode or Prod mode)

    ## Description (Must have)

    A brief description of the example.

    ## High Level Overview (Optional)

    A high level overview of the example.

    Diagrams are encouraged.

    ## Libraries (Must have)

    - Library 1
    - Library 2
    - Library 3

    ## GPU Configuration (Must have)

    - H100, L40s, L4 or etc.

    ## Expected Performance (Must have)

    - Performance
    - Latency
    - Throughput

    ## Profile (Optional)

    - Social media links
    - Personal website
    - Open for Work
    ```
5. Open a pull request
6. We will review your example and merge it into the repository

## License

This repository is licensed under the Apache License 2.0. See [LICENSE](./LICENSE) for more information.

## Contact us 

If you have any questions, please reach out to us on Discord or email us at

- X : [X](https://x.com/float16cloud)
- Discord: [Discord](https://discord.com/invite/j2DVTMjr67)
- Email: matichon[dot]man[at]float16[dot]cloud