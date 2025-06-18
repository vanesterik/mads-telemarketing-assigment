![Telemarketing Assignment](./references/morty-as-stock-broker.jpg)

# Telemarketing Assignment

## Master of Applied Data Science - HAN University

This repository contains my submission for the Predictive Modelling (2024 P3A) practice assignment, part of the Master of Applied Data Science at HAN University.

The project is a fictional machine learning case study using the [Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing) dataset from the [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/).

Within this repository, you will find raw, interim, and processed data, Jupyter notebooks, source code, and reports, all supporting my assignment submission.

## Project Structure

The project is organized as follows:

```plaintext
mads_telemarketing_assignment/
├── data/
│   ├── raw/               # Original dataset
│   ├── interim/           # Intermediate data files
│   └── processed/         # Final processed data
├── notebooks/             # Jupyter notebooks for analysis
├── src/                   # Source code for data processing and modeling
├── reports/               # Reports and documentation
└── README.md              # Project documentation
```

## Requirements

To run this project, you will need the Python packages listed in the `pyproject.toml` file. The main packages used are:

```plaintext
jupyter
matplotlib
numpy
pandas
scikit-learn
```

These packages are required for data manipulation, analysis, visualization, and running the Jupyter notebooks and source code in this project.

## Installation

If you have `make` installed, you can set up the required packages by running:

```bash
make install
```

This command will bootstrap [PDM](https://pdm.fming.dev/), install the dependencies in a virtual environment, and ensure all [Pandoc](https://pandoc.org/) dependencies are available for report generation.

To render the reports, run:

```bash
make reports
```

This will generate the reports and a presentation in the `reports/` directory, including the final report in PDF format.

## Usage

You can explore the data and models by running the Jupyter notebooks in the `notebooks/` directory. The `src/` directory contains functions for data processing and model training.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

This project uses the [Bank Telemarketing](https://archive.ics.uci.edu/dataset/222/bank+marketing) dataset from the [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/). Special thanks to the contributors and maintainers of this dataset.

## Contact

For any questions or feedback, please contact me at [kd.vanesterik@student.han.nl](mailto:kd.vanesterik@student.han.nl).

