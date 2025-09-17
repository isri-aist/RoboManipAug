# How to contribute
Set up pre-commit to automatically perform checks when you make git commits.

Install pre-commit hooks:
```console
$ sudo apt install pre-commit

$ # Go to the top directory of this repository
$ pre-commit install
```

To run hooks on staged files, use:
```console
$ pre-commit
```

To run hooks on all files, use:
```console
$ pre-commit run --all-files
```
