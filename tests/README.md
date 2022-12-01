# Test Suite
In addition to unit tests built into the scripts in `../src/`, there are integration/end-to-end tests, which reside here. These tests will be named using the same name as the script they test, followed by '_test.py`.


## `inputs/`
Many of these tests require example input files. Those files reside in `inputs/`. Within `inputs/`, there are folders which correspond to the respective python script (minus the '.py' extension).

## `test_utils.py`
This file is not an integration test file, but rather a module containing functions that are reused by several tests.