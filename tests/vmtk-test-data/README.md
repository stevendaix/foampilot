# VMTK-Test-Data

[![Build Status](https://travis-ci.org/vmtk/vmtk.svg?branch=master)](https://travis-ci.org/vmtk/vmtk)
[![Build status](https://ci.appveyor.com/api/projects/status/3u6nupu2r47qbesq/branch/master?svg=true)](https://ci.appveyor.com/project/rlizzo/vmtk-bh6uc/branch/master)

The Vascular Modeling Toolkit is an open source project where people from all
over the world can contribute their work, with everyone benefitting from
friendly help and advice, and kindly helping others in return. This is the test
data repository for the main project. If you are looking for VMTK proper, head
on over to the [VMTK Github Repository](https://github.com/vmtk/vmtk) or the
[VMTK Webpage](https://vmtk.org).

## How to help?

### Contributing to vmtk

If you want to contribute to vmtk please [fork](https://help.github.com/articles/fork-a-repo#contributing-to-a-project) the [source code](https://github.com/vmtk/vmtk)

See [github collaborating guide](https://help.github.com/categories/63/articles), and feel free to reach out to us on the [mailing list](https://groups.google.com/forum/#!forum/vmtk-users) or the issue tracker for any questions. 

For contributing instructions, check out the [VMTK Contributing Guide](https://github.com/vmtk/vmtk/blob/master/CONTRIBUTING.md)

## Testing

### Aquiring Test Data

VMTK utilizes a git submodule in order to orchestrate the aquisition of large
binary data files. If you have just simply cloned the vmtk repository, you will
find the tests data director (`tests/vmtk-test-data`) empty.

Setting the CMake variable `VMTK_BUILD_TESTING=ON` and building the project
will use this information to download the actual binary files to this directory
in the build tree. It will also automatically configure the paths to this data
directory referenced in the `tests` directory. 

### Running Tests

We use the [pytest](https://docs.pytest.org/en/latest/) testing framework for
unit testing. After building the library, run `pytest ./` from within the
`Install/tests` directory. 

### Contributing Tests

In order to contribute tests, can clone the
[vmtk-test-data](https://github.com/vmtk/vmtk-test-data) repository at the same
level as vmtk. Please add any test data files in the appropriate folders in the
`vmtk-test-data` repository. if you create new test files within the
`tests/test_FOO` directory, be sure to add that file to the accompanying
`CMakeLists.txt` file as well. When your tests run locally, your data assets
and test changes should be reflected and all tests should appear to pass. 

If you are contributing tests which require new data files, create a PR to the
[vmtk-test-data](https://github.com/vmtk/vmtk-test-data) repository and let us
know the PR# in VMTK which the data files corresponds to. Until we merge the
data, our CI suite will not have access to files, so your tests that passed
locally may appear to fail when they are pushed - Don't worry! It's not you,
it's our system! 

## Questions? Concerns?  If you have any questions about the contributing
process, or just want to learn more about the library, feel free to reach out
to us on the mailing list or the issue tracker. We'd love to chat! 
