SACO
====

Welcome to the Sustainable Abstraction Calculator and Optimiser (SACO) tool repository.

The SACO tool aims to help identify potential ways to meet environmental flow targets
in Water Framework Directive (WFD) surface waterbodies. It has been developed by the
Environment Agency (England).

Please note that development and testing is ongoing, so some aspects of behaviour and
functionality may change in the future.

Installation
------------

After cloning the repository (or downloading/unzipping a specific release), create a
new Python environment (e.g. a virtual environment or conda environment). Then in
a terminal navigate to the root of the repository and install the package with::

    pip install -e .

Note that:

    - The environment.yml file in the repository root can be optionally used to create
      a conda environment with the same dependency versions used in initial development
      and testing.
    - The ``-e`` flag is optional (for installing the package in editable/developer
      mode).

Documentation
-------------

Package `documentation`_ is available that explains the core features of the tool and
how it might be used. The documentation includes a tutorial page, as well as a
reference for the data structure and the main classes/functions in the package.

Examples
--------

Jupyter notebook examples are available in the examples folder of the repository. It is
worth looking at the documentation before/alongside the code examples.

.. _documentation: https://environment-agency-gov.github.io/saco-core/

Contributions
-------------

Contributions to the project (bug reports/fixes, code/documentation improvements, new
features, etc) are welcome. The best starting point might be to discuss any possible
contributions by raising a new issue or commenting on an existing issue. For larger
potential functionality additions or changes, it would be advisable to discuss them
at an early stage. This is because the tool is closely based around a specific
underlying conceptual model and dataset.

Licence and Disclaimer
----------------------

The SACO tool is released under the `Open Government Licence (OGL)`_ and is intended to
be developed in an open and collaborative manner. All contributions to development are
deemed to be licensed under the same terms.

The SACO tool is provided “as is”, without warranty of any kind, express or implied,
including but not limited to warranties of merchantability or fitness for a particular
purpose. Use of the SACO tool is at your own risk. The Environment Agency accepts no
liability for any loss, damage, or claims arising from its use.

The SACO tool may be updated, modified, or replaced at any time. Older versions may
cease to be available. Users who already hold a version may continue to use it under
the licence, but there is no guarantee that future versions will retain compatibility,
features, or functionality. Users should check for updated versions.

The release of the SACO tool does not imply any obligation on the part of the
Environment Agency to provide maintenance, support, or continued availability.

© Environment Agency copyright 2025.

.. _Open Government Licence (OGL): https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/
