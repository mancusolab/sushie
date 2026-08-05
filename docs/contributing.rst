============
Contributing
============

Welcome to ``sushie`` contributor's guide.

This document focuses on getting any potential contributor familiarized
with the development processes, but `other kinds of contributions`_ are also
appreciated.

If you are new to using git_ or have never collaborated in a project previously,
please have a look at `contribution-guide.org`_. Other resources are also
listed in the excellent `guide created by FreeCodeCamp`_ [#contrib1]_.

Please notice, all users and contributors are expected to be **open,
considerate, reasonable, and respectful**. When in doubt, `Python Software
Foundation's Code of Conduct`_ is a good reference in terms of behavior
guidelines.


Issue Reports
=============

If you experience bugs or general issues with ``sushie``, please have a look
on the `issue tracker`_. If you don't see anything useful there, please feel
free to fire an issue report.

.. tip::
   Please don't forget to include the closed issues in your search.
   Sometimes a solution was already reported, and the problem is considered
   **solved**.

New issue reports should include information about your programming environment
(e.g., operating system, Python version) and steps to reproduce the problem.
Please try also to simplify the reproduction steps to a very minimal example
that still illustrates the problem you are facing. By removing other factors,
you help us to identify the root cause of the issue.


Documentation Improvements
==========================

You can help improve ``sushie`` docs by making them more readable and coherent, or
by adding missing information and correcting mistakes.

``sushie`` documentation uses Sphinx_ as its main documentation compiler.
This means that the docs are kept in the same repository as the project code, and
that any documentation update is done in the same way was a code contribution.
``sushie`` uses reStructuredText_ as its principle markup language.

.. tip::
   Please notice that the `GitHub web interface`_ provides a quick way of
   propose changes in ``sushie``'s files. While this mechanism can
   be tricky for normal code contributions, it works perfectly fine for
   contributing to the docs, and can be quite handy.

    If you are interested in trying this method out, please navigate to
    the ``docs`` folder in the source repository_, find which file you
    would like to propose changes and click in the little pencil icon at the
    top, to open `GitHub's code editor`_. Once you finish editing the file,
    please write a message in the form at the bottom of the page describing
    which changes have you made and what are the motivations behind them and
    submit your proposal.

When working on documentation changes in your local machine, you can
build them with the project development dependencies::

    uv sync --extra dev
    uv run make -C docs html

and use Python's built-in web server for a preview in your web browser
(``http://localhost:8000``)::

    python3 -m http.server --directory 'docs/_build/html'

.. important::

   **Building API documentation locally:** The API reference documentation
   requires importing the ``sushie`` package and all its dependencies. If you
   see warnings like ``Failed to import module sushie.infer``, you need to
   install the development environment first::

       uv sync --extra dev
       uv run make -C docs html


Code Contributions
==================

.. todo:: Please include a reference or explanation about the internals of the project.

   An architecture description, design principles or at least a summary of the
   main concepts will make it easy for potential contributors to get started
   quickly.

Submit an issue
---------------

Before you work on any non-trivial code contribution it's best to first create
a report in the `issue tracker`_ to start a discussion on the subject.
This often provides additional considerations and avoids unnecessary work.

Create an environment
---------------------

Before you start coding, create an isolated environment and install the project
with development and testing dependencies. With ``pip``::

    python -m venv .venv
    source .venv/bin/activate
    python -m pip install -U pip
    python -m pip install -e ".[dev,testing]"

With ``uv``::

    uv sync --extra dev --extra testing

Clone the repository
--------------------

#. Create an user account on |the repository service| if you do not already have one.
#. Fork the project repository_: click on the *Fork* button near the top of the
   page. This creates a copy of the code under your account on |the repository service|.
#. Clone this copy to your local disk::

    git clone git@github.com:YourLogin/sushie.git
    cd sushie

#. If you did not already install the development environment, run::

    python -m pip install -e ".[dev,testing]"

   or::

    uv sync --extra dev --extra testing

#. Install |pre-commit|_::

    uv run pre-commit install

   ``sushie`` comes with a lot of hooks configured to automatically help the
   developer to check the code being written.

Implement your changes
----------------------

#. Create a branch to hold your changes::

    git checkout -b my-feature

   and start making changes. Never work on the master branch!

#. Start your work on this branch. Don't forget to add docstrings_ to new
   functions, modules and classes, especially if they are part of public APIs.

#. Add yourself to the list of contributors in ``AUTHORS.rst``.

#. When you’re done editing, do::

    git add <MODIFIED FILES>
    git commit

   to record your changes in git_.

   Please make sure to see the validation messages from |pre-commit|_ and fix
   any eventual issues.
   This project uses Ruff for linting/formatting and ty for type checking.

   .. important:: Don't forget to add unit tests and documentation in case your
      contribution adds an additional feature and is not just a bugfix.

      Moreover, writing a `descriptive commit message`_ is highly recommended.
      In case of doubt, you can check the commit history with::

         git log --graph --decorate --pretty=oneline --abbrev-commit --all

      to look for recurring communication patterns.

#. Please check that your changes don't break any unit tests with::

    uv run --extra testing pytest -p no:capture

   You can also run the main static checks locally::

    uv run ruff check sushie tests data
    uv run ruff format --check sushie tests data
    uv run ty check sushie tests data

Submit your contribution
------------------------

#. If everything works fine, push your local branch to |the repository service| with::

    git push -u origin my-feature

#. Go to the web page of your fork and click |contribute button|
   to send your changes for review.

   Find more detailed information in `creating a PR`_. You might also want to open
   the PR as a draft first and mark it as ready for review after the feedbacks
   from the continuous integration (CI) system or any required fixes.


Troubleshooting
---------------

The following tips can be used when facing problems to build or test the
package:

#. Make sure to fetch all the tags from the upstream repository_.
   The command ``git describe --abbrev=0 --tags`` should return the version you
   are expecting. If you are trying to run CI scripts in a fork repository,
   make sure to push all the tags.
   If versions look wrong, recreate the environment with ``uv sync --reinstall``
   or remove ``.venv`` and run ``uv sync --extra dev --extra testing`` again.

#. `Pytest can drop you`_ in an interactive session in the case an error occurs.
   In order to do that you need to pass a ``--pdb`` option (for example by
   running ``uv run --extra testing pytest -p no:capture -k <TEST NAME> --pdb``).
   You can also setup breakpoints manually instead of using the ``--pdb`` option.


Maintainer tasks
================

Releases
--------

If you are part of the group of maintainers and have correct user permissions
on PyPI_, the following steps can be used to release a new version for
``sushie``:

#. Make sure all unit tests are successful.
#. Tag the current commit on the main branch with a release tag, e.g., ``v1.2.3``.
#. Push the new tag to the upstream repository_, e.g., ``git push upstream v1.2.3``
#. Clean up the ``dist`` and ``build`` folders before building a release.
#. Run ``uv build`` and check that the files in ``dist`` have
   the correct version (no ``.dirty`` or git_ hash) according to the git_ tag.
   Also check the sizes of the distributions, if they are too big (e.g., >
   500KB), unwanted clutter may have been accidentally included.
#. Publish through the configured release workflow or with the PyPI publisher
   used by the maintainers.



.. [#contrib1] Even though, these resources focus on open source projects and
   communities, the general ideas behind collaborating with other developers
   to collectively create software are general and can be applied to all sorts
   of environments, including private companies and proprietary code bases.


.. <-- strart -->
.. |the repository service| replace:: GitHub
.. |contribute button| replace:: "Create pull request"

.. _repository: https://github.com/mancusolab/sushie
.. _issue tracker: https://github.com/mancusolab/sushie/issues
.. <-- end -->


.. |pre-commit| replace:: ``pre-commit``


.. _CommonMark: https://commonmark.org/
.. _contribution-guide.org: http://www.contribution-guide.org/
.. _creating a PR: https://docs.github.com/en/github/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request
.. _descriptive commit message: https://chris.beams.io/posts/git-commit
.. _docstrings: https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html
.. _first-contributions tutorial: https://github.com/firstcontributions/first-contributions
.. _git: https://git-scm.com
.. _GitHub's fork and pull request workflow: https://guides.github.com/activities/forking/
.. _guide created by FreeCodeCamp: https://github.com/FreeCodeCamp/how-to-contribute-to-open-source
.. _MyST: https://myst-parser.readthedocs.io/en/latest/syntax/syntax.html
.. _other kinds of contributions: https://opensource.guide/how-to-contribute
.. _pre-commit: https://pre-commit.com/
.. _PyPI: https://pypi.org/
.. _Pytest can drop you: https://docs.pytest.org/en/stable/usage.html#dropping-to-pdb-python-debugger-at-the-start-of-a-test
.. _Python Software Foundation's Code of Conduct: https://www.python.org/psf/conduct/
.. _reStructuredText: https://www.sphinx-doc.org/en/master/usage/restructuredtext/
.. _Sphinx: https://www.sphinx-doc.org/en/master/
.. _virtual environment: https://realpython.com/python-virtual-environments-a-primer/

.. _GitHub web interface: https://docs.github.com/en/github/managing-files-in-a-repository/managing-files-on-github/editing-files-in-your-repository
.. _GitHub's code editor: https://docs.github.com/en/github/managing-files-in-a-repository/managing-files-on-github/editing-files-in-your-repository
