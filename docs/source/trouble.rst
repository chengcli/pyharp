Troubleshooting
===============

Pyharp should be properly guarded against most common input errors and most
states are printable. However, there are issues that may arise from configuring
radiation bands or from the underlying C++ code, which makes debugging more difficult.

Here are a collection of common issues and how to resolve them:

#. bidir_reflectivity issue.
   If you encounter an issue that looks like this:

   .. code-block:: text

      bidir_reflectivity--surface BDRF model .... not known
        ******* ERROR >>>>>> Existing...

   This is an error emitted by the :class:`pydisort.cpp.Disort` radiative transfer solver.
   It means that you forgot to turn on `lamber` flag in the YAML
   configuration file. To fix this, add the `lamber` flag to your band configuration:

   .. code-block:: yaml

      flags: lamber
